"""Connect5 后端推理服务（FastAPI）。

后端模式的好处：不受浏览器算力限制，可以用更大的模型（v2：96 通道残差网络，
约 116 万参数）和更深的搜索（默认 400 次模拟，GPU 上约 1 秒）。

启动：
    python server.py --model ../models/v2.pt --sims 400 --port 8000

接口：
    GET  /health           健康检查
    GET  /info             模型架构与参数量
    POST /move             给定局面返回落子（含 policy / value / 搜索信息）
    POST /reload           热切换模型（训练出新版后不用重启服务）
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from gomoku import BOARD_SIZE, NUM_CELLS, Gomoku, moves_to_text, tactical_action
from mcts import MCTS
from model import count_parameters, load_checkpoint, pick_device

app = FastAPI(title="Connect5 AI Server", version="1.0")

# 允许本地文件（file://）或任意前端直接访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATE: Dict = {
    "model": None,
    "device": None,
    "path": None,
    "default_sims": 400,
    "batch_size": 16,
    "amp": True,
}


class MoveRequest(BaseModel):
    board: List[int] = Field(..., description="长度 225 的一维棋盘：0 空 / 1 黑 / 2 白")
    player: int = Field(1, description="当前行棋方：1 黑 / 2 白")
    sims: Optional[int] = Field(None, description="MCTS 模拟次数，缺省用服务默认值")
    temperature: float = Field(0.0, description="0=取访问最多；>0 按分布采样")
    tactical: bool = Field(True, description="是否启用强制手检查（成五/堵五/活四）")
    noise: float = Field(0.0, description="根节点先验噪声，自对弈时才需要")


class MoveResponse(BaseModel):
    action: int
    move: str
    policy: List[float]
    value: float
    sims: int
    timeMs: float
    mode: str
    model: str


def load_model(path: str, device: torch.device = None) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"模型文件不存在: {path}")
    device = device or pick_device()
    model = load_checkpoint(path, device)
    model.eval()
    STATE["model"] = model
    STATE["device"] = device
    STATE["path"] = path


@app.get("/health")
def health():
    return {
        "ok": STATE["model"] is not None,
        "model": STATE["path"],
        "device": str(STATE["device"]) if STATE["device"] else None,
        "default_sims": STATE["default_sims"],
    }


@app.get("/info")
def info():
    model = STATE["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    return {
        "path": STATE["path"],
        "arch": model.arch,
        "parameters": count_parameters(model),
        "device": str(STATE["device"]),
        "default_sims": STATE["default_sims"],
    }


@app.post("/move", response_model=MoveResponse)
def move(req: MoveRequest):
    model = STATE["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    if len(req.board) != NUM_CELLS:
        raise HTTPException(status_code=400, detail=f"棋盘长度应为 {NUM_CELLS}，收到 {len(req.board)}")
    if req.player not in (1, 2):
        raise HTTPException(status_code=400, detail="player 只能是 1（黑）或 2（白）")

    board = np.asarray(req.board, dtype=np.int8).reshape(BOARD_SIZE, BOARD_SIZE)
    game = Gomoku(board=board, player=req.player)
    if game.finished:
        raise HTTPException(status_code=400, detail="对局已结束")

    t0 = time.time()
    name = os.path.basename(STATE["path"] or "")

    # 强制手：成五 / 堵五 / 造活四 / 堵活四，优先于搜索
    if req.tactical:
        forced = tactical_action(game)
        if forced is not None:
            pi = np.zeros(NUM_CELLS, dtype=np.float32)
            pi[forced] = 1.0
            return MoveResponse(
                action=int(forced),
                move=moves_to_text([int(forced)]),
                policy=pi.tolist(),
                value=1.0,
                sims=0,
                timeMs=(time.time() - t0) * 1000,
                mode="tactical",
                model=name,
            )

    sims = int(req.sims or STATE["default_sims"])
    mcts = MCTS(model, STATE["device"], batch_size=STATE["batch_size"], amp=STATE["amp"])
    pi, action, value = mcts.search(
        game,
        sims=sims,
        temperature=req.temperature,
        add_noise=req.noise > 0,
        rng=np.random.default_rng(),
    )
    return MoveResponse(
        action=int(action),
        move=moves_to_text([int(action)]),
        policy=[float(x) for x in pi],
        value=float(value),
        sims=sims,
        timeMs=(time.time() - t0) * 1000,
        mode="mcts",
        model=name,
    )


class ReloadRequest(BaseModel):
    path: str


@app.post("/reload")
def reload(req: ReloadRequest):
    try:
        load_model(req.path, STATE["device"])
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True, "model": STATE["path"], "parameters": count_parameters(STATE["model"])}


def parse_args() -> argparse.Namespace:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    p = argparse.ArgumentParser(description="Connect5 后端推理服务")
    p.add_argument("--model", type=str, default=os.path.join(root, "models", "v2.pt"))
    p.add_argument("--sims", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=16, help="MCTS 叶子批量评估大小（GPU 上可调大）")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no-amp", action="store_true", help="关闭混合精度推理")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device(not args.cpu)
    try:
        load_model(args.model, device)
        print(f"已加载模型: {args.model}")
        print(f"  架构: {STATE['model'].arch}")
        print(f"  参数量: {count_parameters(STATE['model']):,}")
        print(f"  设备: {device}")
    except Exception as e:  # noqa: BLE001
        print(f"模型加载失败（服务仍会启动，/move 会返回 503）: {e}")

    STATE["default_sims"] = args.sims
    STATE["batch_size"] = args.batch_size
    STATE["amp"] = not args.no_amp
    print(f"  混合精度: {'开启' if STATE['amp'] else '关闭'}")

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
