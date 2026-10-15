"""一键迭代训练流水线（AlphaZero 简化版）：

    第 0 轮：启发式自对弈冷启动 -> 训练 -> 导出 -> 评估
    第 N 轮：用上一轮模型 + MCTS 自对弈 -> 在经验回放窗口内训练 -> 导出 -> 评估

    python pipeline.py --iters 6 --games 120 --sims 48 --epochs 3
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from typing import List

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "data")
# 默认跑 v1 系列（浏览器版）；训 v2 用 --model-dir ../models/v2
MODEL_DIR = os.path.join(ROOT, "models", "v1")
WEB_WEIGHTS = os.path.join(ROOT, "web", "models", "weights.json")


def run(cmd: List[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run([sys.executable] + cmd, cwd=HERE, check=True)


def latest_model() -> str:
    return os.path.join(MODEL_DIR, "latest.pt")


def data_files(replay_window: int) -> List[str]:
    files = sorted(
        f for f in os.listdir(DATA_DIR) if f.startswith("iter") and f.endswith(".npz")
    ) if os.path.isdir(DATA_DIR) else []
    files = [os.path.join(DATA_DIR, f) for f in files]
    if replay_window > 0:
        files = files[-replay_window:]
    return files


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="五子棋 AI 迭代训练流水线")
    p.add_argument("--iters", type=int, default=5, help="迭代轮数")
    p.add_argument("--start", type=int, default=0, help="起始轮次（可从已有模型继续）")
    p.add_argument("--games", type=int, default=120, help="每轮自对弈局数")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--sims", type=int, default=48, help="MCTS 模拟次数（自对弈）")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--replay-window", type=int, default=3, help="参与训练的最近数据文件数")
    p.add_argument("--eval-games", type=int, default=20, help="每轮评估局数（0 表示跳过）")
    p.add_argument("--eval-sims", type=int, default=64)
    p.add_argument("--extra-data", nargs="*", default=[], help="额外数据（如人类对局 npz、启发式数据）")
    p.add_argument("--model-dir", type=str, default=MODEL_DIR,
                   help="模型目录，决定跑哪条线：models/v1（浏览器）或 models/v2（后端）")
    p.add_argument("--heuristic-value-weight", type=float, default=0.0,
                   help="自对弈 MCTS 叶子价值中棋型知识的权重：0=纯网络，0.5=知识增强搜索（数据更强但更慢）")
    p.add_argument("--h-tau-open", type=float, default=90.0, help="冷启动启发式的开局温度")
    p.add_argument("--h-tau", type=float, default=45.0, help="冷启动启发式的常规温度（越小越接近贪心）")
    p.add_argument("--games0", type=int, default=0, help="第 0 轮冷启动局数，0 表示与 --games 相同")
    return p.parse_args()


def main() -> None:
    global MODEL_DIR
    args = parse_args()
    MODEL_DIR = args.model_dir
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"模型目录: {MODEL_DIR}", flush=True)

    for it in range(args.start, args.start + args.iters):
        print(f"\n================ 迭代 {it} ================", flush=True)
        t0 = time.time()

        out_data = os.path.join(DATA_DIR, f"iter{it}.npz")
        if it == 0:
            run(["selfplay.py", "--policy", "heuristic",
                 "--games", str(args.games0 or args.games),
                 "--workers", str(args.workers), "--seed", str(it),
                 "--h-tau-open", str(args.h_tau_open), "--h-tau", str(args.h_tau),
                 "--out", out_data])
        else:
            cmd = ["selfplay.py", "--model", latest_model(), "--policy", "mcts",
                   "--games", str(args.games), "--workers", str(args.workers),
                   "--sims", str(args.sims), "--seed", str(it * 1000), "--out", out_data]
            if args.heuristic_value_weight > 0:
                cmd += ["--heuristic-value-weight", str(args.heuristic_value_weight)]
            run(cmd)

        files = data_files(args.replay_window) + list(args.extra_data)
        init = latest_model() if (it > 0 and os.path.exists(latest_model())) else None
        cmd = ["train.py", "--data", *files, "--epochs", str(args.epochs),
               "--batch-size", str(args.batch_size), "--lr", str(args.lr),
               "--iteration", str(it),
               "--out", os.path.join(MODEL_DIR, f"iter{it}.pt")]
        if init:
            cmd += ["--init", init]
        run(cmd)

        run(["export.py", "--model", os.path.join(MODEL_DIR, f"iter{it}.pt"),
             "--out", WEB_WEIGHTS, "--iteration", str(it), "--tag", f"iter{it}"])

        if args.eval_games > 0:
            run(["evaluate.py", "--model", latest_model(), "--vs", "heuristic",
                 "--games", str(args.eval_games), "--sims", str(args.eval_sims), "--seed", str(it)])

        print(f"迭代 {it} 完成，用时 {time.time() - t0:.0f}s", flush=True)

    print("\n全部完成。网页权重:", WEB_WEIGHTS)


if __name__ == "__main__":
    main()
