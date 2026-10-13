"""自对弈数据生成：用当前模型（或启发式基线）自己跟自己下棋，产出训练样本。

输出格式（npz 或 jsonl）：
    state : (3, 15, 15) float32  落子前的局面（相对视角）
    pi    : (225,)     float32  MCTS 访问分布（落子目标）
    z     : float32              该状态行棋方视角的对局结果 (+1 胜 / -1 负 / 0 平)

用法示例：
    # 冷启动（无需模型）：启发式自对弈
    python selfplay.py --policy heuristic --games 2000 --out ../data/iter0.npz

    # 迭代自对弈（使用最新模型 + MCTS）
    python selfplay.py --model ../models/latest.pt --sims 64 --games 200 --out ../data/iter1.npz
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from gomoku import BOARD_SIZE, NUM_CELLS, Gomoku, heuristic_policy, tactical_action
from mcts import MCTS
from model import build_model, load_checkpoint


def _device(prefer_gpu: bool = True) -> torch.device:
    from model import pick_device

    return pick_device(prefer_gpu)


def random_opening(game: Gomoku, rng: np.random.Generator, plies: int = 2) -> None:
    """前若干手在天元附近随机落子，增加开局多样性。"""
    center = BOARD_SIZE // 2
    for _ in range(plies):
        lo, hi = max(0, center - 4), min(BOARD_SIZE - 1, center + 4)
        while True:
            r = int(rng.integers(lo, hi + 1))
            c = int(rng.integers(lo, hi + 1))
            if game.board[r, c] == 0:
                game.play(r * BOARD_SIZE + c)
                break


def play_one(
    model=None,
    device: torch.device = torch.device("cpu"),
    policy: str = "mcts",
    sims: int = 64,
    batch_size: int = 8,
    c_puct: float = 1.5,
    temp_moves: int = 10,
    temp_final: float = 0.1,
    h_tau_open: float = 90.0,
    h_tau: float = 45.0,
    opening_plies: int = 2,
    seed: int = 0,
    max_moves: int = BOARD_SIZE * BOARD_SIZE,
    tactical: bool = True,
    heuristic_weight: float = 0.0,
    heuristic_scale: float = 3000.0,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int], int]:
    """自对弈一局，返回 (states, pis, players, winner)。"""
    rng = np.random.default_rng(seed)
    game = Gomoku()
    if opening_plies > 0:
        random_opening(game, rng, opening_plies)

    mcts = None
    if model is not None and policy == "mcts":
        mcts = MCTS(model, device, c_puct=c_puct, batch_size=batch_size,
                    heuristic_weight=heuristic_weight, heuristic_scale=heuristic_scale)

    states: List[np.ndarray] = []
    pis: List[np.ndarray] = []
    players: List[int] = []

    while not game.finished and game.move_count < max_moves:
        temperature = 1.0 if game.move_count < temp_moves else temp_final
        forced = tactical_action(game) if tactical else None
        if forced is not None:
            pi = np.zeros(NUM_CELLS, dtype=np.float32)
            pi[forced] = 1.0
            action = forced
        elif mcts is not None:
            pi, action, _v = mcts.search(game, sims=sims, temperature=temperature, add_noise=True, rng=rng)
        else:
            tau = h_tau_open if game.move_count < temp_moves else h_tau
            action, pi = heuristic_policy(game, tau=tau, topk=6, rng=rng)

        states.append(game.encode())
        pis.append(pi.astype(np.float32))
        players.append(game.player)
        game.play(action)

    if not game.finished:                      # 超过最大步数：按平局处理
        game.finished = True
        game.winner = 0
    return states, pis, players, game.winner


def _worker(args: Tuple) -> Tuple[List[np.ndarray], List[np.ndarray], List[int], int]:
    (model_path, seed, kwargs, device_str) = args
    # device_str 为 None 时交给 _device() 自动选择；多进程默认给 "cpu"
    device = torch.device(device_str) if device_str else _device()
    model = None
    if model_path and os.path.exists(model_path):
        model = load_checkpoint(model_path, device)
        model.eval()
    return play_one(model=model, device=device, seed=seed, **kwargs)


def generate(
    model_path: Optional[str] = None,
    games: int = 100,
    workers: int = 1,
    seed: int = 0,
    verbose: bool = True,
    device: Optional[torch.device] = None,
    **kwargs,
) -> Dict[str, np.ndarray]:
    # 显式指定 > 多进程默认 CPU（避免多个 CUDA context 争抢）> 单进程自动选择
    if device is not None:
        device_str: Optional[str] = str(device)
    elif workers > 1:
        device_str = "cpu"
    else:
        device_str = None

    if verbose:
        print(f"  自对弈设备: {device_str or '自动（单进程）'}", flush=True)

    tasks = [(model_path, seed + i, kwargs, device_str) for i in range(games)]
    all_states: List[np.ndarray] = []
    all_pis: List[np.ndarray] = []
    all_z: List[float] = []

    t0 = time.time()
    done = 0
    if workers > 1:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(workers) as pool:
            for states, pis, players, winner in pool.imap_unordered(_worker, tasks, chunksize=1):
                done += 1
                _collect(all_states, all_pis, all_z, states, pis, players, winner)
                if verbose and done % max(1, games // 10) == 0:
                    print(f"  {done}/{games} 局, 样本 {len(all_states)}, {time.time() - t0:.0f}s", flush=True)
    else:
        device = device if device is not None else _device()
        model = None
        if model_path and os.path.exists(model_path):
            model = load_checkpoint(model_path, device)
            model.eval()
        kwargs["device"] = device
        for i, task in enumerate(tasks):
            states, pis, players, winner = play_one(model=model, seed=task[1], **kwargs)
            done += 1
            _collect(all_states, all_pis, all_z, states, pis, players, winner)
            if verbose and (done % max(1, games // 10) == 0 or done == games):
                print(f"  {done}/{games} 局, 样本 {len(all_states)}, {time.time() - t0:.0f}s", flush=True)

    return {
        "states": np.asarray(all_states, dtype=np.float32),
        "pis": np.asarray(all_pis, dtype=np.float32),
        "z": np.asarray(all_z, dtype=np.float32),
    }


def _collect(all_states, all_pis, all_z, states, pis, players, winner) -> None:
    for s, pi, p in zip(states, pis, players):
        all_states.append(s)
        all_pis.append(pi)
        if winner == 0:
            all_z.append(0.0)
        else:
            all_z.append(1.0 if winner == p else -1.0)


def save_dataset(path: str, data: Dict[str, np.ndarray]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if path.endswith(".npz"):
        np.savez_compressed(path, **data)
    elif path.endswith(".jsonl"):
        with open(path, "w", encoding="utf-8") as f:
            for s, pi, z in zip(data["states"], data["pis"], data["z"]):
                f.write(
                    '{"state": %s, "pi": %s, "z": %s}\n'
                    % (
                        s.round(4).astype(np.float16).astype(np.float32).tolist().__str__().replace(" ", ""),
                        pi.round(4).astype(np.float16).astype(np.float32).tolist().__str__().replace(" ", ""),
                        float(z),
                    )
                )
    else:
        raise ValueError("输出文件必须是 .npz 或 .jsonl")


def load_dataset(path: str) -> Dict[str, np.ndarray]:
    """加载 npz 或 jsonl 数据集。"""
    if path.endswith(".npz"):
        with np.load(path) as d:
            return {k: d[k] for k in d.files}
    states, pis, zs = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            import json

            obj = json.loads(line)
            states.append(np.asarray(obj["state"], dtype=np.float32))
            pis.append(np.asarray(obj["pi"], dtype=np.float32))
            zs.append(float(obj["z"]))
    return {
        "states": np.asarray(states, dtype=np.float32),
        "pis": np.asarray(pis, dtype=np.float32),
        "z": np.asarray(zs, dtype=np.float32),
    }


def parse_args() -> argparse.Namespace:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    p = argparse.ArgumentParser(description="五子棋自对弈数据生成")
    p.add_argument("--model", type=str, default=None, help="模型 checkpoint；缺省时用启发式/随机策略")
    p.add_argument("--policy", type=str, default="mcts", choices=["mcts", "heuristic"],
                   help="mcts=用模型搜索自对弈；heuristic=规则启发式自对弈（冷启动教材）")
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--sims", type=int, default=64, help="每步 MCTS 模拟次数")
    p.add_argument("--batch-size", type=int, default=8, help="MCTS 叶子批量评估大小")
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--temp-moves", type=int, default=10, help="前 N 手使用 temperature=1 采样")
    p.add_argument("--temp-final", type=float, default=0.1)
    p.add_argument("--h-tau-open", type=float, default=90.0, help="启发式策略开局温度（越小越接近贪心）")
    p.add_argument("--h-tau", type=float, default=45.0, help="启发式策略常规温度")
    p.add_argument("--heuristic-value-weight", type=float, default=0.0,
                   help="MCTS 叶子价值中棋型知识的权重（0=纯网络，0.5 表示知识增强搜索）")
    p.add_argument("--heuristic-scale", type=float, default=3000.0)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                   help="自对弈推理设备。cuda 时用 GPU（建议同时把 --batch-size 调大到 32-64）")
    p.add_argument("--opening-plies", type=int, default=2, help="开局随机手数")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=os.path.join(root, "data", "selfplay.npz"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    kwargs = dict(
        policy=args.policy,
        sims=args.sims,
        batch_size=args.batch_size,
        c_puct=args.c_puct,
        temp_moves=args.temp_moves,
        temp_final=args.temp_final,
        h_tau_open=args.h_tau_open,
        h_tau=args.h_tau,
        opening_plies=args.opening_plies,
        heuristic_weight=args.heuristic_value_weight,
        heuristic_scale=args.heuristic_scale,
    )
    device = None
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    print(f"自对弈: {args.games} 局, 策略={args.policy}, sims={args.sims}, workers={args.workers}, device={args.device}")
    data = generate(model_path=args.model, games=args.games, workers=args.workers, seed=args.seed,
                    device=device, **kwargs)
    save_dataset(args.out, data)
    print(f"已保存 {len(data['z'])} 条样本 -> {args.out}")


if __name__ == "__main__":
    main()
