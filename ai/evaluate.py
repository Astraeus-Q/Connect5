"""模型评估：让模型与基线（随机 / 启发式贪心 / 另一个模型）对弈，统计胜率。

    # 最新模型 vs 启发式贪心，各 50 局
    python evaluate.py --model ../models/latest.pt --vs heuristic --games 50 --sims 64

    # 新模型 vs 上一代模型
    python evaluate.py --model ../models/iter3.pt --vs mcts --vs-model ../models/iter2.pt --games 40
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Optional

import numpy as np
import torch

from gomoku import Gomoku, heuristic_policy, random_policy, tactical_action
from mcts import MCTS
from model import load_checkpoint


class Player:
    def __init__(self, kind: str, name: str, model=None, device=None, sims: int = 64, tau: float = 0.0) -> None:
        self.kind = kind
        self.name = name
        self.model = model
        self.device = device or torch.device("cpu")
        self.sims = sims
        self.tau = tau
        self._mcts = MCTS(model, self.device) if (kind == "mcts" and model is not None) else None

    def move(self, game: Gomoku, rng: np.random.Generator) -> int:
        if self.kind in ("mcts", "network"):
            forced = tactical_action(game)
            if forced is not None:
                return forced
        if self.kind == "random":
            return random_policy(game, rng)[0]
        if self.kind == "heuristic":
            return heuristic_policy(game, tau=self.tau, topk=6, rng=rng)[0]
        if self.kind == "mcts" and self._mcts is not None:
            _pi, action, _v = self._mcts.search(game, sims=self.sims, temperature=0.0, add_noise=False, rng=rng)
            return action
        if self.model is not None:                      # 纯网络策略（无搜索）
            pi, _v = self.model.predict(game.encode(), self.device)
            return int(np.argmax(pi))
        return random_policy(game, rng)[0]

    def reset(self) -> None:
        pass


def play_game(black: Player, white: Player, rng: np.random.Generator, opening_plies: int = 2) -> int:
    game = Gomoku()
    center = 7
    for _ in range(opening_plies):                      # 开局随机化，避免每局雷同
        while True:
            r = int(rng.integers(center - 4, center + 5))
            c = int(rng.integers(center - 4, center + 5))
            if game.board[r, c] == 0:
                game.play(r * 15 + c)
                break
    while not game.finished:
        player = black if game.player == 1 else white
        game.play(player.move(game, rng))
    return game.winner


def match(p1: Player, p2: Player, games: int = 40, seed: int = 0, opening_plies: int = 2) -> dict:
    rng = np.random.default_rng(seed)
    w = l = d = 0
    t0 = time.time()
    for i in range(games):
        if i % 2 == 0:
            winner = play_game(p1, p2, rng, opening_plies)
            p1_is_black = True
        else:
            winner = play_game(p2, p1, rng, opening_plies)
            p1_is_black = False
        if winner == 0:
            d += 1
        elif (winner == 1) == p1_is_black:
            w += 1
        else:
            l += 1
        if (i + 1) % max(1, games // 5) == 0:
            print(f"  {i + 1}/{games} 局  胜{w} 负{l} 平{d}  ({time.time() - t0:.0f}s)", flush=True)
    n = max(1, w + l + d)
    return {"win": w, "lose": l, "draw": d, "win_rate": w / max(1, w + l), "score": (w + 0.5 * d) / n}


def make_player(kind: str, model_path: Optional[str], device, sims: int, tau: float, name: str) -> Player:
    model = None
    if model_path and os.path.exists(model_path):
        model = load_checkpoint(model_path, device)
        model.eval()
    return Player(kind, name or kind, model=model, device=device, sims=sims, tau=tau)


def parse_args() -> argparse.Namespace:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    p = argparse.ArgumentParser(description="评估五子棋模型棋力")
    p.add_argument("--model", type=str, default=os.path.join(root, "models", "latest.pt"))
    p.add_argument("--vs", type=str, default="heuristic", choices=["random", "heuristic", "network", "mcts"])
    p.add_argument("--vs-model", type=str, default=None, help="对手模型（vs=network/mcts 时必填）")
    p.add_argument("--kind", type=str, default="mcts", choices=["mcts", "network"], help="被测模型的行棋方式")
    p.add_argument("--games", type=int, default=40)
    p.add_argument("--sims", type=int, default=64)
    p.add_argument("--tau", type=float, default=0.0, help="启发式对手温度（0=完全贪心）")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from model import pick_device

    device = pick_device(not args.cpu)

    p1 = make_player(args.kind, args.model, device, args.sims, 0.0, os.path.basename(args.model))
    p2 = make_player(args.vs, args.vs_model, device, args.sims, args.tau, args.vs)
    print(f"对局: {p1.name}({p1.kind}) vs {p2.name}({p2.kind}), {args.games} 局, sims={args.sims}, device={device}")
    res = match(p1, p2, games=args.games, seed=args.seed)
    print(
        f"结果: 胜 {res['win']} / 负 {res['lose']} / 平 {res['draw']}  "
        f"胜率 {res['win_rate']:.3f}  得分率 {res['score']:.3f}"
    )


if __name__ == "__main__":
    main()
