"""把网页端记录的对局（JSON）转成训练样本。

网页导出的格式见 `web/js/storage.js`：
    {"games": [{"moves": [action...], "winner": 1|2|0, ...}, ...]}

转换规则：每个 (局面, 实际落子) 组成一个样本，policy 为该落子的 one-hot，
value 为该局从「走这一步的玩家」视角看到的胜负。

    python import_games.py --input ../data/human/games.json --out ../data/human.npz --winners-only
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import List

import numpy as np

from gomoku import BOARD_SIZE, NUM_CELLS, Gomoku
from selfplay import save_dataset


def load_games(paths: List[str]) -> List[dict]:
    import json

    games: List[dict] = []
    files: List[str] = []
    for p in paths:
        files.extend(sorted(glob.glob(p)) if not os.path.isdir(p) else sorted(glob.glob(os.path.join(p, "*.json"))))
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            games.extend(data.get("games", []))
        elif isinstance(data, list):
            games.extend(data)
        print(f"  载入 {f}: {len(data.get('games', []) if isinstance(data, dict) else data)} 局")
    return games


def convert(games: List[dict], winners_only: bool = False, min_moves: int = 5, max_moves: int = 225):
    states, pis, zs = [], [], []
    for g in games:
        moves = g.get("moves") or []
        if len(moves) < min_moves or len(moves) > max_moves:
            continue
        winner = g.get("winner")
        if winner is None:                       # 兼容 result 字段
            res = g.get("result")
            color = g.get("playerColor", 1)
            winner = {"win": color, "lose": 3 - color, "draw": 0}.get(res, 0)
        game = Gomoku()
        for action in moves:
            action = int(action)
            player = game.player
            if not game.is_legal(action):
                break
            if winners_only and winner != 0 and player != winner:
                game.play(action)
                continue
            pi = np.zeros(NUM_CELLS, dtype=np.float32)
            pi[action] = 1.0
            states.append(game.encode())
            pis.append(pi)
            zs.append(0.0 if winner == 0 else (1.0 if winner == player else -1.0))
            game.play(action)
            if game.finished:
                break
    return {
        "states": np.asarray(states, dtype=np.float32),
        "pis": np.asarray(pis, dtype=np.float32),
        "z": np.asarray(zs, dtype=np.float32),
    }


def parse_args() -> argparse.Namespace:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    p = argparse.ArgumentParser(description="导入网页对局为训练数据")
    p.add_argument("--input", nargs="+", required=True, help="对局 JSON 文件或目录")
    p.add_argument("--out", type=str, default=os.path.join(root, "data", "human.npz"))
    p.add_argument("--winners-only", action="store_true", help="只保留获胜方的落子作为正样本")
    p.add_argument("--min-moves", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    games = load_games(args.input)
    data = convert(games, winners_only=args.winners_only, min_moves=args.min_moves)
    save_dataset(args.out, data)
    print(f"已转换 {len(games)} 局 -> {len(data['z'])} 条样本: {args.out}")


if __name__ == "__main__":
    main()
