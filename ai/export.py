"""把训练好的 checkpoint 导出为网页端可直接加载的权重（BN 已融合进卷积）。

    python export.py --model ../models/latest.pt --out ../web/models/weights.json
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from model import export_weights_json, load_checkpoint


def parse_args() -> argparse.Namespace:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    p = argparse.ArgumentParser(description="导出网页端权重")
    p.add_argument("--model", type=str, default=os.path.join(root, "models", "latest.pt"))
    p.add_argument("--out", type=str, default=os.path.join(root, "web", "models", "weights.json"))
    p.add_argument("--iteration", type=int, default=0)
    p.add_argument("--tag", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model = load_checkpoint(args.model, torch.device("cpu"))
    model.eval()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    meta = {"iteration": args.iteration, "tag": args.tag, "source": os.path.basename(args.model)}
    export_weights_json(model, args.out, meta=meta)
    size_mb = os.path.getsize(args.out) / 1024 / 1024
    print(f"已导出 {args.out} ({size_mb:.2f} MB), 架构: {model.arch}")


if __name__ == "__main__":
    main()
