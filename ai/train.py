"""训练策略-价值网络。

损失 = 策略交叉熵（软标签为 MCTS 访问分布）
     + value_weight * 价值 MSE
     + weight_decay * L2

用法：
    python train.py --data ../data/*.npz --epochs 4 --out ../models/iter1.pt
    python train.py --data ../data --init ../models/latest.pt --epochs 2 --lr 3e-4
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model import build_model, count_parameters, load_checkpoint, save_checkpoint
from selfplay import load_dataset


def collect_files(paths: List[str]) -> List[str]:
    files: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            files.extend(sorted(glob.glob(os.path.join(p, "*.npz"))))
            files.extend(sorted(glob.glob(os.path.join(p, "*.jsonl"))))
        else:
            files.extend(sorted(glob.glob(p)))
    return [f for f in files if f.endswith((".npz", ".jsonl"))]


def load_all(files: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    states, pis, zs = [], [], []
    for f in files:
        d = load_dataset(f)
        states.append(d["states"])
        pis.append(d["pis"])
        zs.append(d["z"])
        print(f"  载入 {f}: {len(d['z'])} 条")
    return (
        np.concatenate(states, axis=0).astype(np.float32),
        np.concatenate(pis, axis=0).astype(np.float32),
        np.concatenate(zs, axis=0).astype(np.float32),
    )


class SelfPlayDataset(Dataset):
    """训练时按 8 种对称随机增强（同一变换同时作用于 state 与 pi）。"""

    def __init__(self, states: np.ndarray, pis: np.ndarray, z: np.ndarray, augment: bool = True) -> None:
        self.states = states
        self.pis = pis.reshape(-1, 15, 15)
        self.z = z
        self.augment = augment

    def __len__(self) -> int:
        return len(self.z)

    def __getitem__(self, idx: int):
        s = self.states[idx]
        pi = self.pis[idx]
        if self.augment:
            k = int(np.random.randint(0, 4))
            flip = bool(np.random.randint(0, 2))
            s = np.rot90(s, k, axes=(1, 2))
            pi = np.rot90(pi, k)
            if flip:
                s = s[:, :, ::-1]
                pi = pi[:, ::-1]
        return (
            torch.from_numpy(np.ascontiguousarray(s, dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(pi, dtype=np.float32)).flatten(),
            torch.tensor(self.z[idx], dtype=torch.float32),
        )


def train(args: argparse.Namespace) -> None:
    files = collect_files(args.data)
    if not files:
        raise SystemExit(f"没有找到训练数据: {args.data}")
    print(f"共 {len(files)} 个数据文件")
    states, pis, z = load_all(files)
    print(f"样本总数: {len(z)}  (胜方样本占比 {(z > 0).mean():.3f}, 平均 |z|={np.abs(z).mean():.3f})")

    n = len(z)
    perm = np.random.default_rng(args.seed).permutation(n)
    n_val = max(1, int(n * args.val_ratio))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    train_ds = SelfPlayDataset(states[train_idx], pis[train_idx], z[train_idx], augment=args.augment)
    val_ds = SelfPlayDataset(states[val_idx], pis[val_idx], z[val_idx], augment=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=False,
        pin_memory=args.device.type == "cuda",
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(device=args.device) if not args.init else load_checkpoint(args.init, args.device)
    model.to(args.device)
    print(f"模型参数量: {count_parameters(model):,}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs), eta_min=args.lr * 0.05)

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        tot_p = tot_v = tot_acc = 0.0
        nb = 0
        for s, pi_t, z_t in train_loader:
            s, pi_t, z_t = s.to(args.device), pi_t.to(args.device), z_t.to(args.device)
            logits, value = model(s)
            logp = F.log_softmax(logits, dim=1)
            policy_loss = -(pi_t * logp).sum(1).mean()
            value_loss = F.mse_loss(value, z_t)
            loss = policy_loss + args.value_weight * value_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            opt.step()

            with torch.no_grad():
                acc = ((logits.argmax(1) == pi_t.argmax(1)).float().mean()).item()
            tot_p += policy_loss.item()
            tot_v += value_loss.item()
            tot_acc += acc
            nb += 1

        scheduler.step()
        val_p, val_v, val_acc = evaluate(model, val_loader, args)
        print(
            f"epoch {epoch}/{args.epochs} {time.time() - t0:.0f}s | "
            f"policy {tot_p / nb:.4f} value {tot_v / nb:.4f} acc {tot_acc / nb:.3f} | "
            f"val policy {val_p:.4f} value {val_v:.4f} acc {val_acc:.3f} | lr {opt.param_groups[0]['lr']:.2e}",
            flush=True,
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_checkpoint(args.out, model, iteration=args.iteration, extra={"tag": args.tag, "samples": int(len(z))})
    latest = os.path.join(os.path.dirname(os.path.abspath(args.out)), "latest.pt")
    save_checkpoint(latest, model, iteration=args.iteration, extra={"tag": args.tag, "samples": int(len(z))})
    print(f"模型已保存: {args.out} / {latest}")


@torch.no_grad()
def evaluate(model, loader, args) -> Tuple[float, float, float]:
    model.eval()
    tp = tv = ta = 0.0
    nb = 0
    for s, pi_t, z_t in loader:
        s, pi_t, z_t = s.to(args.device), pi_t.to(args.device), z_t.to(args.device)
        logits, value = model(s)
        tp += (-(pi_t * F.log_softmax(logits, dim=1)).sum(1).mean()).item()
        tv += F.mse_loss(value, z_t).item()
        ta += (logits.argmax(1) == pi_t.argmax(1)).float().mean().item()
        nb += 1
    model.train()
    return tp / max(nb, 1), tv / max(nb, 1), ta / max(nb, 1)


def parse_args() -> argparse.Namespace:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    p = argparse.ArgumentParser(description="训练五子棋策略-价值网络")
    p.add_argument("--data", nargs="+", required=True, help="npz/jsonl 文件或目录")
    p.add_argument("--init", type=str, default=None, help="从已有 checkpoint 继续训练")
    p.add_argument("--out", type=str, default=os.path.join(root, "models", "model.pt"))
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--value-weight", type=float, default=1.0)
    p.add_argument("--clip-grad", type=float, default=5.0)
    p.add_argument("--val-ratio", type=float, default=0.05)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iteration", type=int, default=0)
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    args.augment = not args.no_augment
    def _cuda_usable(min_gib: float = 1.5) -> bool:
        try:
            free, _total = torch.cuda.mem_get_info()
            return free / (1024 ** 3) > min_gib
        except Exception:
            return False

    if args.cpu:
        args.device = torch.device("cpu")
    elif torch.cuda.is_available() and _cuda_usable():
        args.device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        args.device = torch.device("mps")
    else:
        args.device = torch.device("cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    return args


if __name__ == "__main__":
    train(parse_args())
