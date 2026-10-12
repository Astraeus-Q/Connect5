"""五子棋策略-价值网络（PyTorch）。

结构：
    输入 (3, 15, 15) -> 卷积主干 -> 两个头
    policy head: 225 维落子 logits
    value  head: 当前行棋方视角的胜率估计，tanh 输出到 [-1, 1]

卷积后接 BatchNorm，推理时可把 BN 融合进卷积权重（`fused_weights`），
网页端只需实现 conv + relu + 全连接即可复现完全相同的输出。
"""

from __future__ import annotations

import json
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gomoku import BOARD_SIZE, NUM_CELLS


class ConvBN(nn.Module):
    """conv(无 bias) + batchnorm + relu，便于推理时融合。"""

    def __init__(self, c_in: int, c_out: int, kernel: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel, padding=kernel // 2, bias=False)
        self.bn = nn.BatchNorm2d(c_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))


class GomokuNet(nn.Module):
    def __init__(self, channels: int = 48, n_blocks: int = 4, size: int = BOARD_SIZE, in_channels: int = 3) -> None:
        super().__init__()
        self.channels = channels
        self.n_blocks = n_blocks
        self.size = size
        self.in_channels = in_channels

        self.stem = ConvBN(in_channels, channels, 3)
        self.blocks = nn.ModuleList([ConvBN(channels, channels, 3) for _ in range(n_blocks)])

        hidden = size * size
        # policy head
        self.p_conv = ConvBN(channels, 2, 1)
        self.p_fc = nn.Linear(2 * hidden, hidden)
        # value head
        self.v_conv = ConvBN(channels, 1, 1)
        self.v_fc1 = nn.Linear(hidden, 64)
        self.v_fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.stem(x)
        for blk in self.blocks:
            h = blk(h)

        p = self.p_conv(h)
        p = self.p_fc(p.flatten(1))

        v = self.v_conv(h)
        v = F.relu(self.v_fc1(v.flatten(1)))
        v = torch.tanh(self.v_fc2(v))
        return p, v.squeeze(-1)

    def predict(self, state: np.ndarray, device: torch.device = None) -> Tuple[np.ndarray, float]:
        """单局面推理，返回 (masked policy 概率, value)。"""
        device = device or next(self.parameters()).device
        was_training = self.training
        self.eval()
        with torch.no_grad():
            x = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            logits, value = self.forward(x)
            legal = torch.as_tensor((state[0] + state[1]) == 0, dtype=torch.float32, device=device).flatten()
            masked = torch.where(legal > 0.5, logits, torch.full_like(logits, -1e9)).flatten()
            pi = torch.softmax(masked, dim=0).cpu().numpy()
        if was_training:
            self.train()
        return pi, float(value.item())

    @property
    def arch(self) -> Dict:
        return {
            "size": self.size,
            "channels": self.channels,
            "n_blocks": self.n_blocks,
            "in_channels": self.in_channels,
        }


def pick_device(prefer_gpu: bool = True, min_free_gib: float = 1.5) -> torch.device:
    """选择可用设备；显存不足时自动回退 CPU（避免共享显存机器上 OOM）。"""
    if not prefer_gpu:
        return torch.device("cpu")
    try:
        if torch.cuda.is_available():
            free, _total = torch.cuda.mem_get_info()
            if free / (1024 ** 3) > min_free_gib:
                return torch.device("cuda")
    except Exception:
        pass
    try:
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
    except Exception:
        pass
    return torch.device("cpu")


def build_model(arch: Dict = None, device: torch.device = None) -> GomokuNet:
    arch = arch or {}
    model = GomokuNet(
        channels=int(arch.get("channels", 48)),
        n_blocks=int(arch.get("n_blocks", 4)),
        size=int(arch.get("size", BOARD_SIZE)),
        in_channels=int(arch.get("in_channels", 3)),
    )
    if device is not None:
        model.to(device)
    return model


# ------------------------------------------------------------------ BN 融合导出
def _fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> Tuple[np.ndarray, np.ndarray]:
    """把 BatchNorm 的统计量折进卷积的 weight/bias。"""
    w = conv.weight.detach().cpu().numpy().astype(np.float64)
    gamma = bn.weight.detach().cpu().numpy().astype(np.float64)
    beta = bn.bias.detach().cpu().numpy().astype(np.float64)
    mean = bn.running_mean.detach().cpu().numpy().astype(np.float64)
    var = bn.running_var.detach().cpu().numpy().astype(np.float64)
    eps = float(bn.eps)

    scale = gamma / np.sqrt(var + eps)
    w_fused = w * scale.reshape(-1, 1, 1, 1)
    b_fused = beta - mean * scale
    return w_fused.astype(np.float32), b_fused.astype(np.float32)


def fused_weights(model: GomokuNet) -> Tuple[Dict, Dict[str, np.ndarray]]:
    """返回 (arch, 融合后的权重字典)，权重为 float32 numpy 数组。"""
    weights: Dict[str, np.ndarray] = {}

    def put(prefix: str, block: ConvBN) -> None:
        w, b = _fuse_conv_bn(block.conv, block.bn)
        weights[f"{prefix}.weight"] = w
        weights[f"{prefix}.bias"] = b

    put("stem", model.stem)
    for i, blk in enumerate(model.blocks):
        put(f"block{i}", blk)
    put("p_conv", model.p_conv)
    put("v_conv", model.v_conv)

    weights["p_fc.weight"] = model.p_fc.weight.detach().cpu().numpy().astype(np.float32)
    weights["p_fc.bias"] = model.p_fc.bias.detach().cpu().numpy().astype(np.float32)
    weights["v_fc1.weight"] = model.v_fc1.weight.detach().cpu().numpy().astype(np.float32)
    weights["v_fc1.bias"] = model.v_fc1.bias.detach().cpu().numpy().astype(np.float32)
    weights["v_fc2.weight"] = model.v_fc2.weight.detach().cpu().numpy().astype(np.float32)
    weights["v_fc2.bias"] = model.v_fc2.bias.detach().cpu().numpy().astype(np.float32)

    return model.arch, weights


# ------------------------------------------------------------------ checkpoint
def save_checkpoint(path: str, model: GomokuNet, iteration: int = 0, extra: Dict = None) -> None:
    payload = {
        "arch": model.arch,
        "state_dict": model.state_dict(),
        "iteration": iteration,
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_checkpoint(path: str, device: torch.device = None) -> GomokuNet:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ckpt.get("arch", {}))
    model.load_state_dict(ckpt["state_dict"])
    if device is not None:
        model.to(device)
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _enc(arr: np.ndarray) -> str:
    import base64

    return base64.b64encode(np.ascontiguousarray(arr, dtype=np.float32).tobytes()).decode("ascii")


def export_weights_json(model: GomokuNet, path: str, meta: Dict = None) -> Dict:
    """导出浏览器可直接加载的权重文件（base64 编码的 float32）。"""
    arch, weights = fused_weights(model)
    payload = {
        "format": "gomoku-weights-v1",
        "arch": arch,
        "meta": meta or {},
        "weights": {k: _enc(v) for k, v in weights.items()},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return payload


def load_weights_json(path: str) -> Tuple[Dict, Dict[str, np.ndarray]]:
    import base64

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    weights = {
        k: np.frombuffer(base64.b64decode(v), dtype=np.float32)
        for k, v in payload["weights"].items()
    }
    return payload["arch"], weights


if __name__ == "__main__":
    m = GomokuNet()
    print(f"参数量: {count_parameters(m):,}")
    arch, w = fused_weights(m)
    print("融合后张量:", {k: tuple(v.shape) for k, v in list(w.items())[:4]})
