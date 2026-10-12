"""PUCT 蒙特卡洛树搜索。

特点：
* 叶子节点先挂起、攒够 `batch_size` 个再一次性送入网络，避免 batch=1 的低效推理；
* 挂起期间施加虚拟损失，减少多条模拟路径撞到同一个叶子；
* 支持 Dirichlet 根噪声与温度退火（AlphaZero 风格的自对弈探索）。
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from gomoku import BOARD_SIZE, NUM_CELLS, Gomoku, heuristic_value, is_five


class Node:
    __slots__ = ("board", "player", "pi", "legal_actions", "children", "N", "W", "pending", "terminal", "prior")

    def __init__(self, board: np.ndarray, player: int, terminal: Optional[float] = None) -> None:
        self.board = board
        self.player = player
        self.pi: Optional[np.ndarray] = None          # 本节点的落子先验（网络输出）
        self.legal_actions: np.ndarray = np.flatnonzero(board.reshape(-1) == 0)
        self.children: Dict[int, "Node"] = {}
        self.N = 0
        self.W = 0.0
        self.pending = 0
        self.terminal = terminal                       # 终局节点：从「本节点行棋方」视角的价值
        self.prior = 0.0

    def state(self) -> np.ndarray:
        """网络输入：(自己, 对手, 行棋方常数平面)。"""
        s = np.zeros((3, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        s[0] = self.board == self.player
        s[1] = self.board == (3 - self.player)
        s[2] = 1.0 if self.player == 1 else 0.0
        return s

    def make_child(self, action: int, prior: float) -> "Node":
        r, c = divmod(action, BOARD_SIZE)
        board = self.board.copy()
        board[r, c] = self.player
        child_player = 3 - self.player
        if is_five(board, r, c, self.player):
            child = Node(board, child_player, terminal=-1.0)   # 轮到对手走，但对手已输
        elif not (board == 0).any():
            child = Node(board, child_player, terminal=0.0)    # 平局
        else:
            child = Node(board, child_player)
        child.prior = prior
        self.children[action] = child
        return child

    def select(self, c_puct: float, virtual_loss: float) -> int:
        children = self.children
        n_sum = 0
        for ch in children.values():
            n_sum += ch.N + ch.pending
        sqrt_sum = math.sqrt(n_sum + 1e-8)
        pi = self.pi
        best_score, best_action = -1e18, -1
        for a in self.legal_actions:
            ch = children.get(int(a))
            if ch is None:
                n = 0
                q = 0.0
            elif ch.terminal is not None:
                n = ch.N
                q = -ch.terminal            # 父节点视角
            else:
                n = ch.N + ch.pending
                q = (ch.W - virtual_loss * ch.pending) / n if n > 0 else 0.0
            score = q + c_puct * pi[int(a)] * sqrt_sum / (1 + n)
            if score > best_score:
                best_score, best_action = score, int(a)
        return best_action


class MCTS:
    def __init__(
        self,
        model=None,
        device=None,
        c_puct: float = 1.5,
        batch_size: int = 8,
        virtual_loss: float = 1.0,
        dirichlet_alpha: float = 0.05,
        dirichlet_eps: float = 0.25,
        heuristic_weight: float = 0.0,
        heuristic_scale: float = 3000.0,
    ) -> None:
        self.model = model
        self.device = device or torch.device("cpu")
        self.c_puct = c_puct
        self.batch_size = batch_size
        self.virtual_loss = virtual_loss
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        # 仅用于自对弈：把棋型知识混入叶子价值，让搜索产生更高质量的训练数据
        self.heuristic_weight = heuristic_weight
        self.heuristic_scale = heuristic_scale

        self._pending: List[Tuple[Node, List[Tuple[Node, int, Node]]]] = []
        self._root: Optional[Node] = None

    # ------------------------------------------------------------------ 网络推理
    @torch.no_grad()
    def _evaluate(self, nodes: List[Node]) -> Tuple[np.ndarray, np.ndarray]:
        states = np.stack([n.state() for n in nodes])
        x = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        logits, values = self.model(x)
        legal = (x[:, 0] + x[:, 1]).flatten(1) == 0
        masked = logits.masked_fill(~legal, -1e9)
        pi = torch.softmax(masked, dim=1).cpu().numpy().astype(np.float32)
        values = values.cpu().numpy().astype(np.float32)
        if self.heuristic_weight > 0:
            for i, node in enumerate(nodes):
                hv = heuristic_value(node.board, node.player, self.heuristic_scale)
                values[i] = (1 - self.heuristic_weight) * values[i] + self.heuristic_weight * hv
        return pi, values

    # ------------------------------------------------------------------ 搜索主体
    def search(
        self,
        game: Gomoku,
        sims: int = 64,
        temperature: float = 1.0,
        add_noise: bool = True,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, int, float]:
        """返回 (pi, action, root_value)。root_value 为当前行棋方视角的价值估计。"""
        root = Node(game.board.copy(), game.player)
        self._root = root

        # 根节点先评估（同时施加探索噪声）
        self._pending.append((root, []))
        self._flush()
        if add_noise:
            rng = rng or np.random.default_rng()
            noise = rng.dirichlet(np.full(len(root.legal_actions), self.dirichlet_alpha)).astype(np.float32)
            mixed = (1 - self.dirichlet_eps) * root.pi.copy()
            mixed[root.legal_actions] += self.dirichlet_eps * noise
            root.pi = mixed / mixed.sum()

        for _ in range(int(sims)):
            self._simulate(root)
            if len(self._pending) >= self.batch_size:
                self._flush()
        if self._pending:
            self._flush()

        pi = self._visit_distribution(root, temperature)
        if temperature is not None and temperature <= 1e-6:
            action = int(np.argmax(pi))
        else:
            rng = rng or np.random.default_rng()
            pi_sum = pi.sum()
            action = int(rng.choice(NUM_CELLS, p=pi / pi_sum if pi_sum > 0 else None))
        value = root.W / root.N if root.N > 0 else 0.0
        return pi, action, float(value)

    def _simulate(self, root: Node) -> None:
        node = root
        path: List[Tuple[Node, int, Node]] = []
        while True:
            if node.terminal is not None:
                node.N += 1
                self._backprop(path, node.terminal)
                return
            if node.pi is None:
                if node.pending > 0:
                    return                      # 已在评估队列中，放弃本次模拟
                node.pending += 1
                self._pending.append((node, path))
                return
            action = node.select(self.c_puct, self.virtual_loss)
            child = node.children.get(action) or node.make_child(action, float(node.pi[action]))
            path.append((node, action, child))
            node = child

    def _backprop(self, path: List[Tuple[Node, int, Node]], leaf_value: float) -> None:
        v = leaf_value
        for parent, _a, _child in reversed(path):
            parent.N += 1
            parent.W += -v
            v = -v

    def _flush(self) -> None:
        if not self._pending:
            return
        nodes = [item[0] for item in self._pending]
        pis, values = self._evaluate(nodes)
        for (node, path), pi, value in zip(self._pending, pis, values):
            node.pi = pi
            node.pending -= 1
            node.N += 1
            node.W += float(value)
            self._backprop(path, float(value))
        self._pending.clear()

    @staticmethod
    def _visit_distribution(root: Node, temperature: float) -> np.ndarray:
        visits = np.zeros(NUM_CELLS, dtype=np.float32)
        for a, ch in root.children.items():
            visits[a] = ch.N
        if temperature is not None and temperature > 1e-6:
            visits = visits ** (1.0 / temperature)
        total = visits.sum()
        if total <= 0:                                  # 极端情况：回退到先验
            return root.pi.copy()
        return visits / total

