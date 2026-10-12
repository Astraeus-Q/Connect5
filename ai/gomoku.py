"""五子棋规则引擎：15x15 棋盘，无禁手，横/竖/斜任一方向先连成五子者获胜。

该模块同时被自对弈（Python）和棋谱校验使用；网页端 `web/js/rules.js`
实现了完全相同的规则，保证训练数据与线上对局语义一致。
"""

from __future__ import annotations

from typing import Iterator, List, Optional, Tuple

import numpy as np

BOARD_SIZE = 15
NUM_CELLS = BOARD_SIZE * BOARD_SIZE
WIN_LEN = 5

BLACK, WHITE = 1, 2

DIRS: Tuple[Tuple[int, int], ...] = ((1, 0), (0, 1), (1, 1), (1, -1))

# 棋型评分表：(连子数, 两端空位数) -> 分数。用于启发式基线 AI 与冷启动数据生成。
_PATTERN_SCORE = {
    (5, 0): 10_000_000, (5, 1): 10_000_000, (5, 2): 10_000_000,
    (4, 2): 100_000, (4, 1): 10_000, (4, 0): 0,
    (3, 2): 8_000, (3, 1): 800, (3, 0): 0,
    (2, 2): 600, (2, 1): 60, (2, 0): 0,
    (1, 2): 40, (1, 1): 6, (1, 0): 0,
}


def is_five(board: np.ndarray, r: int, c: int, color: int) -> bool:
    """判断 (r, c) 处的 color 棋子是否构成五连（假定该点已落子）。"""
    for dr, dc in DIRS:
        cnt = 1
        for sign in (1, -1):
            rr, cc = r + dr * sign, c + dc * sign
            while 0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE and board[rr, cc] == color:
                cnt += 1
                rr += dr * sign
                cc += dc * sign
        if cnt >= WIN_LEN:
            return True
    return False


class Gomoku:
    """一局五子棋的状态。board: 0 空 / 1 黑 / 2 白，黑先。"""

    __slots__ = ("board", "player", "last_move", "winner", "finished")

    def __init__(
        self,
        board: Optional[np.ndarray] = None,
        player: int = BLACK,
        last_move: int = -1,
        winner: int = 0,
        finished: bool = False,
    ) -> None:
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8) if board is None else board
        self.player = player
        self.last_move = last_move
        self.winner = winner
        self.finished = finished

    # ---------------------------------------------------------------- 基础操作
    def clone(self) -> "Gomoku":
        return Gomoku(self.board.copy(), self.player, self.last_move, self.winner, self.finished)

    @property
    def move_count(self) -> int:
        return int(np.count_nonzero(self.board))

    def legal_moves(self) -> np.ndarray:
        """返回 (225,) 的布尔掩码。"""
        return self.board.reshape(-1) == 0

    def is_legal(self, action: int) -> bool:
        r, c = divmod(action, BOARD_SIZE)
        return self.board[r, c] == 0

    def play(self, action: int) -> "Gomoku":
        """在 action 处落子并切换行棋方；返回自身便于链式调用。"""
        r, c = divmod(action, BOARD_SIZE)
        if self.board[r, c] != 0:
            raise ValueError(f"非法落子: {action} -> ({r}, {c})")
        self.board[r, c] = self.player
        if self._check_win(r, c, self.player):
            self.winner = self.player
            self.finished = True
        elif not (self.board == 0).any():
            self.winner = 0
            self.finished = True
        else:
            self.player = 3 - self.player
        self.last_move = action
        return self

    def _check_win(self, r: int, c: int, color: int) -> bool:
        for dr, dc in DIRS:
            cnt = 1
            for sign in (1, -1):
                rr, cc = r, c
                while True:
                    rr += dr * sign
                    cc += dc * sign
                    if 0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE and self.board[rr, cc] == color:
                        cnt += 1
                    else:
                        break
            if cnt >= WIN_LEN:
                return True
        return False

    def result(self, player: Optional[int] = None) -> float:
        """从 player 视角看的结果：胜 1 / 负 -1 / 平或未结束 0。"""
        if not self.finished:
            return 0.0
        if self.winner == 0:
            return 0.0
        p = self.player if player is None else player
        return 1.0 if self.winner == p else -1.0

    def key(self) -> bytes:
        return self.board.tobytes() + bytes([self.player])

    # ---------------------------------------------------------------- 特征编码
    def encode(self, player: Optional[int] = None) -> np.ndarray:
        """编码为 (3, 15, 15) 的浮点张量，采用「相对视角」。

        通道 0：当前行棋方的棋子
        通道 1：对手的棋子
        通道 2：常数平面（轮到黑棋为 1，否则为 0）
        """
        p = self.player if player is None else player
        s = np.zeros((3, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        s[0] = self.board == p
        s[1] = self.board == (3 - p)
        s[2] = 1.0 if p == BLACK else 0.0
        return s

    def __str__(self) -> str:
        chars = {0: ".", 1: "X", 2: "O"}
        rows = [" ".join(chars[v] for v in row) for row in self.board]
        return "\n".join(rows)


# -------------------------------------------------------------------- 数据增强
def augment(state: np.ndarray, pi: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """枚举 8 种对称（4 次旋转 x 翻转），同步变换 state 与 policy。"""
    p = pi.reshape(BOARD_SIZE, BOARD_SIZE)
    for k in range(4):
        for flip in (False, True):
            sb = np.rot90(state, k, axes=(1, 2))
            pb = np.rot90(p, k)
            if flip:
                sb = sb[:, :, ::-1]
                pb = pb[:, ::-1]
            yield np.ascontiguousarray(sb), np.ascontiguousarray(pb).reshape(-1)


def random_augment(state: np.ndarray, pi: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    k = int(rng.integers(0, 4))
    flip = bool(rng.integers(0, 2))
    sb = np.rot90(state, k, axes=(1, 2))
    pb = np.rot90(pi.reshape(BOARD_SIZE, BOARD_SIZE), k)
    if flip:
        sb = sb[:, :, ::-1]
        pb = pb[:, ::-1]
    return np.ascontiguousarray(sb), np.ascontiguousarray(pb).reshape(-1).astype(np.float32)


# -------------------------------------------------------------------- 启发式基线
def evaluate_point(board: np.ndarray, r: int, c: int, color: int) -> float:
    """假设 (r, c) 已落下 color，评估该点形成的棋型价值（不考虑跳子）。"""
    total = 0.0
    for dr, dc in DIRS:
        cnt = 1
        openings = 0
        for sign in (1, -1):
            rr, cc = r + dr * sign, c + dc * sign
            while 0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE and board[rr, cc] == color:
                cnt += 1
                rr += dr * sign
                cc += dc * sign
            if 0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE and board[rr, cc] == 0:
                openings += 1
        cnt = min(cnt, 5)
        total += _PATTERN_SCORE.get((cnt, openings), 0.0)
    return total


def _candidate_mask(board: np.ndarray, radius: int = 2) -> np.ndarray:
    occupied = board != 0
    mask = np.zeros_like(occupied)
    if not occupied.any():
        mask[BOARD_SIZE // 2, BOARD_SIZE // 2] = True
        return mask
    for r, c in np.argwhere(occupied):
        r0, r1 = max(0, r - radius), min(BOARD_SIZE, r + radius + 1)
        c0, c1 = max(0, c - radius), min(BOARD_SIZE, c + radius + 1)
        mask[r0:r1, c0:c1] = True
    return mask & ~occupied


def heuristic_scores(board: np.ndarray, player: int, defense: float = 0.9) -> np.ndarray:
    """返回每个空位的启发式分数（进攻分 + defense * 防守分）。"""
    opp = 3 - player
    scores = np.zeros(NUM_CELLS, dtype=np.float32)
    for r, c in np.argwhere(_candidate_mask(board)):
        attack = evaluate_point(board, r, c, player)
        block = evaluate_point(board, r, c, opp)
        scores[r * BOARD_SIZE + c] = attack + defense * block
    return scores


def heuristic_policy(
    game: Gomoku,
    tau: float = 200.0,
    topk: int = 6,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[int, np.ndarray]:
    """启发式策略：在分数最高的 topk 个候选中按 softmax 采样。

    返回 (action, pi)，pi 可用于训练目标或作为 MCTS 的先验。
    """
    scores = heuristic_scores(game.board, game.player)
    scores[~game.legal_moves()] = -1e18
    idx = np.argsort(-scores)[:topk]
    vals = scores[idx]
    if tau <= 1e-6:
        action = int(idx[0])
        pi = np.zeros(NUM_CELLS, dtype=np.float32)
        pi[action] = 1.0
        return action, pi
    p = np.exp((vals - vals.max()) / tau)
    p /= p.sum()
    action = int(idx[0]) if rng is None else int(rng.choice(idx, p=p))
    pi = np.zeros(NUM_CELLS, dtype=np.float32)
    pi[idx] = p.astype(np.float32)
    return action, pi


def threat_score(board: np.ndarray, color: int) -> float:
    """color 方当前最大的进攻威胁分（活四/活三/冲四等的粗略量度）。"""
    scores = heuristic_scores(board, color, defense=0.0)
    return float(scores.max()) if scores.size else 0.0


def heuristic_value(board: np.ndarray, player: int, scale: float = 3000.0) -> float:
    """棋型价值：己方最大威胁与对手最大威胁之差，压缩到 [-1, 1]。

    只在自对弈时作为 MCTS 叶子价值的辅助项（知识增强搜索），
    用于生成更高质量的对局数据；对局/网页端推理不依赖它。
    """
    mine = threat_score(board, player)
    opp = threat_score(board, 3 - player)
    return float(np.tanh((mine - opp) / scale))


def random_policy(game: Gomoku, rng: np.random.Generator) -> Tuple[int, np.ndarray]:
    legal = np.flatnonzero(game.legal_moves())
    action = int(rng.choice(legal))
    pi = np.zeros(NUM_CELLS, dtype=np.float32)
    pi[legal] = 1.0 / len(legal)
    return action, pi


def winning_moves(board: np.ndarray, color: int) -> List[int]:
    """返回 color 落子后能立即连成五的所有空点。"""
    out: List[int] = []
    for r, c in np.argwhere(_candidate_mask(board, radius=4)):
        if is_five(board, r, c, color):
            out.append(int(r * BOARD_SIZE + c))
    return out


def open_four_moves(board: np.ndarray, color: int) -> List[int]:
    """返回 color 落子后能形成「活四」（四连且两端皆空，对手挡不住）的点。"""
    out: List[int] = []
    for r, c in np.argwhere(_candidate_mask(board, radius=4)):
        for dr, dc in DIRS:
            cnt = 1
            openings = 0
            for sign in (1, -1):
                rr, cc = r + dr * sign, c + dc * sign
                while 0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE and board[rr, cc] == color:
                    cnt += 1
                    rr += dr * sign
                    cc += dc * sign
                if 0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE and board[rr, cc] == 0:
                    openings += 1
            if cnt == 4 and openings == 2:
                out.append(int(r * BOARD_SIZE + c))
                break
    return out


def tactical_action(game: "Gomoku") -> Optional[int]:
    """强制手检查，优先级：

    1. 己方能一手成五    -> 直接取胜
    2. 对手能一手成五    -> 必须封堵
    3. 己方能造活四      -> 必胜威胁（此时对手已无一手杀）
    4. 对手能造活四      -> 必须封堵

    搜索类玩家容易因探索项漏掉这类强制手，所以在所有策略之前先做检查。
    """
    board = game.board
    me, opp = game.player, 3 - game.player

    mine = winning_moves(board, me)
    if mine:
        return mine[0]
    theirs = winning_moves(board, opp)
    if theirs:
        return theirs[0]

    my_fours = open_four_moves(board, me)
    if my_fours:
        return my_fours[0]
    their_fours = open_four_moves(board, opp)
    if their_fours:
        return their_fours[0]
    return None


def moves_to_text(moves: List[int]) -> str:
    """把落子序列转成 `h8,i9,...` 形式的棋谱文本（列用字母，行用数字）。"""
    return ",".join(f"{chr(ord('a') + a % BOARD_SIZE)}{a // BOARD_SIZE + 1}" for a in moves)


def text_to_moves(text: str) -> List[int]:
    out = []
    for tok in text.replace(" ", "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        c = ord(tok[0].lower()) - ord("a")
        r = int(tok[1:]) - 1
        out.append(r * BOARD_SIZE + c)
    return out
