/* 五子棋规则与启发式（与 ai/gomoku.py 完全一致的语义）
 * board: Int8Array(225)，索引 = row * 15 + col；0 空 / 1 黑 / 2 白，黑先。
 */
(function (global) {
  'use strict';

  const SIZE = 15;
  const CELLS = 225;
  const WIN_LEN = 5;
  const DIRS = [[1, 0], [0, 1], [1, 1], [1, -1]];

  function createBoard() {
    return new Int8Array(CELLS);
  }

  function checkWin(board, r, c, color) {
    for (let d = 0; d < DIRS.length; d++) {
      const dr = DIRS[d][0], dc = DIRS[d][1];
      let cnt = 1;
      for (let s = -1; s <= 1; s += 2) {
        let rr = r + dr * s, cc = c + dc * s;
        while (rr >= 0 && rr < SIZE && cc >= 0 && cc < SIZE && board[rr * SIZE + cc] === color) {
          cnt++;
          rr += dr * s;
          cc += dc * s;
        }
      }
      if (cnt >= WIN_LEN) return true;
    }
    return false;
  }

  /** 返回构成五连的完整坐标序列（用于胜利连线高亮），未获胜时返回 null */
  function winLine(board, r, c, color) {
    for (let d = 0; d < DIRS.length; d++) {
      const dr = DIRS[d][0], dc = DIRS[d][1];
      const line = [[r, c]];
      for (let s = -1; s <= 1; s += 2) {
        let rr = r + dr * s, cc = c + dc * s;
        while (rr >= 0 && rr < SIZE && cc >= 0 && cc < SIZE && board[rr * SIZE + cc] === color) {
          if (s === -1) line.unshift([rr, cc]); else line.push([rr, cc]);
          rr += dr * s;
          cc += dc * s;
        }
      }
      if (line.length >= WIN_LEN) return line;
    }
    return null;
  }

  function isFull(board) {
    for (let i = 0; i < CELLS; i++) if (board[i] === 0) return false;
    return true;
  }

  function legalActions(board) {
    const out = [];
    for (let i = 0; i < CELLS; i++) if (board[i] === 0) out.push(i);
    return out;
  }

  /** 网络输入：(自己, 对手, 行棋方常数平面)，长度 3*225 */
  function encodeState(board, player) {
    const s = new Float32Array(3 * CELLS);
    const opp = 3 - player;
    for (let i = 0; i < CELLS; i++) {
      if (board[i] === player) s[i] = 1;
      else if (board[i] === opp) s[CELLS + i] = 1;
    }
    if (player === 1) {
      for (let i = 0; i < CELLS; i++) s[2 * CELLS + i] = 1;
    }
    return s;
  }

  // ------------------------------------------------------------------ 启发式
  function patternScore(cnt, open) {
    if (cnt >= 5) return 10000000;
    if (cnt === 4) return open === 2 ? 100000 : (open === 1 ? 10000 : 0);
    if (cnt === 3) return open === 2 ? 8000 : (open === 1 ? 800 : 0);
    if (cnt === 2) return open === 2 ? 600 : (open === 1 ? 60 : 0);
    return open === 2 ? 40 : (open === 1 ? 6 : 0);
  }

  function evaluatePoint(board, r, c, color) {
    let total = 0;
    for (let d = 0; d < DIRS.length; d++) {
      const dr = DIRS[d][0], dc = DIRS[d][1];
      let cnt = 1, open = 0;
      for (let s = -1; s <= 1; s += 2) {
        let rr = r + dr * s, cc = c + dc * s;
        while (rr >= 0 && rr < SIZE && cc >= 0 && cc < SIZE && board[rr * SIZE + cc] === color) {
          cnt++;
          rr += dr * s;
          cc += dc * s;
        }
        if (rr >= 0 && rr < SIZE && cc >= 0 && cc < SIZE && board[rr * SIZE + cc] === 0) open++;
      }
      total += patternScore(cnt, open);
    }
    return total;
  }

  function candidateMask(board, radius) {
    radius = radius || 2;
    const mask = new Uint8Array(CELLS);
    let any = false;
    for (let i = 0; i < CELLS; i++) if (board[i] !== 0) { any = true; break; }
    if (!any) { mask[7 * SIZE + 7] = 1; return mask; }
    for (let r = 0; r < SIZE; r++) {
      for (let c = 0; c < SIZE; c++) {
        if (board[r * SIZE + c] === 0) continue;
        for (let rr = Math.max(0, r - radius); rr <= Math.min(SIZE - 1, r + radius); rr++) {
          for (let cc = Math.max(0, c - radius); cc <= Math.min(SIZE - 1, c + radius); cc++) {
            const idx = rr * SIZE + cc;
            if (board[idx] === 0) mask[idx] = 1;
          }
        }
      }
    }
    return mask;
  }

  function heuristicScores(board, player) {
    const opp = 3 - player;
    const mask = candidateMask(board, 2);
    const scores = new Float32Array(CELLS);
    for (let i = 0; i < CELLS; i++) {
      if (!mask[i]) continue;
      const r = (i / SIZE) | 0, c = i % SIZE;
      scores[i] = evaluatePoint(board, r, c, player) + 0.9 * evaluatePoint(board, r, c, opp);
    }
    return scores;
  }

  /** 无模型时的回退策略：top-k 中按 softmax 采样（tau<=0 时取最优） */
  function heuristicAction(board, player, tau, topk, rng) {
    tau = tau === undefined ? 200 : tau;
    topk = topk || 6;
    const scores = heuristicScores(board, player);
    const cand = [];
    for (let i = 0; i < CELLS; i++) if (board[i] === 0) cand.push(i);
    cand.sort((a, b) => scores[b] - scores[a]);
    const top = cand.slice(0, topk);
    if (!top.length) return -1;
    if (tau <= 0) return top[0];
    let max = -Infinity;
    for (const a of top) max = Math.max(max, scores[a]);
    const ws = top.map((a) => Math.exp((scores[a] - max) / tau));
    const sum = ws.reduce((x, y) => x + y, 0);
    const rnd = (rng ? rng() : Math.random()) * sum;
    let acc = 0;
    for (let i = 0; i < top.length; i++) {
      acc += ws[i];
      if (rnd <= acc) return top[i];
    }
    return top[0];
  }

  /** 一步成五的所有点 */
  function winningMoves(board, color) {
    const mask = candidateMask(board, 4);
    const out = [];
    for (let i = 0; i < CELLS; i++) {
      if (!mask[i]) continue;
      const r = (i / SIZE) | 0, c = i % SIZE;
      if (checkWin(board, r, c, color)) out.push(i);
    }
    return out;
  }

  function inBounds(r, c) { return r >= 0 && r < SIZE && c >= 0 && c < SIZE; }

  /** 假设 (r,c) 已落 color，返回四个方向的 [连子数, 两端空位数] */
  function lineStats(board, r, c, color) {
    const out = [];
    for (let d = 0; d < DIRS.length; d++) {
      const dr = DIRS[d][0], dc = DIRS[d][1];
      let cnt = 1, open = 0;
      for (let s = -1; s <= 1; s += 2) {
        let rr = r + dr * s, cc = c + dc * s;
        while (inBounds(rr, cc) && board[rr * SIZE + cc] === color) {
          cnt++;
          rr += dr * s;
          cc += dc * s;
        }
        if (inBounds(rr, cc) && board[rr * SIZE + cc] === 0) open++;
      }
      out.push([cnt, open]);
    }
    return out;
  }

  /** 落子后能形成「活四」（四连且两端皆空）的点 */
  function openFourMoves(board, color) {
    const mask = candidateMask(board, 4);
    const out = [];
    for (let i = 0; i < CELLS; i++) {
      if (!mask[i]) continue;
      const r = (i / SIZE) | 0, c = i % SIZE;
      const stats = lineStats(board, r, c, color);
      for (let d = 0; d < stats.length; d++) {
        if (stats[d][0] === 4 && stats[d][1] === 2) { out.push(i); break; }
      }
    }
    return out;
  }

  /** 强制手：成五 > 堵五 > 造活四 > 堵活四；无强制手时返回 -1 */
  function tacticalAction(board, player) {
    const me = player;
    const opp = 3 - player;
    const mine = winningMoves(board, me);
    if (mine.length) return mine[0];
    const theirs = winningMoves(board, opp);
    if (theirs.length) return theirs[0];
    const myFours = openFourMoves(board, me);
    if (myFours.length) return myFours[0];
    const theirFours = openFourMoves(board, opp);
    if (theirFours.length) return theirFours[0];
    return -1;
  }

  /** 动作 -> "h8" */
  function moveToText(action) {
    const r = (action / SIZE) | 0, c = action % SIZE;
    return String.fromCharCode(97 + c) + (r + 1);
  }

  global.GomokuRules = {
    SIZE, CELLS, WIN_LEN, DIRS,
    createBoard, checkWin, winLine, isFull, legalActions,
    encodeState, heuristicScores, heuristicAction, winningMoves, openFourMoves, lineStats, tacticalAction, moveToText,
  };
})(typeof window !== 'undefined' ? window : globalThis);
