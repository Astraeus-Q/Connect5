/* 浏览器端 AI：神经网络 + PUCT 蒙特卡洛树搜索（含一步杀/必防战术检查）
 * 搜索按块异步执行，避免长时间阻塞主线程。
 */
(function (global) {
  'use strict';

  const R = global.GomokuRules;
  const CELLS = R.CELLS;
  const SIZE = R.SIZE;
  const CPUCT = 1.5;
  const VIRTUAL_LOSS = 1.0;

  class Node {
    constructor(board, player, terminal) {
      this.board = board;
      this.player = player;
      this.pi = null;
      this.value = 0;
      this.children = new Map();
      this.N = 0;
      this.W = 0;
      this.pending = 0;
      this.terminal = terminal === undefined ? null : terminal;
      this.prior = 0;
      this._legal = null;
    }

    legalActions() {
      if (this._legal === null) {
        const out = [];
        for (let i = 0; i < CELLS; i++) if (this.board[i] === 0) out.push(i);
        this._legal = out;
      }
      return this._legal;
    }

    state() {
      return R.encodeState(this.board, this.player);
    }

    makeChild(action, prior) {
      const nb = this.board.slice();
      nb[action] = this.player;
      const r = (action / SIZE) | 0;
      const c = action % SIZE;
      let child;
      if (R.checkWin(nb, r, c, this.player)) child = new Node(nb, 3 - this.player, -1);
      else if (R.isFull(nb)) child = new Node(nb, 3 - this.player, 0);
      else child = new Node(nb, 3 - this.player);
      child.prior = prior;
      this.children.set(action, child);
      return child;
    }

    select() {
      let nSum = 0;
      for (const ch of this.children.values()) nSum += ch.N + ch.pending;
      const sqrtSum = Math.sqrt(nSum + 1e-8);
      const legal = this.legalActions();
      let bestScore = -Infinity;
      let bestAction = -1;
      for (let i = 0; i < legal.length; i++) {
        const a = legal[i];
        const ch = this.children.get(a);
        let n = 0;
        let q = 0;
        if (ch) {
          if (ch.terminal !== null) {
            n = ch.N;
            q = -ch.terminal;
          } else {
            n = ch.N + ch.pending;
            q = n > 0 ? (ch.W - VIRTUAL_LOSS * ch.pending) / n : 0;
          }
        }
        const score = q + CPUCT * this.pi[a] * sqrtSum / (1 + n);
        if (score > bestScore) {
          bestScore = score;
          bestAction = a;
        }
      }
      return bestAction;
    }
  }

  class GomokuAI {
    constructor(net) {
      this.net = net || null;
      this.nodesSearched = 0;
    }

    setNetwork(net) {
      this.net = net || null;
    }

    _evaluate(node) {
      const out = this.net.forward(node.state());
      node.pi = out.policy;
      node.value = out.value;
      node.pending -= 1;
      node.N += 1;
      node.W += out.value;
    }

    _backprop(parents, leafValue) {
      let v = leafValue;
      for (let i = parents.length - 1; i >= 0; i--) {
        const p = parents[i];
        p.N += 1;
        p.W += -v;
        v = -v;
      }
    }

    _simulate(root) {
      let node = root;
      const parents = [];
      for (;;) {
        if (node.terminal !== null) {
          node.N += 1;
          this._backprop(parents, node.terminal);
          return;
        }
        if (node.pi === null) {
          if (node.pending > 0) return;          // 已被其它路径占用
          node.pending += 1;
          this._evaluate(node);
          this._backprop(parents, node.value);
          return;
        }
        const action = node.select();
        let child = node.children.get(action);
        if (!child) child = node.makeChild(action, node.pi[action]);
        parents.push(node);
        node = child;
      }
    }

    _addNoise(root, eps) {
      if (!eps) return;
      const legal = root.legalActions();
      const u = 1 / legal.length;
      let sum = 0;
      for (let i = 0; i < CELLS; i++) {
        root.pi[i] = (1 - eps) * root.pi[i] + (root.board[i] === 0 ? eps * u : 0);
        sum += root.pi[i];
      }
      for (let i = 0; i < CELLS; i++) root.pi[i] /= sum;
    }

    /**
     * 计算下一手。返回 { action, policy, value, sims, timeMs, forced, mode }
     */
    async think(board, player, opts) {
      opts = opts || {};
      const t0 = (global.performance || Date).now();
      const base = { action: -1, policy: null, value: null, sims: 0, forced: false, mode: 'tactical' };

      const forced = R.tacticalAction(board, player);
      if (forced >= 0) {
        base.action = forced;
        base.value = 1;
        base.mode = 'tactical';
        base.timeMs = (global.performance || Date).now() - t0;
        return base;
      }

      // forceHeuristic：网页端可选「启发式对手」，不加载权重也能对战
      if (!this.net || opts.forceHeuristic) {
        const action = R.heuristicAction(board, player, opts.tau !== undefined ? opts.tau : 150, 6);
        base.action = action;
        base.mode = 'heuristic';
        base.timeMs = (global.performance || Date).now() - t0;
        return base;
      }

      const sims = Math.max(1, opts.sims || 40);
      const temperature = opts.temperature !== undefined ? opts.temperature : 0.15;
      const root = new Node(board.slice(), player);
      this._evaluate(root);
      this._addNoise(root, opts.noise !== undefined ? opts.noise : 0.08);

      const chunk = Math.max(1, opts.chunk || 4);
      let done = 0;
      while (done < sims) {
        const end = Math.min(sims, done + chunk);
        for (; done < end; done++) this._simulate(root);
        this.nodesSearched = done;
        if (opts.onProgress) opts.onProgress(done, sims);
        if (opts.timeBudgetMs && (global.performance || Date).now() - t0 > opts.timeBudgetMs) break;
        if (done < sims) await new Promise((resolve) => setTimeout(resolve, 0));
      }

      const visits = new Float32Array(CELLS);
      for (const [a, ch] of root.children) visits[a] = ch.N;
      const pi = new Float32Array(CELLS);
      if (temperature > 1e-6) {
        for (let i = 0; i < CELLS; i++) pi[i] = Math.pow(visits[i], 1 / temperature);
      } else {
        for (let i = 0; i < CELLS; i++) pi[i] = visits[i];
      }
      let sum = 0;
      for (let i = 0; i < CELLS; i++) sum += pi[i];
      if (sum <= 0) return Object.assign(base, { mode: 'network', policy: root.pi, action: argmax(root.pi) });
      for (let i = 0; i < CELLS; i++) pi[i] /= sum;

      let action;
      if (temperature > 1e-6 && opts.greedy !== true) {
        const rnd = Math.random();
        let acc = 0;
        action = argmax(pi);
        for (let i = 0; i < CELLS; i++) {
          acc += pi[i];
          if (rnd <= acc) { action = i; break; }
        }
      } else {
        action = argmax(visits);
      }

      // 兜底：任何情况下都不能落在非法点
      if (board[action] !== 0) action = argmax(pi);

      return {
        action,
        policy: pi,
        value: root.N > 0 ? root.W / root.N : 0,
        sims: done,
        forced: false,
        mode: 'mcts',
        timeMs: (global.performance || Date).now() - t0,
      };
    }
  }

  function argmax(arr) {
    let best = -Infinity;
    let idx = 0;
    for (let i = 0; i < arr.length; i++) {
      if (arr[i] > best) { best = arr[i]; idx = i; }
    }
    return idx;
  }

  global.GomokuAI = GomokuAI;
  global.GomokuAI.Node = Node;
})(typeof window !== 'undefined' ? window : globalThis);
