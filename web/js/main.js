/* 主流程：模式控制、人机对弈、对局记录与回放 */
(function () {
  'use strict';

  const R = window.GomokuRules;
  const SIZE = R.SIZE;

  const LEVELS = {
    easy: { label: '简单', sims: 16, noise: 0.25, temperature: 1.0, budgetMs: 800 },
    normal: { label: '普通', sims: 64, noise: 0.1, temperature: 0.35, budgetMs: 2500 },
    hard: { label: '困难', sims: 240, noise: 0.04, temperature: 0.08, budgetMs: 8000 },
  };
  const SELFPLAY_CFG = { sims: 10, noise: 0.2, temperature: 0.8, budgetMs: 400 };

  const state = {
    board: R.createBoard(),
    player: 1,
    mode: 'pve',
    humanColor: 1,
    level: 'normal',
    aiKind: 'net',
    history: [],
    over: false,
    winner: 0,
    lastMove: -1,
    winCells: null,
    marks: [],
    busy: false,
    token: 0,
    startedAt: Date.now(),
    ai: new window.GomokuAI(null),
    net: null,
    arch: null,
    meta: null,
    replay: null,
  };

  let view = null;
  const el = (id) => document.getElementById(id);
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  // ------------------------------------------------------------------ 初始化
  function init() {
    view = new window.BoardView(el('board'));
    view.onMove = onHumanMove;

    bindSeg('modeSeg', 'mode', (v) => { state.mode = v; syncControls(); newGame(); });
    bindSeg('oppSeg', 'opp', (v) => { state.aiKind = v; syncControls(); });
    bindSeg('levelSeg', 'level', (v) => { state.level = v; });
    bindSeg('colorSeg', 'color', (v) => { state.humanColor = Number(v); newGame(); });

    el('btnNew').onclick = () => newGame();
    el('btnUndo').onclick = undo;
    el('btnHint').onclick = hint;

    bindTabs();
    bindHistory();
    bindReplay();
    bindModel();

    renderHistory();
    loadWeights();
    syncControls();
    newGame();
  }

  function bindSeg(id, key, cb) {
    const seg = el(id);
    seg.addEventListener('click', (e) => {
      const btn = e.target.closest('button');
      if (!btn) return;
      Array.prototype.forEach.call(seg.querySelectorAll('button'), (b) => b.classList.toggle('active', b === btn));
      cb(btn.dataset[key]);
    });
  }

  function bindTabs() {
    el('tabs').addEventListener('click', (e) => {
      const btn = e.target.closest('button');
      if (!btn) return;
      Array.prototype.forEach.call(el('tabs').querySelectorAll('button'), (b) => b.classList.toggle('active', b === btn));
      ['current', 'history', 'model'].forEach((t) => { el('tab' + cap(t)).hidden = t !== btn.dataset.tab; });
    });
  }

  function cap(s) { return s.charAt(0).toUpperCase() + s.slice(1); }

  function syncControls() {
    const disabled = state.mode !== 'pve';
    Array.prototype.forEach.call(el('colorSeg').querySelectorAll('button'), (b) => { b.disabled = disabled; });
    el('colorSeg').style.opacity = disabled ? 0.45 : 1;

    Array.prototype.forEach.call(el('oppSeg').querySelectorAll('button'), (b) => { b.disabled = disabled; });
    el('oppSeg').style.opacity = disabled ? 0.45 : 1;

    // 启发式对手不做搜索，强度档位对它无意义
    const levelOff = state.mode === 'pvp' || state.aiKind === 'heuristic';
    Array.prototype.forEach.call(el('levelSeg').querySelectorAll('button'), (b) => { b.disabled = levelOff; });
    el('levelSeg').style.opacity = levelOff ? 0.45 : 1;
  }

  // ------------------------------------------------------------------ 对局流程
  function newGame() {
    state.token++;
    state.board = R.createBoard();
    state.player = 1;
    state.history = [];
    state.over = false;
    state.winner = 0;
    state.lastMove = -1;
    state.winCells = null;
    state.marks = [];
    state.busy = false;
    state.startedAt = Date.now();
    state.replay = null;
    el('replayBar').hidden = true;
    render();
    maybeAi();
  }

  function onHumanMove(action) {
    if (state.busy || state.over || state.replay) return;
    if (state.board[action] !== 0) return;
    state.marks = [];
    applyMove(action);
  }

  function applyMove(action) {
    const player = state.player;
    state.board[action] = player;
    state.history.push(action);
    state.lastMove = action;

    const r = (action / SIZE) | 0;
    const c = action % SIZE;
    const line = R.checkWin(state.board, r, c, player) ? R.winLine(state.board, r, c, player) : null;
    if (line) {
      state.over = true;
      state.winner = player;
      state.winCells = line;
    } else if (R.isFull(state.board)) {
      state.over = true;
      state.winner = 0;
    } else {
      state.player = 3 - player;
    }
    render();
    if (state.over) endGame();
    else maybeAi();
  }

  function isAiTurn() {
    if (state.mode === 'eva') return true;
    if (state.mode === 'pve') return state.player !== state.humanColor;
    return false;
  }

  function maybeAi() {
    if (state.over || state.replay) return;
    if (!isAiTurn()) {
      view.setInteractive(true);
      updateStatus();
      evaluateNow();
      return;
    }
    view.setInteractive(false);
    updateStatus();
    aiMove();
  }

  async function aiMove() {
    const cfg = state.mode === 'eva' ? SELFPLAY_CFG : LEVELS[state.level];
    const heuristic = state.aiKind === 'heuristic';
    const token = state.token;
    state.busy = true;
    if (!heuristic) showThink(0, cfg.sims);
    let res;
    try {
      res = await state.ai.think(state.board, state.player, {
        sims: cfg.sims,
        noise: cfg.noise,
        temperature: cfg.temperature,
        timeBudgetMs: cfg.budgetMs,
        chunk: 4,
        forceHeuristic: heuristic,
        tau: 0,
        onProgress: (done, total) => { if (token === state.token) showThink(done, total); },
      });
    } finally {
      state.busy = false;
      hideThink();
    }
    if (token !== state.token || !res || res.action < 0) return;
    renderAnalysis(res);
    if (state.mode === 'eva') await sleep(120);
    if (token !== state.token || state.over) return;
    applyMove(res.action);
  }

  function endGame() {
    view.setInteractive(false);
    const playerColor = state.mode === 'pve' ? state.humanColor : 1;
    let result = 'draw';
    if (state.winner !== 0) result = state.winner === playerColor ? 'win' : 'lose';
    window.GameStore.save({
      mode: state.mode,
      playerColor,
      aiLevel: state.mode === 'pvp' ? null
        : (state.aiKind === 'heuristic' ? 'heuristic' : (state.mode === 'eva' ? 'selfplay' : state.level)),
      result,
      winner: state.winner,
      moves: state.history.slice(),
      durationMs: Date.now() - state.startedAt,
      modelTag: state.meta && state.meta.iteration !== undefined ? state.meta.iteration : null,
    });
    renderHistory();
    updateStatus();
  }

  function undo() {
    if (state.busy || !state.history.length) return;
    state.token++;
    state.over = false;
    state.winner = 0;
    state.winCells = null;
    state.marks = [];
    const pop = () => {
      const a = state.history.pop();
      state.board[a] = 0;
      state.player = 3 - state.player;
    };
    pop();
    if (state.mode === 'pve' && state.player !== state.humanColor && state.history.length) pop();
    state.lastMove = state.history.length ? state.history[state.history.length - 1] : -1;
    render();
    maybeAi();
  }

  async function hint() {
    if (state.busy || state.over || state.replay) return;
    const cfg = LEVELS[state.level];
    const token = state.token;
    state.busy = true;
    showThink(0, cfg.sims);
    let res;
    try {
      res = await state.ai.think(state.board, state.player, {
        sims: Math.max(32, cfg.sims), noise: 0.05, temperature: 0, greedy: true,
        timeBudgetMs: 3000, chunk: 4,
        forceHeuristic: state.aiKind === 'heuristic', tau: 0,
        onProgress: (d, t) => { if (token === state.token) showThink(d, t); },
      });
    } finally {
      state.busy = false;
      hideThink();
    }
    if (token !== state.token || !res || res.action < 0) return;
    state.marks = [{ action: res.action, color: 'rgba(99,210,255,.95)' }];
    render();
    renderAnalysis(res);
  }

  // ------------------------------------------------------------------ 渲染
  function render() {
    view.setBoard(state.board, { lastMove: state.lastMove, winCells: state.winCells, marks: state.marks });
    view.setPreviewColor(state.player);
    renderMoves();
    updateStatus();
  }

  function renderMoves() {
    const list = el('moveList');
    list.innerHTML = '';
    let player = 1;
    for (let i = 0; i < state.history.length; i++) {
      const a = state.history[i];
      const li = document.createElement('li');
      if (i === state.history.length - 1) li.className = 'last';
      li.innerHTML = `<span class="n">${i + 1}.</span><span class="st ${player === 1 ? 'black' : 'white'}"></span><span>${R.moveToText(a)}</span>`;
      list.appendChild(li);
      player = 3 - player;
    }
    el('moveCount').textContent = state.history.length + ' 手';
    list.scrollTop = list.scrollHeight;
  }

  function updateStatus() {
    const stone = el('turnStone');
    stone.className = 'turn-stone ' + (state.player === 1 ? 'black' : 'white');
    let text;
    if (state.over) {
      const who = state.winner === 0 ? '平局' : (state.winner === 1 ? '黑方获胜' : '白方获胜');
      text = who;
      if (state.mode === 'pve' && state.winner !== 0) {
        text += state.winner === state.humanColor ? ' · 你赢了' : ' · AI 获胜';
      }
    } else {
      const side = state.player === 1 ? '黑方' : '白方';
      const kindText = state.aiKind === 'heuristic' ? '启发式 AI' : 'AI';
      if (state.mode === 'pve') {
        text = state.player === state.humanColor ? `轮到你落子（${side}）` : `${kindText} 思考中（${side}）`;
      } else if (state.mode === 'eva') {
        text = `${kindText} 自对弈中（${side}）`;
      } else {
        text = `轮到${side}落子`;
      }
    }
    el('statusText').textContent = text;
    el('btnUndo').disabled = state.busy || !state.history.length || state.mode === 'eva';
  }

  function showThink(done, total) {
    el('thinkChip').hidden = false;
    el('thinkChip').textContent = `搜索 ${done}/${total}`;
  }

  function hideThink() { el('thinkChip').hidden = true; }

  function evaluateNow() {
    if (!state.net || state.over) return;
    const out = state.net.forward(R.encodeState(state.board, state.player));
    renderEval(out.value, state.mode === 'pve' && state.player === state.humanColor ? '你的胜率' : '当前方胜率');
    el('aiModeText').textContent = state.net ? '网络直评' : '启发式';
  }

  function renderAnalysis(res) {
    el('aiModeText').textContent = res.mode === 'tactical' ? '战术强制手' : (res.mode === 'mcts' ? 'MCTS 搜索' : '启发式');
    el('kvNodes').textContent = res.sims ? res.sims + ' 次模拟' : '--';
    el('kvTime').textContent = res.timeMs ? Math.round(res.timeMs) + ' ms' : '--';
    if (res.value !== null && res.value !== undefined) {
      renderEval(res.value, 'AI 视角胜率');
    }
    renderCandidates(res.policy, res.action);
  }

  function renderEval(value, label) {
    const p = Math.max(0, Math.min(1, (value + 1) / 2));
    el('evalFill').style.width = (p * 100).toFixed(1) + '%';
    el('evalText').textContent = `${label} ${(p * 100).toFixed(0)}%`;
    el('evalChip').hidden = false;
    el('evalChip').textContent = `评估 ${value >= 0 ? '+' : ''}${value.toFixed(2)}`;
  }

  function renderCandidates(policy, action) {
    const box = el('candidateList');
    box.innerHTML = '';
    if (!policy) return;
    const idx = Array.from(policy.keys()).sort((a, b) => policy[b] - policy[a]).slice(0, 3);
    for (const i of idx) {
      if (policy[i] <= 0.0005) continue;
      const span = document.createElement('span');
      span.className = 'cand';
      span.innerHTML = `${R.moveToText(i)} <b>${(policy[i] * 100).toFixed(1)}%</b>${i === action ? ' ✓' : ''}`;
      box.appendChild(span);
    }
  }

  // ------------------------------------------------------------------ 记录
  function bindHistory() {
    el('btnExportAll').onclick = () => {
      window.GameStore.download(`connect5-games-${stamp()}.json`, window.GameStore.exportJSON());
    };
    el('btnExportTrain').onclick = () => {
      const json = window.GameStore.exportTrainingJSON({ minMoves: 8 });
      window.GameStore.download(`connect5-training-${stamp()}.json`, json);
    };
    el('btnImport').onclick = () => el('importFile').click();
    el('importFile').onchange = (e) => {
      const f = e.target.files && e.target.files[0];
      if (!f) return;
      const reader = new FileReader();
      reader.onload = () => {
        try {
          const n = window.GameStore.importJSON(reader.result);
          renderHistory();
          alert(`已导入 ${n} 局`);
        } catch (err) {
          alert('导入失败：' + err.message);
        }
      };
      reader.readAsText(f);
      e.target.value = '';
    };
    el('btnClear').onclick = () => {
      if (!confirm('确定清空全部对局记录？')) return;
      window.GameStore.clear();
      renderHistory();
    };
    el('gameList').addEventListener('click', (e) => {
      const btn = e.target.closest('button');
      if (!btn) return;
      const id = btn.closest('.game-item').dataset.id;
      if (btn.dataset.op === 'replay') startReplay(id);
      else if (btn.dataset.op === 'del') {
        window.GameStore.remove(id);
        renderHistory();
      }
    });
  }

  function renderHistory() {
    const games = window.GameStore.all();
    const s = window.GameStore.stats(games);
    el('historyCount').textContent = games.length + ' 局';
    el('statsGrid').innerHTML = [
      ['对局', s.total], ['胜', s.win], ['负', s.lose],
      ['平', s.draw], ['胜率', (s.winRate * 100).toFixed(0) + '%'], ['平均步数', s.avgMoves],
    ].map(([k, v]) => `<div class="stat"><div class="num">${v}</div><div class="lbl">${k}</div></div>`).join('');

    const list = el('gameList');
    if (!games.length) {
      list.innerHTML = '<div class="empty">还没有记录，先下一局吧</div>';
      return;
    }
    list.innerHTML = games.slice(0, 200).map((g) => {
      const label = g.result === 'win' ? '胜' : (g.result === 'lose' ? '负' : '平');
      const cls = g.result === 'win' ? 'win' : (g.result === 'lose' ? 'lose' : 'draw');
      const modeText = { pve: '人机', pvp: '双人', eva: 'AI自战' }[g.mode] || g.mode;
      const colorText = g.winner === 0 ? '和棋' : (g.winner === 1 ? '黑胜' : '白胜');
      const date = new Date(g.date).toLocaleString('zh-CN', { hour12: false });
      const tag = g.modelTag !== null && g.modelTag !== undefined ? ` · iter${g.modelTag}` : '';
      return `<div class="game-item" data-id="${g.id}">
        <span class="badge ${cls}">${label}</span>
        <div class="meta"><b>${modeText}${tag}</b> ${colorText} · ${(g.moves || []).length} 手<br><span>${date}</span></div>
        <div class="ops">
          <button data-op="replay">回放</button>
          <button data-op="del" class="danger">删除</button>
        </div>
      </div>`;
    }).join('');
  }

  function stamp() {
    const d = new Date();
    const p = (n) => String(n).padStart(2, '0');
    return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}`;
  }

  // ------------------------------------------------------------------ 回放
  function bindReplay() {
    el('rpFirst').onclick = () => replayTo(0);
    el('rpPrev').onclick = () => replayTo(state.replay.idx - 1);
    el('rpNext').onclick = () => replayTo(state.replay.idx + 1);
    el('rpLast').onclick = () => replayTo(state.replay.moves.length);
    el('rpPlay').onclick = toggleReplayPlay;
    el('rpExit').onclick = exitReplay;
  }

  function startReplay(id) {
    const game = window.GameStore.all().find((g) => g.id === id);
    if (!game) return;
    state.token++;
    state.replay = { game, moves: game.moves || [], idx: 0, timer: null };
    el('replayBar').hidden = false;
    el('replayTitle').textContent = `回放 · ${new Date(game.date).toLocaleString('zh-CN', { hour12: false })}`;
    view.setInteractive(false);
    replayTo(0);
  }

  function replayTo(idx) {
    const rp = state.replay;
    if (!rp) return;
    rp.idx = Math.max(0, Math.min(rp.moves.length, idx));
    const b = R.createBoard();
    let player = 1;
    let winCells = null;
    for (let i = 0; i < rp.idx; i++) {
      const a = rp.moves[i];
      b[a] = player;
      const r = (a / SIZE) | 0;
      const c = a % SIZE;
      if (R.checkWin(b, r, c, player)) winCells = R.winLine(b, r, c, player);
      player = 3 - player;
    }
    view.setBoard(b, { lastMove: rp.idx ? rp.moves[rp.idx - 1] : -1, winCells, marks: [] });
    el('replayStep').textContent = `${rp.idx} / ${rp.moves.length}`;
    if (rp.idx >= rp.moves.length) stopReplayPlay();
  }

  function toggleReplayPlay() {
    const rp = state.replay;
    if (!rp) return;
    if (rp.timer) { stopReplayPlay(); return; }
    if (rp.idx >= rp.moves.length) replayTo(0);
    el('rpPlay').textContent = '⏸ 暂停';
    rp.timer = setInterval(() => {
      if (!state.replay) return stopReplayPlay();
      if (state.replay.idx >= state.replay.moves.length) return stopReplayPlay();
      replayTo(state.replay.idx + 1);
    }, 420);
  }

  function stopReplayPlay() {
    if (state.replay && state.replay.timer) {
      clearInterval(state.replay.timer);
      state.replay.timer = null;
    }
    el('rpPlay').textContent = '▶ 播放';
  }

  function exitReplay() {
    stopReplayPlay();
    state.replay = null;
    el('replayBar').hidden = true;
    render();
    maybeAi();
  }

  // ------------------------------------------------------------------ 模型
  function bindModel() {
    el('btnLoadWeight').onclick = () => el('weightFile').click();
    el('weightFile').onchange = async (e) => {
      const f = e.target.files && e.target.files[0];
      if (!f) return;
      try {
        const loaded = await window.GomokuNet.loadFromFile(f);
        applyNet(loaded);
        el('weightHint').textContent = '已从本地文件加载权重：' + f.name;
      } catch (err) {
        el('weightHint').textContent = '加载失败：' + err.message;
      }
      e.target.value = '';
    };
  }

  async function loadWeights() {
    try {
      const loaded = await window.GomokuNet.load('models/weights.json');
      applyNet(loaded);
    } catch (e) {
      setBadge('warn', '未加载权重 · 启发式 AI');
      el('weightHint').innerHTML =
        '自动加载 <code>models/weights.json</code> 失败（' + escapeHtml(e.message) +
        '）。可直接选择权重文件，或用 <code>python -m http.server</code> 启动后再打开本页面。';
    }
  }

  function applyNet(loaded) {
    state.net = loaded.net;
    state.arch = loaded.arch;
    state.meta = loaded.meta || {};
    state.ai.setNetwork(loaded.net);
    const it = state.meta.iteration;
    setBadge('ok', `模型${typeof it === 'number' ? ' iter' + it : '已加载'} · ${loaded.arch.channels}ch×${loaded.arch.n_blocks}层`);
    const total = Object.values(loaded.net.w).reduce((a, w) => a + w.length, 0);
    el('modelInfo').innerHTML = [
      ['架构', `${loaded.arch.in_channels}→${loaded.arch.channels}ch × ${loaded.arch.n_blocks} 层卷积`],
      ['参数量', `${(total / 1000).toFixed(1)} K`],
      ['迭代轮次', state.meta.iteration !== undefined ? state.meta.iteration : '--'],
      ['来源', state.meta.source || '--'],
      ['推理', '纯 JS（BN 已融合）'],
    ].map(([k, v]) => `<div><span class="k">${k}</span><span class="v">${escapeHtml(String(v))}</span></div>`).join('');
  }

  function setBadge(kind, text) {
    el('modelBadge').className = 'model-badge ' + kind;
    el('modelBadgeText').textContent = text;
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
