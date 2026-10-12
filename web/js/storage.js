/* 对局记录：localStorage 持久化、导入导出、训练数据生成 */
(function (global) {
  'use strict';

  const R = global.GomokuRules;
  const KEY = 'connect5.games.v2';

  function uid() {
    return Date.now().toString(36) + Math.random().toString(36).slice(2, 7);
  }

  function readAll() {
    try {
      const raw = global.localStorage.getItem(KEY);
      if (!raw) return [];
      const list = JSON.parse(raw);
      return Array.isArray(list) ? list : [];
    } catch (e) {
      console.warn('读取对局记录失败', e);
      return [];
    }
  }

  function writeAll(list) {
    try {
      global.localStorage.setItem(KEY, JSON.stringify(list));
      return true;
    } catch (e) {
      console.warn('保存对局记录失败', e);
      return false;
    }
  }

  const Store = {
    all() {
      return readAll().sort((a, b) => b.date - a.date);
    },

    save(game) {
      const list = readAll();
      const record = Object.assign({ id: uid(), date: Date.now(), boardSize: R.SIZE }, game);
      list.push(record);
      if (list.length > 500) list.splice(0, list.length - 500);
      writeAll(list);
      return record;
    },

    remove(id) {
      writeAll(readAll().filter((g) => g.id !== id));
    },

    clear() {
      writeAll([]);
    },

    /** 完整导出（含元信息），可直接被 ai/import_games.py 读取 */
    exportJSON(games) {
      const list = games || this.all();
      return JSON.stringify(
        {
          app: 'connect5',
          version: 2,
          boardSize: R.SIZE,
          exportedAt: new Date().toISOString(),
          games: list,
        },
        null,
        2
      );
    },

    /** 训练数据：只保留必要字段，可选只导出人机对局中人类的走子价值高的对局 */
    exportTrainingJSON(opts) {
      opts = opts || {};
      const games = (opts.games || this.all()).filter((g) => {
        if (opts.minMoves && (g.moves || []).length < opts.minMoves) return false;
        if (opts.humanOnly && g.mode !== 'pve') return false;
        return (g.moves || []).length > 0;
      });
      return JSON.stringify(
        {
          app: 'connect5',
          version: 2,
          kind: 'training',
          boardSize: R.SIZE,
          exportedAt: new Date().toISOString(),
          games: games.map((g) => ({
            moves: g.moves,
            winner: g.winner,
            playerColor: g.playerColor,
            result: g.result,
            mode: g.mode,
          })),
        },
        null,
        2
      );
    },

    importJSON(text) {
      const data = typeof text === 'string' ? JSON.parse(text) : text;
      const incoming = Array.isArray(data) ? data : data.games || [];
      const list = readAll();
      const existing = new Set(list.map((g) => (g.moves || []).join(',') + '|' + g.date));
      let added = 0;
      for (const g of incoming) {
        if (!g || !Array.isArray(g.moves)) continue;
        const key = (g.moves || []).join(',') + '|' + (g.date || '');
        if (existing.has(key)) continue;
        list.push(Object.assign({ id: uid(), date: g.date || Date.now(), boardSize: R.SIZE }, g));
        added++;
      }
      writeAll(list);
      return added;
    },

    stats(games) {
      const list = games || this.all();
      const s = { total: list.length, win: 0, lose: 0, draw: 0, avgMoves: 0, byMode: {} };
      let moves = 0;
      for (const g of list) {
        if (g.result === 'win') s.win++;
        else if (g.result === 'lose') s.lose++;
        else s.draw++;
        moves += (g.moves || []).length;
        s.byMode[g.mode] = (s.byMode[g.mode] || 0) + 1;
      }
      s.avgMoves = list.length ? Math.round(moves / list.length) : 0;
      s.winRate = s.win + s.lose > 0 ? s.win / (s.win + s.lose) : 0;
      return s;
    },

    download(filename, text) {
      const blob = new Blob([text], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    },
  };

  global.GameStore = Store;
})(typeof window !== 'undefined' ? window : globalThis);
