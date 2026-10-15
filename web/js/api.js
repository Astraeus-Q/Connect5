/* 后端推理服务调用（ai/server.py）
 *
 * 网页默认是本地 JS 推理（v1 系列，20 万参数，离线可用）。
 * v2 系（v2 / v2.2，残差大模型）浏览器跑不动，只能由后端服务提供 —— 这个文件就是两者的桥。
 *
 * 服务器不可用时会自动回退本地推理，不影响下棋。
 */
(function (global) {
  'use strict';

  const DEFAULT_URL = 'http://127.0.0.1:8000';

  const Backend = {
    url: DEFAULT_URL,
    available: false,
    info: null,

    async check() {
      try {
        const res = await fetch(`${this.url}/health`, { cache: 'no-store' });
        if (!res.ok) throw new Error('HTTP ' + res.status);
        const data = await res.json();
        this.available = !!data.ok;
        this.info = data;
        return this.available;
      } catch (e) {
        this.available = false;
        this.info = null;
        return false;
      }
    },

    /** 向后端要一步棋。board 为 Int8Array(225)，返回 { action, policy, value, sims, timeMs, mode, model } */
    async move(board, player, sims, temperature) {
      const res = await fetch(`${this.url}/move`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          board: Array.prototype.slice.call(board),
          player,
          sims: sims || 200,
          temperature: temperature || 0,
        }),
      });
      if (!res.ok) throw new Error(`后端返回 ${res.status}`);
      return await res.json();
    },
  };

  global.GomokuBackend = Backend;
})(typeof window !== 'undefined' ? window : globalThis);
