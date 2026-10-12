/* Canvas 棋盘：绘制、悬停预览、落子动画与点击交互 */
(function (global) {
  'use strict';

  const R = global.GomokuRules;
  const SIZE = R.SIZE;

  class BoardView {
    constructor(canvas) {
      this.canvas = canvas;
      this.ctx = canvas.getContext('2d');
      this.board = R.createBoard();
      this.lastMove = -1;
      this.winCells = null;
      this.hover = -1;
      this.previewColor = 1;
      this.interactive = false;
      this.onMove = null;
      this.padding = 26;
      this.cssSize = 480;
      this.lastMoveAt = 0;
      this.marks = [];                 // [{action, color, kind}] 覆盖标记（候选点等）
      this._bind();
      this.resize();
    }

    _bind() {
      const canvas = this.canvas;
      canvas.addEventListener('mousemove', (e) => {
        const a = this._actionAt(e.clientX, e.clientY);
        if (a !== this.hover) {
          this.hover = a;
          if (this.interactive) this.draw();
        }
      });
      canvas.addEventListener('mouseleave', () => {
        if (this.hover !== -1) { this.hover = -1; this.draw(); }
      });
      canvas.addEventListener('click', (e) => {
        if (!this.interactive || !this.onMove) return;
        const a = this._actionAt(e.clientX, e.clientY);
        if (a >= 0 && this.board[a] === 0) this.onMove(a);
      });
      canvas.addEventListener('touchstart', (e) => {
        if (!this.interactive || !this.onMove || !e.touches.length) return;
        const t = e.touches[0];
        const a = this._actionAt(t.clientX, t.clientY);
        if (a >= 0 && this.board[a] === 0) { e.preventDefault(); this.onMove(a); }
      }, { passive: false });
      if (global.ResizeObserver) {
        this._ro = new ResizeObserver(() => this.resize());
        this._ro.observe(canvas.parentElement || canvas);
      } else {
        global.addEventListener('resize', () => this.resize());
      }
    }

    resize() {
      const parent = this.canvas.parentElement;
      const width = parent ? parent.clientWidth : 480;
      const size = Math.max(260, Math.min(width, 760));
      const dpr = global.devicePixelRatio || 1;
      this.cssSize = size;
      this.canvas.style.width = size + 'px';
      this.canvas.style.height = size + 'px';
      this.canvas.width = Math.round(size * dpr);
      this.canvas.height = Math.round(size * dpr);
      this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      this.cell = (size - this.padding * 2) / (SIZE - 1);
      this.draw();
    }

    _actionAt(clientX, clientY) {
      const rect = this.canvas.getBoundingClientRect();
      const x = clientX - rect.left;
      const y = clientY - rect.top;
      const cell = this.cell;
      const pad = this.padding;
      const c = Math.round((x - pad) / cell);
      const r = Math.round((y - pad) / cell);
      if (r < 0 || r >= SIZE || c < 0 || c >= SIZE) return -1;
      const dx = x - (pad + c * cell);
      const dy = y - (pad + r * cell);
      if (dx * dx + dy * dy > (cell * 0.58) * (cell * 0.58)) return -1;
      return r * SIZE + c;
    }

    setBoard(board, opts) {
      opts = opts || {};
      this.board = board;
      this.lastMove = opts.lastMove !== undefined ? opts.lastMove : -1;
      this.winCells = opts.winCells || null;
      this.marks = opts.marks || [];
      this.lastMoveAt = (global.performance || Date).now();
      this.draw();
    }

    setPreviewColor(color) { this.previewColor = color; }

    setInteractive(v) {
      this.interactive = v;
      this.canvas.classList.toggle('thinking', !v);
    }

    _x(i) { return this.padding + i * this.cell; }

    draw() {
      const ctx = this.ctx;
      const size = this.cssSize;
      const cell = this.cell;
      const pad = this.padding;
      ctx.clearRect(0, 0, size, size);

      // 棋盘底色
      const bg = ctx.createLinearGradient(0, 0, size, size);
      bg.addColorStop(0, '#f0d7ab');
      bg.addColorStop(0.5, '#e8c892');
      bg.addColorStop(1, '#dcb57c');
      ctx.fillStyle = bg;
      roundRect(ctx, 1, 1, size - 2, size - 2, 12);
      ctx.fill();
      ctx.strokeStyle = 'rgba(90, 60, 30, 0.55)';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // 网格
      ctx.strokeStyle = 'rgba(60, 40, 20, 0.65)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let i = 0; i < SIZE; i++) {
        ctx.moveTo(this._x(0), this._x(i));
        ctx.lineTo(this._x(SIZE - 1), this._x(i));
        ctx.moveTo(this._x(i), this._x(0));
        ctx.lineTo(this._x(i), this._x(SIZE - 1));
      }
      ctx.stroke();

      // 外框加粗
      ctx.strokeStyle = 'rgba(50, 32, 16, 0.85)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(this._x(0), this._x(0));
      ctx.lineTo(this._x(SIZE - 1), this._x(0));
      ctx.lineTo(this._x(SIZE - 1), this._x(SIZE - 1));
      ctx.lineTo(this._x(0), this._x(SIZE - 1));
      ctx.closePath();
      ctx.stroke();

      // 星位
      ctx.fillStyle = 'rgba(50, 32, 16, 0.8)';
      for (const [r, c] of [[3, 3], [3, 11], [11, 3], [11, 11], [7, 7]]) {
        ctx.beginPath();
        ctx.arc(this._x(c), this._x(r), Math.max(2.5, cell * 0.08), 0, Math.PI * 2);
        ctx.fill();
      }

      // 坐标
      ctx.fillStyle = 'rgba(70, 45, 22, 0.65)';
      ctx.font = `${Math.max(9, cell * 0.34)}px ui-sans-serif, system-ui, sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      for (let i = 0; i < SIZE; i++) {
        ctx.fillText(String.fromCharCode(65 + i), this._x(i), pad * 0.5);
        ctx.fillText(String.fromCharCode(65 + i), this._x(i), size - pad * 0.5);
        ctx.fillText(String(i + 1), pad * 0.5, this._x(i));
        ctx.fillText(String(i + 1), size - pad * 0.5, this._x(i));
      }

      // 悬停预览
      if (this.interactive && this.hover >= 0 && this.board[this.hover] === 0) {
        const r = (this.hover / SIZE) | 0;
        const c = this.hover % SIZE;
        ctx.globalAlpha = 0.35;
        this._stone(this._x(c), this._x(r), cell * 0.44, this.previewColor);
        ctx.globalAlpha = 1;
      }

      // 棋子
      const now = (global.performance || Date).now();
      const age = now - this.lastMoveAt;
      for (let i = 0; i < this.board.length; i++) {
        const v = this.board[i];
        if (!v) continue;
        const r = (i / SIZE) | 0;
        const c = i % SIZE;
        let radius = cell * 0.44;
        if (i === this.lastMove && age < 220) {
          const t = age / 220;
          radius *= 0.55 + 0.45 * (1 - Math.pow(1 - t, 3));
        }
        this._stone(this._x(c), this._x(r), radius, v);
      }

      // 最后一手标记
      if (this.lastMove >= 0 && this.board[this.lastMove]) {
        const r = (this.lastMove / SIZE) | 0;
        const c = this.lastMove % SIZE;
        ctx.beginPath();
        ctx.arc(this._x(c), this._x(r), Math.max(3, cell * 0.13), 0, Math.PI * 2);
        ctx.fillStyle = this.board[this.lastMove] === 1 ? 'rgba(255,255,255,0.9)' : 'rgba(220,60,60,0.95)';
        ctx.fill();
      }

      // 候选/提示标记
      for (const m of this.marks) {
        const r = (m.action / SIZE) | 0;
        const c = m.action % SIZE;
        ctx.beginPath();
        ctx.arc(this._x(c), this._x(r), cell * 0.42, 0, Math.PI * 2);
        ctx.strokeStyle = m.color || 'rgba(226, 88, 34, 0.95)';
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      // 胜利连线
      if (this.winCells && this.winCells.length) {
        const first = this.winCells[0];
        const last = this.winCells[this.winCells.length - 1];
        ctx.beginPath();
        ctx.moveTo(this._x(first[1]), this._x(first[0]));
        ctx.lineTo(this._x(last[1]), this._x(last[0]));
        ctx.strokeStyle = 'rgba(255, 70, 70, 0.9)';
        ctx.lineWidth = Math.max(3, cell * 0.12);
        ctx.lineCap = 'round';
        ctx.stroke();
      }

      if (this.lastMove >= 0 && age < 240) {
        global.requestAnimationFrame(() => this.draw());
      }
    }

    _stone(x, y, radius, color) {
      const ctx = this.ctx;
      ctx.save();
      ctx.beginPath();
      ctx.arc(x, y, radius * 1.08, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(0,0,0,0.18)';
      ctx.filter = 'none';
      ctx.fill();
      const grad = ctx.createRadialGradient(x - radius * 0.35, y - radius * 0.4, radius * 0.15, x, y, radius);
      if (color === 1) {
        grad.addColorStop(0, '#5a5a5a');
        grad.addColorStop(0.55, '#1d1d1d');
        grad.addColorStop(1, '#050505');
      } else {
        grad.addColorStop(0, '#ffffff');
        grad.addColorStop(0.6, '#f4f1ea');
        grad.addColorStop(1, '#cfc9bd');
      }
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fillStyle = grad;
      ctx.fill();
      ctx.lineWidth = 0.8;
      ctx.strokeStyle = color === 1 ? 'rgba(0,0,0,0.6)' : 'rgba(120,110,95,0.55)';
      ctx.stroke();
      ctx.restore();
    }
  }

  function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  global.BoardView = BoardView;
})(typeof window !== 'undefined' ? window : globalThis);
