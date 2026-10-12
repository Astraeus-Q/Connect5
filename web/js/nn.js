/* 纯 JS 的卷积神经网络推理（无任何依赖）
 * 加载 ai/export.py 导出的 weights.json（BN 已融合进卷积，故只需 conv/relu/linear）。
 */
(function (global) {
  'use strict';

  const SIZE = 15;
  const HW = SIZE * SIZE;

  function base64ToFloats(b64) {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return new Float32Array(bytes.buffer);
  }

  /** im2col + GEMM 的 3x3 / 1x1 卷积，后接 ReLU */
  function convRelu(input, xc, weight, bias, oc, k) {
    const pad = (k - 1) >> 1;
    const K = xc * k * k;
    const cols = new Float32Array(K * HW);
    for (let c = 0; c < xc; c++) {
      for (let kh = 0; kh < k; kh++) {
        for (let kw = 0; kw < k; kw++) {
          const rowBase = ((c * k + kh) * k + kw) * HW;
          const yStart = kh - pad;
          const xStart = kw - pad;
          for (let y = 0; y < SIZE; y++) {
            const yy = y + yStart;
            if (yy < 0 || yy >= SIZE) continue;           // 补零，保持 0
            const inRow = (c * SIZE + yy) * SIZE;
            const outRow = rowBase + y * SIZE;
            for (let xi = 0; xi < SIZE; xi++) {
              const xxi = xi + xStart;
              if (xxi < 0 || xxi >= SIZE) continue;
              cols[outRow + xi] = input[inRow + xxi];
            }
          }
        }
      }
    }

    const out = new Float32Array(oc * HW);
    for (let o = 0; o < oc; o++) {
      const wOff = o * K;
      const oOff = o * HW;
      const b = bias[o];
      for (let p = 0; p < HW; p++) out[oOff + p] = b;
      for (let kk = 0; kk < K; kk++) {
        const wv = weight[wOff + kk];
        if (wv === 0) continue;
        const cOff = kk * HW;
        for (let p = 0; p < HW; p++) out[oOff + p] += wv * cols[cOff + p];
      }
      for (let p = 0; p < HW; p++) if (out[oOff + p] < 0) out[oOff + p] = 0;   // ReLU
    }
    return out;
  }

  function linear(x, weight, bias, inDim, outDim, relu) {
    const out = new Float32Array(outDim);
    for (let o = 0; o < outDim; o++) {
      let s = bias[o];
      const off = o * inDim;
      for (let i = 0; i < inDim; i++) s += weight[off + i] * x[i];
      out[o] = relu && s < 0 ? 0 : s;
    }
    return out;
  }

  class GomokuNet {
    constructor(arch, weights) {
      this.arch = arch;
      this.w = weights;
      this.channels = arch.channels;
      this.nBlocks = arch.n_blocks;
    }

    /** state: Float32Array(3*225) -> { policy: Float32Array(225), value: number } */
    forward(state) {
      const W = this.w;
      const C = this.channels;
      let h = convRelu(state, 3, W['stem.weight'], W['stem.bias'], C, 3);
      for (let i = 0; i < this.nBlocks; i++) {
        h = convRelu(h, C, W[`block${i}.weight`], W[`block${i}.bias`], C, 3);
      }

      // policy head
      const p = convRelu(h, C, W['p_conv.weight'], W['p_conv.bias'], 2, 1);
      const logits = linear(p, W['p_fc.weight'], W['p_fc.bias'], 2 * HW, HW, false);
      // mask 非法点后 softmax
      const policy = new Float32Array(HW);
      let max = -Infinity;
      for (let i = 0; i < HW; i++) {
        const legal = state[i] === 0 && state[HW + i] === 0;
        const v = legal ? logits[i] : -1e9;
        logits[i] = v;
        if (v > max) max = v;
      }
      let sum = 0;
      for (let i = 0; i < HW; i++) {
        const e = Math.exp(logits[i] - max);
        policy[i] = e;
        sum += e;
      }
      for (let i = 0; i < HW; i++) policy[i] /= sum;

      // value head
      const v1 = convRelu(h, C, W['v_conv.weight'], W['v_conv.bias'], 1, 1);
      const v2 = linear(v1, W['v_fc1.weight'], W['v_fc1.bias'], HW, 64, true);
      const v3 = linear(v2, W['v_fc2.weight'], W['v_fc2.bias'], 64, 1, false);
      const value = Math.tanh(v3[0]);
      return { policy, value };
    }

    static fromJSON(payload) {
      const weights = {};
      for (const key of Object.keys(payload.weights)) {
        weights[key] = base64ToFloats(payload.weights[key]);
      }
      return new GomokuNet(payload.arch, weights);
    }

    /** 从 URL 加载（需通过 http(s) 访问；file:// 下请改用文件选择） */
    static async load(url) {
      const res = await fetch(url, { cache: 'no-store' });
      if (!res.ok) throw new Error(`权重加载失败 (HTTP ${res.status})`);
      const payload = await res.json();
      if (!payload || !payload.weights) throw new Error('权重文件格式不正确');
      return { net: GomokuNet.fromJSON(payload), meta: payload.meta || {}, arch: payload.arch };
    }

    /** 从本地文件（File 对象）加载 */
    static loadFromFile(file) {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
          try {
            const payload = JSON.parse(reader.result);
            resolve({ net: GomokuNet.fromJSON(payload), meta: payload.meta || {}, arch: payload.arch });
          } catch (e) {
            reject(e);
          }
        };
        reader.onerror = () => reject(new Error('文件读取失败'));
        reader.readAsText(file);
      });
    }
  }

  global.GomokuNet = GomokuNet;
  global.GomokuNetOps = { convRelu, linear, base64ToFloats };
})(typeof window !== 'undefined' ? window : globalThis);
