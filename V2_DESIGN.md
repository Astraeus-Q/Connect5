# v2 设计方案（后端大模型版）

> 状态：**设计完成，尚未训练**（等 v1.1 出结果后再启动）
> 定位：跑在后端的大模型，摆脱浏览器算力限制；与 v1 系列（浏览器版）并行存在。

---

## 1. 双轨制

| 系列 | 参数量 | 运行环境 | 用途 |
|---|---|---|---|
| **v1 / v1.1** | 20 万 | 浏览器 JS 推理 | 离线可用、零部署 |
| **v2** | **116 万** | 后端（GPU 优先，CPU 可跑） | 追求棋力上限 |

两者**输入/输出/数据格式完全一致**，共用同一套训练脚本、MCTS 和评估流程，只是 `--arch` 不同。

---

## 2. 架构选型

| 预设 | 结构 | 参数量 | 说明 |
|---|---|---|---|
| `v1` | 48ch × 4 直筒 | 200,874 | 现行浏览器版 |
| `v2s` | 64ch × 5 **残差** | 502,506 | 保守档，CPU 后端友好 |
| **`v2`** ✅ | **96ch × 6 残差** | **1,160,298** | **推荐** |
| `v2l` | 128ch × 8 残差 | 2,527,082 | 需要大量数据，容易过拟合 |

**为什么选 96ch × 6：**

- 是 v1 的 **5.8 倍**容量，足以修正 v1 的两个短板（价值头学不动、策略表达力不足）
- 116 万参数配上 26 万样本 + 8 倍对称增强，比例仍在合理区间；再往上（`v2l` 253 万）以现有数据量会明显过拟合
- 实测速度可接受（见下表）

---

## 3. 架构细节

```
输入 (3,15,15)
   ↓
stem: Conv3×3(3→96) + BN + ReLU
   ↓
ResBlock × 6:
   Conv3×3(96→96)+BN+ReLU → Conv3×3(96→96)+BN → 加回输入 → ReLU
   ↓
┌──────────────┴──────────────┐
策略头                        价值头
Conv1×1(96→2)+BN+ReLU         Conv1×1(96→1)+BN+ReLU
flatten 450                   flatten 225
FC(450→225) → logits          FC(225→256)+ReLU → FC(256→1) → tanh
```

两处关键升级：

1. **残差块**：6 个残差块 = 12 层卷积，没有残差梯度传不回去，必然训不动。PyTorch 里就是 `x + f(x)`，成本几乎为零。
2. **价值头加深**（64 → 256 隐层）：v1 的价值头太小是 value loss 降不下去的原因之一，v2 给它更多容量。

**没动的部分**（保证兼容）：输入表示、输出维度、MCTS、数据格式、损失函数——一行都不用改。

---

## 4. 实测性能

本机 CPU（且同时跑着 12 进程自对弈，属于偏保守的数字）：

| 架构 | 单样本前向 | batch=16 摊薄 | 128 sims | 400 sims |
|---|---|---|---|---|
| v1 | 77 ms | 6.3 ms | 0.8 s | 2.5 s |
| v2s | 60 ms | 12.8 ms | 1.6 s | 5.1 s |
| **v2** | 226 ms | **20.4 ms** | **2.6 s** | **8.2 s** |

结论：

- **GPU 空闲时**（预计快 5-10 倍）：v2 开 400-800 sims，单步 1-2 秒 ✓
- **只有 CPU 时**：v2 开 128-256 sims，单步 2.6-5 秒，仍然可用 ✓
- ⚠️ 当前这台机器 GPU 被 vllm 占满（113/119.7 GB），`pick_device()` 已自动回退 CPU

---

## 5. 后端服务

`ai/server.py`（FastAPI，依赖已装：fastapi 0.136 / uvicorn 0.52）

| 接口 | 作用 |
|---|---|
| `GET /health` | 健康检查，返回当前模型与设备 |
| `GET /info` | 架构、参数量、默认 sims |
| `POST /move` | 传入 225 长度棋盘 → 返回落子、policy、value、耗时 |
| `POST /reload` | 热切换模型（训出新版本不用重启服务） |

启动：

```bash
cd ai
python3 server.py --model ../models/v2.pt --sims 400 --port 8000
# 交互式文档：http://127.0.0.1:8000/docs
```

请求示例：

```bash
curl -X POST http://127.0.0.1:8000/move \
  -H "Content-Type: application/json" \
  -d '{"board": [0,0,...225个...], "player": 1, "sims": 400}'
```

强制手（成五/堵五/活四）在服务端先检查，命中就直接返回，不浪费搜索。

---

## 6. 训练策略

### 数据配方

| 数据 | 样本量 | 作用 |
|---|---|---|
| `heur2000.npz` | 19.8 万 | 棋型基础（防止忘记怎么下棋） |
| `know600.npz` | 6.2 万 | 知识增强搜索教材（v1.1 正在生成） |
| **v2 自对弈（待生成）** | 目标 10 万+ | 用 v2 自己搜索产生，质量更高 |

**关键**：v2 容量大了，需要更多新数据。目标是把"搜索教材"占比从 24% 提到 **40% 以上**，否则大模型学到的还是老一套。

### 超参（针对 116 万参数从头训练）

```bash
--arch v2 --epochs 8 --batch-size 512 --lr 1e-3 \
--warmup-ratio 0.05 --min-lr-ratio 0.05 --weight-decay 1e-4
```

- **batch 512**：比 v1 的 256 大，大模型需要更稳定的梯度
- **warmup 5% + 余弦退火**（已实现，按 step 调度）：大模型刚开始用大 lr 容易训崩，warmup 是标配
- **epochs 8**：容量大，需要更多轮才能收敛
- **梯度裁剪 5.0**：已有

### 两阶段

```
阶段 1（打底）：现有 heur2000 + know600 → 从头训 v2
               目标：acc > 0.55，value loss < 0.35

阶段 2（迭代）：v2 自对弈（sims 128，带知识增强）→ 新数据 → 继续训练
               每代跑评估，得分率超过上一代才采纳
```

### 评估口径

与 v1 保持一致才能对比（30 局，seed=2026）：

```bash
for s in 64 200 400; do
  python3 evaluate.py --model ../models/v2.pt --vs heuristic --games 30 --sims $s --seed 2026
done
```

达标线：**超过 v1.1 的普通档得分率**（v1 是 25.0%）。

---

## 7. 命令行速查

> GPU 可用时（显存 > 1.5 GB，代码会自动检测）加 `--amp` 和更大的 batch；
> 显存被别的进程占满时会自动回退 CPU，无需改命令。

```bash
cd ai

# 从头训练 v2（阶段 1）—— GPU + 混合精度
python3 train.py --arch v2 \
        --data ../data/know600.npz ../data/heur2000.npz \
        --epochs 8 --batch-size 1024 --lr 1e-3 --amp \
        --out ../models/v2.pt --iteration 20 --tag v2

# 对照实验：同样数据训一个 v2l，用 val loss 判断是否过拟合
python3 train.py --arch v2l \
        --data ../data/know600.npz ../data/heur2000.npz \
        --epochs 8 --batch-size 1024 --lr 1e-3 --amp \
        --out ../models/v2l.pt --iteration 20 --tag v2l

# v2 自对弈生成教材（阶段 2）—— 可以搬到 GPU 了
python3 selfplay.py --model ../models/v2.pt --policy mcts \
        --games 400 --workers 12 --sims 128 --seed 30001 \
        --heuristic-value-weight 0.5 --batch-size 32 --device cuda \
        --out ../data/know_v2_400.npz

# 启动后端服务（默认开启混合精度，加 --no-amp 关闭）
python3 server.py --model ../models/v2.pt --sims 400
```

### 新增的两个开关

| 参数 | 作用 |
|---|---|
| `selfplay.py --device {auto,cpu,cuda}` | 自对弈推理设备。`cuda` 用 GPU（建议同时 `--batch-size 32-64`）；`auto` 时多进程仍默认 CPU（避免多个 CUDA context 争抢） |
| `train.py --amp` | 混合精度训练，仅 CUDA 生效，大模型训练再快约一倍 |
| `server.py --no-amp` | 关闭后端推理的混合精度（默认开启） |

> v2 是残差结构，**不支持导出到浏览器**（`fused_weights` 会明确报错）。
> 浏览器继续用 v1 系列的 `weights.json`。

---

## 8. 风险与对策

| 风险 | 对策 |
|---|---|
| 数据不够 → 过拟合 | 先加自对弈数据再训；必要时降到 `v2s` |
| GPU 被 vllm 占满 | 已自动回退 CPU；sims 降到 128-256 |
| 训练时间变长 | 116 万参数在 GPU 上约 1 分钟/epoch，可接受；CPU 上约 10 分钟/epoch |
| 浏览器用不上 v2 | 双轨制：浏览器继续 v1 系列，后端跑 v2，各司其职 |
