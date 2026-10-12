# 训练指南（手动版）

这份文档讲清楚**每一步在做什么、会产生什么文件、要花多久、怎么判断有没有变强**。
所有命令都在 `ai/` 目录下执行。

```bash
cd /home/pingtaibu_2/work/CodeBuddy/Connect5/ai
```

---

## 0. 训练到底在干什么

网络初始权重是随机的（真的什么都不会），训练分三个阶段：

```
阶段 1  冷启动         规则启发式 AI 自己跟自己下  →  数据  →  训出 iter0
                      （只有棋型常识，没有"棋感"）

阶段 2  迭代自对弈     上一代模型 + MCTS 自己跟自己下  →  新数据  →  再训
        （重复 N 轮）  搜索比纯模仿看得更远，每代把搜索结果蒸馏回网络

阶段 3  导出          把 BatchNorm 折进卷积 → weights.json → 网页刷新即变强
```

一句话：**用搜索产生更高质量的对局，再把"搜索会怎么走"教给网络，让网络下次不用搜索也能走出接近的水平。**

### 每一轮的文件流向

| 步骤 | 命令 | 产出 |
| --- | --- | --- |
| 自对弈 | `selfplay.py` | `data/iter{N}.npz`（局面 + 落子分布 + 胜负） |
| 训练 | `train.py` | `models/iter{N}.pt` 和 `models/latest.pt` |
| 导出 | `export.py` | `web/models/weights.json`（网页 AI 的大脑） |
| 评估 | `evaluate.py` | 打印胜率（不产生文件） |

> ⚠️ **只有执行了 `export.py`，网页里的 AI 才会变。**

---

## 1. 环境准备

```bash
pip install -r ai/requirements.txt     # torch + numpy
```

设备选择是自动的（`model.pick_device`）：有 GPU 且剩余显存 > 1.5 GB 就用 GPU，否则回退 CPU。
想强制 CPU：给 `train.py` / `evaluate.py` 加 `--cpu`。

---

## 2. 五条路线，选一条跑

### 路线 A：一条命令全自动（推荐第一次用）

```bash
cd ai
python3 pipeline.py --iters 7 --games 150 --sims 64 --epochs 3 --workers 12 --eval-games 20
```

- 第 0 轮用启发式自对弈冷启动，第 1~6 轮用 MCTS 自对弈
- 每轮结束自动导出 + 评估
- 想后台跑：`nohup python3 pipeline.py ... > ../train.log 2>&1 &`，然后 `tail -f ../train.log`

常用参数：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--iters` | 5 | 总轮数 |
| `--start` | 0 | 起始轮次（接着上次的模型继续用 `--start 4`） |
| `--games` | 120 | 每轮自对弈局数 |
| `--games0` | 0 | 第 0 轮局数，0 表示同 `--games` |
| `--sims` | 48 | 每步 MCTS 模拟次数（越大越慢越强） |
| `--epochs` | 3 | 每轮训练轮数 |
| `--workers` | 1 | 自对弈并行进程数（建议 = CPU 核数的一半） |
| `--replay-window` | 3 | 只用最近几个数据文件训练 |
| `--eval-games` | 20 | 每轮评估局数，0 = 跳过 |
| `--extra-data` | 无 | 额外数据文件（可多次填），如 `../data/heur_lt2000.npz` |
| `--heuristic-value-weight` | 0.0 | 见路线 D |
| `--h-tau` / `--h-tau-open` | 45 / 90 | 冷启动启发式温度，越小越接近贪心 |

### 路线 B：分步手动（想看清每一步）

```bash
# ① 冷启动数据：启发式自对弈 300 局（约 10 秒）
python3 selfplay.py --policy heuristic --games 300 --workers 12 --seed 0 --out ../data/iter0.npz

# ② 训出 iter0
python3 train.py --data ../data/iter0.npz --epochs 3 --lr 1e-3 --out ../models/iter0.pt --iteration 0

# ③ MCTS 自对弈（第 1 轮，约 8 分钟）
python3 selfplay.py --model ../models/latest.pt --policy mcts \
        --games 150 --workers 12 --sims 64 --seed 1000 --out ../data/iter1.npz

# ④ 在 iter0+iter1 上训练
python3 train.py --data ../data/iter0.npz ../data/iter1.npz \
        --init ../models/iter0.pt --epochs 3 --lr 1e-3 --out ../models/iter1.pt --iteration 1

# ⑤ 导出到网页
python3 export.py --model ../models/iter1.pt --out ../web/models/weights.json --iteration 1 --tag iter1

# ⑥ 评估
python3 evaluate.py --model ../models/iter1.pt --vs heuristic --games 20 --sims 64
```

第 2 轮开始重复 ③④⑤⑥，把编号 +1、`--model` / `--init` 指向上一轮即可。

### 路线 C：5 分钟冒烟（只想确认流程能跑通）

```bash
python3 selfplay.py --policy heuristic --games 50 --workers 4 --out /tmp/smoke.npz
python3 train.py --data /tmp/smoke.npz --epochs 1 --out /tmp/smoke.pt
python3 evaluate.py --model /tmp/smoke.pt --vs random --games 4 --sims 8
```

### 路线 D：知识增强搜索蒸馏（当前棋力最高的配方）

普通自对弈的搜索价值全靠网络，网络弱 → 搜索也弱 → 数据质量上不去，容易原地打转。
这个开关让自对弈时叶子价值变成：

```
叶子价值 = (1-w) × 网络价值 + w × 棋型评估（活四/活三/冲四的量化差）
```

搜索立刻具备棋型常识，产出的对局质量更高，再把它的落子分布蒸馏回网络。
**网页端推理成本完全不变**（那里还是纯网络 + MCTS）。

```bash
# 用 v1 当起点，开知识增强搜索生成数据（600 局，约 10 分钟）
python3 selfplay.py --model ../models/v1.pt --policy mcts \
        --games 600 --workers 12 --sims 64 --seed 777 \
        --heuristic-value-weight 0.5 --out ../data/know600.npz

# 蒸馏训练
python3 train.py \
        --data ../data/know600.npz ../data/heur2000.npz \
        --init ../models/v1.pt --epochs 5 --lr 5e-4 --out ../models/v2.pt --iteration 2

# 评估（30 局更稳）
python3 evaluate.py --model ../models/v2.pt --vs heuristic --games 30 --sims 64 --seed 1

# 比 v1 的 21.7% 好就导出
python3 export.py --model ../models/v2.pt --out ../web/models/weights.json --iteration 2 --tag v2
```

> 说明：这条路线是我最后加的，**尚未实测胜率**（脚本改动只做过小规模运行验证）。
> 建议先 `--games 100` 跑一小批确认没问题，再放大。

### 路线 E：用网页对局反哺（数据闭环）

1. 网页「对局记录」→ **导出训练数据** → 得到 `connect5-training-*.json`
2. 转成训练样本：

```bash
python3 import_games.py --input ../data/human/connect5-training.json \
                        --out ../data/human.npz --winners-only
```

`--winners-only` 表示只保留获胜方的落子（避免把坏手当教材）。

3. 与自对弈数据一起训练（权重给小一点，避免少量人类数据带偏）：

```bash
python3 train.py --data ../data/iter3.npz ../data/human.npz \
                 --init ../models/latest.pt --epochs 2 --lr 3e-4 --out ../models/final_h.pt
```

---

## 3. 脚本参数速查

| 脚本 | 干什么 | 关键参数 |
| --- | --- | --- |
| `selfplay.py` | 自对弈产数据 | `--policy mcts/heuristic/random`、`--model`、`--games`、`--sims`、`--workers`、`--seed`、`--out`、`--temp-moves/--temp-final`（温度退火）、`--opening-plies`（开局随机手数）、`--heuristic-value-weight` |
| `train.py` | 训练 | `--data`（多个文件/目录）、`--init`（继续训练）、`--epochs`、`--lr`、`--batch-size`、`--weight-decay`、`--value-weight`、`--no-augment`、`--cpu`、`--out` |
| `export.py` | 导出网页权重 | `--model`、`--out`、`--iteration`、`--tag` |
| `evaluate.py` | 对战评估 | `--model`、`--vs random/heuristic/network/mcts`、`--vs-model`、`--kind mcts/network`、`--games`、`--sims`、`--seed` |
| `import_games.py` | 网页对局 → 训练样本 | `--input`、`--out`、`--winners-only`、`--min-moves` |
| `pipeline.py` | 上面四步串起来 | 见路线 A 的表 |

---

## 4. 数据格式

**训练样本（npz / jsonl）**

| 字段 | 形状 | 含义 |
| --- | --- | --- |
| `states` | (N, 3, 15, 15) float32 | 落子前的局面，通道 = 自己 / 对手 / 行棋方常数平面 |
| `pis` | (N, 225) float32 | 落子目标分布（MCTS 访问计数归一化；启发式数据是 top-k 采样分布） |
| `z` | (N,) float32 | 该局从「这一步的行棋方」看的结果：+1 胜 / -1 负 / 0 平 |

**网页导出的对局 JSON**（`import_games.py` 直接吃这个）

```json
{ "app": "connect5", "games": [
    { "moves": [112, 113, ...], "winner": 1, "playerColor": 1, "result": "win", "mode": "pve" }
]}
```

`moves` 用 `row * 15 + col` 编码；`winner` 是 1 黑 / 2 白 / 0 平。

---

## 5. 怎么判断模型变强了

```bash
python3 evaluate.py --model ../models/latest.pt --vs heuristic --games 30 --sims 64 --seed 1
```

输出示例：

```
结果: 胜 8 / 负 15 / 平 7  胜率 0.348  得分率 0.383
```

- **胜率** = 胜 / (胜+负)；**得分率** = (胜 + 0.5×平) / 总局数，更公平
- 双方各半先手，开局随机 2 手，所以要多局才稳：**20 局的噪声在 ±10% 量级，30 局以上再下结论**
- 想看绝对棋力，先打 `--vs random`（应当接近 100%），再打 `--vs heuristic`
- 训练日志里同步看两个数：
  - `acc`（策略预测命中率）：从 0.03 → 0.58 说明网络在稳定学会落子；
  - `value`（价值 MSE）：越接近 0 说明局面判断越准

**注意：acc 高 ≠ 棋力强。** 实测过两个模型：acc 0.596 的那个对贪心基线胜率反而低于 acc 0.356 的那个（19% vs 29%）——因为它学的是"采样策略"而不是最优策略。最终一定要以 `evaluate.py` 的对战胜率为准。

---

## 6. 实测耗时参考

本机 20 核 CPU + 一张被别的进程占掉大半显存的 GPU：

| 操作 | 耗时 |
| --- | --- |
| 启发式自对弈 2000 局（12 workers） | 37 秒 |
| MCTS 自对弈 150 局，sims=64（12 workers） | 约 8 分钟 |
| 训练 20 万样本 5 epoch（GPU） | 约 55 秒 |
| 训练 3 万样本 3 epoch（CPU） | 约 75 秒 |
| 导出权重 | 1 秒 |
| 评估 20 局 sims=64 | 6 秒 |

---

## 7. 常见坑

1. **网页 AI 没变化**
   忘了 `export.py`；或浏览器缓存——刷新时用 `Ctrl+F5`。

2. **加载旧模型报形状不匹配**
   改过 `channels` / `n_blocks` 后，旧 checkpoint 不能直接 `--init`。
   删掉 `models/*.pt` 重新训，或保持架构不变。

3. **CUDA out of memory**
   代码已自动检测剩余显存并回退 CPU；也可以显式加 `--cpu`。

4. **数据被覆盖**
   `iter{N}.npz` 按轮次命名，重跑同一轮会覆盖旧数据。重要数据另存文件名。

5. **内存**
   20 万样本加载后约占 500 MB+，样本再翻倍时注意内存。

6. **日志文件**
   `train.log`、`step*.log` 都是过程日志，随时可删，不影响训练。

---

## 8. 当前仓库状态

**已清理**：早期未收敛的迭代产物与训练日志都已删除，仓库只保留当前的 v1。

现在剩下的：

| 路径 | 内容 |
| --- | --- |
| `data/heur2000.npz` | 启发式自对弈 2000 局，**19.8 万样本**，v1 的主要教材 |
| `models/v1.pt` | 当前模型：用 heur2000 从头训 6 epoch |
| `models/latest.pt` | 等于 v1（`train.py` 总会自动更新它，pipeline 靠它做 `--init`） |
| `web/models/weights.json` | 已导出为 v1（`meta.iteration = 1`） |

**v1 实测成绩**（对手：启发式贪心，30 局，seed=2026）

| 难度 | sims | 胜 / 负 / 平 | 得分率 |
| --- | --- | --- | --- |
| 简单 | 16 | 6 / 19 / 5 | **28.3%** |
| 普通 | 64 | 4 / 19 / 7 | **25.0%** |
| 困难 | 240 | 4 / 19 / 7 | **25.0%** |
| vs 随机 | 32 | 10 / 0 / 0 | **100%** |

策略 acc **0.536**、policy loss **1.92**、value loss 0.47。

**判断基准**：新模型在**普通档（sims=64，30 局）得分率 > 25.0%** 才值得 `export.py` 覆盖网页权重。
注意 30 局仍有 ±5% 噪声（同一模型换 seed 曾测出 21.7% 与 25.0%），接近时换 seed 复测一次再决定。

完整记录见 [MODEL_HISTORY.md](MODEL_HISTORY.md)——每训出一代就往里追加一节。

---

## 9. 建议的起手式

**现在最该做的**：以 v1 为地基跑迭代自对弈。网络已有棋型概念，MCTS 先验终于有信息量了，
迭代循环才转得动（早期那批失败就是卡在这里）：

```bash
python3 pipeline.py --start 2 --iters 6 --games 150 --sims 64 \
        --epochs 4 --workers 12 --replay-window 8 --eval-games 30 \
        --extra-data ../data/heur2000.npz
```

- `--start 2`：跳过冷启动轮，直接 MCTS 自对弈，并自动 `--init models/latest.pt`（也就是 v1）
- `--extra-data ../data/heur2000.npz`：把棋型数据**永久钉在训练集里**，不会再被窗口挤掉
  （这正是上一轮 7 轮白跑的根因）
- `--replay-window 8`：自对弈数据也全程保留

想另起炉灶走路线 A；想试更强的搜索教材走路线 D。
不管哪条，**每轮结束都跑 `evaluate.py`，得分率超过 v1 的 21.7% 再 `export.py`**。
