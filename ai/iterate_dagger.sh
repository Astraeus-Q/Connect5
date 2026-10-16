#!/bin/bash
# 迭代 DAgger：模型走局面 -> 专家标注 -> 训练 -> 评估 -> 若仍有提升则继续
# 停止条件：本轮得分率相对上一轮提升 < 0.5 点（500 局噪声边缘），或到达 MAXR 轮
set -u
cd "$(dirname "$0")"

BASE=33.1                                  # 第 1 轮（v2.3_test_dagger）的成绩
MODEL=../models/experiments/v2_3_test_dagger.pt
ROUND=2
MAXR=5

while [ $ROUND -le $MAXR ]; do
  TAG=v2.3_test_dagger$ROUND
  DATA=../data/dagger_r$ROUND.npz
  echo "=== ROUND $ROUND : $TAG  $(date +%H:%M:%S) ==="

  # 1) 生成：用上一轮模型走局面，启发式专家重新标注动作
  python3 selfplay.py --model $MODEL --policy mcts --sims 64 --games 10000 \
      --workers 6 --device cuda --batch-size 32 \
      --relabel heuristic --relabel-tau 0 --no-root-noise \
      --temp-moves 0 --temp-final 0 --opening-plies 2 \
      --seed $((1000 + ROUND)) --out $DATA \
      > ../logs/selfplay/selfplay_dagger_r$ROUND.log 2>&1

  # 2) 训练：与 v2.2 / 第 1 轮完全相同的超参
  python3 train.py --arch v2 --data $DATA \
      --epochs 6 --lr 1e-3 --batch-size 1024 --amp --value-weight 1.0 \
      --out ../models/experiments/$TAG.pt --iteration $((79 + ROUND)) --tag $TAG \
      > ../logs/train/train_$TAG.log 2>&1

  # 3) 评估：统一口径 500 局
  SCORE=$(python3 evaluate.py --model ../models/experiments/$TAG.pt \
      --vs heuristic --games 500 --sims 64 --seed 2026 2>&1 | tail -1)
  echo "$SCORE" > ../logs/eval/$TAG-vs-heuristic.log
  echo "  $SCORE"

  # 4) 判断是否继续
  NEW=$(echo "$SCORE" | python3 -c "import sys,re; m=re.search(r'得分率 ([\d.]+)', sys.stdin.read()); print(round(float(m.group(1))*100,1) if m else 0)")
  echo "  ROUND $ROUND => $NEW %  (prev $BASE %)"

  OK=$(python3 -c "print('y' if ($NEW - $BASE) >= 0.5 else 'n')")
  if [ "$OK" = "n" ]; then
    echo "=== 瓶颈：本轮 $NEW% 对上轮 $BASE%，提升不足 0.5 点，停止迭代 ==="
    break
  fi

  BASE=$NEW
  MODEL=../models/experiments/$TAG.pt
  ROUND=$((ROUND + 1))
done
echo "=== ITERATION DONE $(date +%H:%M:%S) ==="
