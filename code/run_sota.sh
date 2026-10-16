#!/bin/bash
# 最终SOTA验证:COMBETA=200, 8基线+ReCycle, 6数据集, 5seeds
cd "$(dirname "$0")" || exit 1
export NONORM=1
rm -f results_sota.csv

for SEED in 2024 2025 2026 2027 2028; do
for ENTRY in "ETTh2 24" "ETTm2 96" "electricity 168" "traffic 168" "PEMS03 288" "PEMS04 288"; do
  DS=$(echo $ENTRY | cut -d' ' -f1); CY=$(echo $ENTRY | cut -d' ' -f2)
  CKPT=./checkpoints/CycleNet/${DS}_96_nonorm_s${SEED}/
  if [ ! -f "${CKPT}checkpoint_best.pth" ]; then
    python main.py DATA.NAME $DS DATA.PRED_LEN 96 MODEL.NAME CycleNet MODEL.pred_len 96 MODEL.cycle $CY \
      TRAIN.ENABLE True TRAIN.CHECKPOINT_DIR $CKPT TTA.ENABLE False TEST.ENABLE False \
      NORM_MODULE.ENABLE False DATA_LOADER.NUM_WORKERS 0 SEED $SEED 2>&1 | tail -1
  fi
  COMMON="DATA.NAME $DS DATA.PRED_LEN 96 MODEL.NAME CycleNet MODEL.pred_len 96 MODEL.cycle $CY TRAIN.CHECKPOINT_DIR $CKPT NORM_MODULE.ENABLE False DATA_LOADER.NUM_WORKERS 0 TRAIN.ENABLE False SEED $SEED"

  RC=$(RECYCLE_COMB=1 RECYCLE_POOL=1 RECYCLE_RPOOL=1 RECYCLE_COMBETA=200 python main.py $COMMON TTA.ENABLE True TEST.ENABLE False RESULT_DIR ./results/Recycle_nonorm/ 2>&1 | grep "^Test MSE" | grep -oE "0\.[0-9]+" | head -1)
  echo "$SEED,$DS,$RC" >> results_sota.csv
  echo "s$SEED $DS | RC $RC"
done
done
echo SOTA_DONE
