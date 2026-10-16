#!/bin/bash
# 最终版:8个纯TTA基线 + ReCycle,6数据集,5seeds
cd "$(dirname "$0")" || exit 1
export NONORM=1
rm -f results_final8.csv

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

  V=$(python main.py $COMMON TTA.ENABLE False TEST.ENABLE True TEST.BATCH_SIZE 20000 TEST.DROP_LAST False 2>&1 | grep test_mse | grep -oE "0\.[0-9]+" | head -1)
  T=$(python main.py $COMMON TTA.ENABLE True TEST.ENABLE False RESULT_DIR ./results/TAFAS_nonorm/ TTA.SOLVER.BASE_LR 0.001 TTA.SOLVER.WEIGHT_DECAY 0.0001 TTA.TAFAS.GATING_INIT 0.01 2>&1 | grep "^Test MSE" | grep -oE "0\.[0-9]+" | head -1)
  P=$(python main.py $COMMON TTA.ENABLE True TEST.ENABLE False RESULT_DIR ./results/PETSA_nonorm/ TTA.SOLVER.BASE_LR 0.005 TTA.SOLVER.WEIGHT_DECAY 0.0001 2>&1 | grep "^Test MSE" | grep -oE "0\.[0-9]+" | head -1)
  CS=$(python main.py $COMMON TTA.ENABLE True TEST.ENABLE False RESULT_DIR ./results/COSA_nonorm/ 2>&1 | grep "^Test MSE" | grep -oE "0\.[0-9]+" | head -1)
  DA=$(python main.py $COMMON TTA.ENABLE True TEST.ENABLE False RESULT_DIR ./results/DAdapterOfficial_nonorm/ 2>&1 | grep "^Test MSE" | grep -oE "0\.[0-9]+" | head -1)
  L=$(python main.py $COMMON TTA.ENABLE True TEST.ENABLE False RESULT_DIR ./results/LoRA_nonorm/ 2>&1 | grep "^Test MSE" | grep -oE "0\.[0-9]+" | head -1)
  RV=$(python main.py $COMMON TTA.ENABLE True TEST.ENABLE False RESULT_DIR ./results/RevIN_nonorm/ 2>&1 | grep "^Test MSE" | grep -oE "0\.[0-9]+" | head -1)
  SO=$(python main.py $COMMON TTA.ENABLE True TEST.ENABLE False RESULT_DIR ./results/SOLID_nonorm/ 2>&1 | grep "^Test MSE" | grep -oE "0\.[0-9]+" | head -1)
  PR=$(python main.py $COMMON TTA.ENABLE True TEST.ENABLE False RESULT_DIR ./results/ProceedTTA_nonorm/ 2>&1 | grep "^Test MSE" | grep -oE "0\.[0-9]+" | head -1)
  RC=$(RECYCLE_COMB=1 RECYCLE_POOL=1 RECYCLE_RPOOL=1 RECYCLE_COMBETA=5 python main.py $COMMON TTA.ENABLE True TEST.ENABLE False RESULT_DIR ./results/Recycle_nonorm/ 2>&1 | grep "^Test MSE" | grep -oE "0\.[0-9]+" | head -1)

  echo "$SEED,$DS,$V,$T,$P,$CS,$DA,$L,$RV,$SO,$PR,$RC" >> results_final8.csv
  echo "s$SEED $DS | RC $RC"
done
done
echo FINAL8_DONE
