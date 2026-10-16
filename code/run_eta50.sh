#!/bin/bash
# eta=50扫描点补多种子(全6数据集×5seeds=30 runs) — 使eta敏感性图/正文全部为5-seed均值、4点{5,50,200,500}
cd "$(dirname "$0")" || exit 1
export NONORM=1
LOG=eta50.log
: > $LOG

R() { local DS=$1 CYC=$2 SEED=$3
  RECYCLE_COMB=1 RECYCLE_POOL=1 RECYCLE_RPOOL=1 RECYCLE_COMBETA=50 python main.py DATA.NAME $DS DATA.PRED_LEN 96 MODEL.NAME CycleNet MODEL.pred_len 96 MODEL.cycle $CYC \
    TRAIN.ENABLE False TRAIN.CHECKPOINT_DIR ./checkpoints/CycleNet/${DS}_96_nonorm_s$SEED/ TTA.ENABLE True TEST.ENABLE False \
    RESULT_DIR ./results/Recycle_nonorm/ NORM_MODULE.ENABLE False DATA_LOADER.NUM_WORKERS 0 SEED $SEED \
    2>&1 | grep "Test MSE" | sed "s/^/MS|eta50|$DS|s$SEED|/" >> $LOG
}

for ENTRY in "ETTh2 24" "ETTm2 96" "electricity 168" "traffic 168" "PEMS03 288" "PEMS04 288"; do
  set -- $ENTRY
  for SEED in 2024 2025 2026 2027 2028; do
    R $1 $2 $SEED
  done
done
echo "ETA50_ALL_DONE" >> $LOG
