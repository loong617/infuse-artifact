#!/bin/bash
# hist 端点扫描补全: 4数据集 x 2端点(480,2880) x 5种子
cd "$(dirname "$0")" || exit 1
export NONORM=1
LOG=hist4.log
: > $LOG
FINAL="RECYCLE_COMB=1 RECYCLE_POOL=1 RECYCLE_RPOOL=1 RECYCLE_COMBETA=200"
for D in "ETTm2 96 ETTm2_96_nonorm_s{S}" "traffic 168 traffic_96_nonorm_s{S}" "PEMS03 288 PEMS03_96_nonorm_s{S}" "PEMS04 288 PEMS04_96_nonorm_s{S}"; do
  set -- $D
  for SEED in 2024 2025 2026 2027 2028; do
    CK=${3/\{S\}/$SEED}
    [ ! -d "./checkpoints/CycleNet/$CK" ] && continue
    for H in 480 2880; do
      env $FINAL RECYCLE_HIST=$H python main.py DATA.NAME $1 DATA.PRED_LEN 96 MODEL.NAME CycleNet MODEL.pred_len 96 MODEL.cycle $2 \
        TRAIN.ENABLE False TRAIN.CHECKPOINT_DIR ./checkpoints/CycleNet/$CK/ TTA.ENABLE True TEST.ENABLE False \
        RESULT_DIR ./results/Recycle_nonorm/ NORM_MODULE.ENABLE False DATA_LOADER.NUM_WORKERS 0 SEED $SEED \
        2>&1 | grep "Test MSE" | sed "s/^/H4|$1|hist$H|s$SEED|/" >> $LOG
    done
  done
done
echo "HIST4 DONE" >> $LOG
