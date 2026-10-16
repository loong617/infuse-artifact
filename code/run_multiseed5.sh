#!/bin/bash
# 子实验多种子:消融5配置+eta5点+hist3点 x 3数据集(ETTh2/ETTm2/Elec) x 种子2025/2026
cd "$(dirname "$0")" || exit 1
export NONORM=1
LOG=multiseed.log
: > $LOG
FINAL="RECYCLE_COMB=1 RECYCLE_POOL=1 RECYCLE_RPOOL=1"

R() { local DS=$1 CYC=$2 CK=$3 TAG=$4; shift 4
  env $@ python main.py DATA.NAME $DS DATA.PRED_LEN 96 MODEL.NAME CycleNet MODEL.pred_len 96 MODEL.cycle $CYC \
    TRAIN.ENABLE False TRAIN.CHECKPOINT_DIR ./checkpoints/CycleNet/$CK/ TTA.ENABLE True TEST.ENABLE False \
    RESULT_DIR ./results/Recycle_nonorm/ NORM_MODULE.ENABLE False DATA_LOADER.NUM_WORKERS 0 SEED $SEED \
    2>&1 | grep "Test MSE" | sed "s/^/MS|$TAG|$DS|s$SEED|/" >> $LOG
}

for SEED in 2025 2026 2027 2028; do
  # ablation: 3 datasets x (outaff, ridge, comb, pool, rpool) [final eta200已有主表数据]
  for D in "ETTh2 24 ETTh2_96_nonorm_s$SEED" "ETTm2 96 ETTm2_96_nonorm_s$SEED" "electricity 168 electricity_96_nonorm_s$SEED"; do
    set -- $D
    R $1 $2 $3 outaff RECYCLE_OUTAFF=causal
    R $1 $2 $3 ridge
    R $1 $2 $3 comb RECYCLE_COMB=1
    R $1 $2 $3 pool RECYCLE_COMB=1 RECYCLE_POOL=1
    R $1 $2 $3 rpool RECYCLE_COMB=1 RECYCLE_POOL=1 RECYCLE_RPOOL=1
    R $1 $2 $3 eta5 $FINAL RECYCLE_COMBETA=5
    R $1 $2 $3 eta500 $FINAL RECYCLE_COMBETA=500
    R $1 $2 $3 hist480 $FINAL RECYCLE_HIST=480
    R $1 $2 $3 hist2880 $FINAL RECYCLE_HIST=2880
  done
done
echo "MULTISEED DONE" >> $LOG
