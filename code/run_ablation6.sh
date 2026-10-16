#!/bin/bash
# 消融+eta敏感性补齐到全6数据集×5seeds(第七轮只覆盖了ETTh2/ETTm2/elec)
# 缺口: traffic/PEMS03/PEMS04 × {outaff,ridge,comb,pool,rpool} + {eta5,eta500} × 5seeds = 105 runs
# 日志格式与multiseed.log一致: MS|tag|DS|sSEED|Test MSE: ...
cd "$(dirname "$0")" || exit 1
export NONORM=1
LOG=ablation6.log
: > $LOG
FINAL="RECYCLE_COMB=1 RECYCLE_POOL=1 RECYCLE_RPOOL=1"

R() { local DS=$1 CYC=$2 CK=$3 TAG=$4; shift 4
  env $@ python main.py DATA.NAME $DS DATA.PRED_LEN 96 MODEL.NAME CycleNet MODEL.pred_len 96 MODEL.cycle $CYC \
    TRAIN.ENABLE False TRAIN.CHECKPOINT_DIR ./checkpoints/CycleNet/$CK/ TTA.ENABLE True TEST.ENABLE False \
    RESULT_DIR ./results/Recycle_nonorm/ NORM_MODULE.ENABLE False DATA_LOADER.NUM_WORKERS 0 SEED $SEED \
    2>&1 | grep "Test MSE" | sed "s/^/MS|$TAG|$DS|s$SEED|/" >> $LOG
}

# 先小后大: PEMS04 -> PEMS03 -> traffic
for D in "PEMS04 288" "PEMS03 288" "traffic 168"; do
  set -- $D
  for SEED in 2024 2025 2026 2027 2028; do
    R $1 $2 ${1}_96_nonorm_s$SEED outaff RECYCLE_OUTAFF=causal
    R $1 $2 ${1}_96_nonorm_s$SEED ridge
    R $1 $2 ${1}_96_nonorm_s$SEED comb RECYCLE_COMB=1
    R $1 $2 ${1}_96_nonorm_s$SEED pool RECYCLE_COMB=1 RECYCLE_POOL=1
    R $1 $2 ${1}_96_nonorm_s$SEED rpool RECYCLE_COMB=1 RECYCLE_POOL=1 RECYCLE_RPOOL=1
    R $1 $2 ${1}_96_nonorm_s$SEED eta5 $FINAL RECYCLE_COMBETA=5
    R $1 $2 ${1}_96_nonorm_s$SEED eta500 $FINAL RECYCLE_COMBETA=500
    echo "DONE $1 s$SEED ($(date +%H:%M))" >> $LOG
  done
done
echo "ABLATION6_ALL_DONE" >> $LOG
