#!/bin/bash
cd "$(dirname "$0")" || exit 1
export NONORM=1
LOG=subexp2_stdout.log
: > $LOG

R() { # ds cyc tag; env vars via $ENVS (expanded through env)
  local DS=$1 CYC=$2 TAG=$3
  env $ENVS python main.py DATA.NAME $DS DATA.PRED_LEN 96 MODEL.NAME CycleNet MODEL.pred_len 96 MODEL.cycle $CYC \
    TRAIN.ENABLE False TRAIN.CHECKPOINT_DIR ./checkpoints/CycleNet/${DS}_96_nonorm_s2024/ TTA.ENABLE True TEST.ENABLE False \
    RESULT_DIR ./results/Recycle_nonorm/ NORM_MODULE.ENABLE False DATA_LOADER.NUM_WORKERS 0 SEED 2024 \
    2>&1 | grep -E "Test MSE|Avg update time|Number of adaptations" | sed "s/^/SUB|$TAG|$DS|/"
}
FINAL="RECYCLE_COMB=1 RECYCLE_POOL=1 RECYCLE_RPOOL=1 COMBETA=200"

for DS in "ETTm2 96" "traffic 168" "PEMS03 288" "PEMS04 288"; do
  set -- $DS
  ENVS="RECYCLE_COMB=1" R $1 $2 comb
  ENVS="RECYCLE_COMB=1 RECYCLE_POOL=1" R $1 $2 pool
  ENVS="RECYCLE_COMB=1 RECYCLE_POOL=1 RECYCLE_RPOOL=1" R $1 $2 rpool
  ENVS="RECYCLE_OUTAFF=causal" R $1 $2 outaff
  ENVS="RECYCLE_COMB=1 RECYCLE_POOL=1 RECYCLE_RPOOL=1 RECYCLE_COMBETA=5" R $1 $2 eta5
  ENVS="RECYCLE_COMB=1 RECYCLE_POOL=1 RECYCLE_RPOOL=1 RECYCLE_COMBETA=50" R $1 $2 eta50
  ENVS="RECYCLE_COMB=1 RECYCLE_POOL=1 RECYCLE_RPOOL=1 RECYCLE_COMBETA=500" R $1 $2 eta500
  ENVS="RECYCLE_COMB=1 RECYCLE_POOL=1 RECYCLE_RPOOL=1 RECYCLE_COMBETA=200" R $1 $2 recfinal
done
echo "SUBEXP2 ALL DONE" >> $LOG
