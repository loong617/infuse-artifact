#!/bin/bash
cd "$(dirname "$0")" || exit 1
export NONORM=1
timing() {  # $1=ds $2=cyc $3=ckpt $4=resultdir-tag $5... extra args
  local DS=$1 CYC=$2 CK=$3 TAG=$4; shift 4
  python main.py DATA.NAME $DS DATA.PRED_LEN 96 MODEL.NAME CycleNet MODEL.pred_len 96 MODEL.cycle $CYC \
    TRAIN.ENABLE False TRAIN.CHECKPOINT_DIR ./checkpoints/CycleNet/$CK/ TTA.ENABLE True TEST.ENABLE False \
    RESULT_DIR ./results/$TAG NORM_MODULE.ENABLE False DATA_LOADER.NUM_WORKERS 0 SEED 2024 "$@" \
    2>&1 | grep -E "Test MSE|adaptations|update time" | while read -r ln; do echo "TIME $DS $TAG $ln"; done
}
timing ETTh2 24 ETTh2_96_nonorm_s2024 TAFAS_nonorm/ TTA.SOLVER.BASE_LR 0.001 TTA.SOLVER.WEIGHT_DECAY 0.0001 TTA.TAFAS.GATING_INIT 0.01
timing ETTh2 24 ETTh2_96_nonorm_s2024 PETSA_nonorm/ TTA.SOLVER.BASE_LR 0.005 TTA.SOLVER.WEIGHT_DECAY 0.0001
timing ETTh2 24 ETTh2_96_nonorm_s2024 COSA_nonorm/
RECYCLE_COMB=1 RECYCLE_POOL=1 RECYCLE_RPOOL=1 RECYCLE_COMBETA=200 timing ETTh2 24 ETTh2_96_nonorm_s2024 Recycle_nonorm/
timing electricity 168 electricity_96_nonorm_s2024 TAFAS_nonorm/ TTA.SOLVER.BASE_LR 0.001 TTA.SOLVER.WEIGHT_DECAY 0.0001 TTA.TAFAS.GATING_INIT 0.01
timing electricity 168 electricity_96_nonorm_s2024 PETSA_nonorm/ TTA.SOLVER.BASE_LR 0.005 TTA.SOLVER.WEIGHT_DECAY 0.0001
timing electricity 168 electricity_96_nonorm_s2024 COSA_nonorm/
RECYCLE_COMB=1 RECYCLE_POOL=1 RECYCLE_RPOOL=1 RECYCLE_COMBETA=200 timing electricity 168 electricity_96_nonorm_s2024 Recycle_nonorm/
echo "TIMING DONE"
