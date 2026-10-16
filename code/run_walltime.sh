#!/bin/bash
cd "$(dirname "$0")" || exit 1
export NONORM=1
wall() {  # $1=ds $2=cyc $3=ckpt $4=resultdir-tag $5... extra args
  local DS=$1 CYC=$2 CK=$3 TAG=$4; shift 4
  local T0=$(date +%s)
  python main.py DATA.NAME $DS DATA.PRED_LEN 96 MODEL.NAME CycleNet MODEL.pred_len 96 MODEL.cycle $CYC \
    TRAIN.ENABLE False TRAIN.CHECKPOINT_DIR ./checkpoints/CycleNet/$CK/ TTA.ENABLE True TEST.ENABLE False \
    RESULT_DIR ./results/$TAG NORM_MODULE.ENABLE False DATA_LOADER.NUM_WORKERS 0 SEED 2024 "$@" \
    > /dev/null 2>&1
  local T1=$(date +%s)
  echo "WALL $DS $TAG $((T1-T0))s"
}
# twice: first run warms OS caches, second is the reported one
wall ETTh2 24 ETTh2_96_nonorm_s2024 TAFAS_nonorm/ TTA.SOLVER.BASE_LR 0.001 TTA.SOLVER.WEIGHT_DECAY 0.0001 TTA.TAFAS.GATING_INIT 0.01
wall ETTh2 24 ETTh2_96_nonorm_s2024 PETSA_nonorm/ TTA.SOLVER.BASE_LR 0.005 TTA.SOLVER.WEIGHT_DECAY 0.0001
wall ETTh2 24 ETTh2_96_nonorm_s2024 COSA_nonorm/
RECYCLE_COMB=1 RECYCLE_POOL=1 RECYCLE_RPOOL=1 RECYCLE_COMBETA=200 wall ETTh2 24 ETTh2_96_nonorm_s2024 Recycle_nonorm/
wall electricity 168 electricity_96_nonorm_s2024 TAFAS_nonorm/ TTA.SOLVER.BASE_LR 0.001 TTA.SOLVER.WEIGHT_DECAY 0.0001 TTA.TAFAS.GATING_INIT 0.01
wall electricity 168 electricity_96_nonorm_s2024 PETSA_nonorm/ TTA.SOLVER.BASE_LR 0.005 TTA.SOLVER.WEIGHT_DECAY 0.0001
wall electricity 168 electricity_96_nonorm_s2024 COSA_nonorm/
RECYCLE_COMB=1 RECYCLE_POOL=1 RECYCLE_RPOOL=1 RECYCLE_COMBETA=200 wall electricity 168 electricity_96_nonorm_s2024 Recycle_nonorm/
wall ETTh2 24 ETTh2_96_nonorm_s2024 TAFAS_nonorm/ TTA.SOLVER.BASE_LR 0.001 TTA.SOLVER.WEIGHT_DECAY 0.0001 TTA.TAFAS.GATING_INIT 0.01
wall ETTh2 24 ETTh2_96_nonorm_s2024 PETSA_nonorm/ TTA.SOLVER.BASE_LR 0.005 TTA.SOLVER.WEIGHT_DECAY 0.0001
wall ETTh2 24 ETTh2_96_nonorm_s2024 COSA_nonorm/
RECYCLE_COMB=1 RECYCLE_POOL=1 RECYCLE_RPOOL=1 RECYCLE_COMBETA=200 wall ETTh2 24 ETTh2_96_nonorm_s2024 Recycle_nonorm/
wall electricity 168 electricity_96_nonorm_s2024 TAFAS_nonorm/ TTA.SOLVER.BASE_LR 0.001 TTA.SOLVER.WEIGHT_DECAY 0.0001 TTA.TAFAS.GATING_INIT 0.01
wall electricity 168 electricity_96_nonorm_s2024 PETSA_nonorm/ TTA.SOLVER.BASE_LR 0.005 TTA.SOLVER.WEIGHT_DECAY 0.0001
wall electricity 168 electricity_96_nonorm_s2024 COSA_nonorm/
RECYCLE_COMB=1 RECYCLE_POOL=1 RECYCLE_RPOOL=1 RECYCLE_COMBETA=200 wall electricity 168 electricity_96_nonorm_s2024 Recycle_nonorm/
echo "WALL DONE"
