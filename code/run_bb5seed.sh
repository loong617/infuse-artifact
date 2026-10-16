#!/bin/bash
# Table 5 (backbone generality) 5-seed sweep: RCF-DLinear host
# seeds 2024 uses the original no-suffix checkpoint; 2025-2028 train fresh per-seed backbones
cd "$(dirname "$0")" || exit 1
export NONORM=1
LOG=bb5seed.log
CSV=results_bb5seed.csv
: > $LOG
[ -f $CSV ] || echo "seed,dataset,vanilla,TAFAS,PETSA,DAdapter,ReCycle" > $CSV

for SEED in 2025 2026 2027 2028; do
for ENTRY in "ETTh2 24" "ETTm2 96" "electricity 168" "traffic 168" "PEMS03 288" "PEMS04 288"; do
  set -- $ENTRY; DS=$1; CYC=$2
  CKPT=./checkpoints/RCFDLinear/${DS}_96_nonorm_s${SEED}/
  if [ ! -f "${CKPT}checkpoint_best.pth" ]; then
    python main.py DATA.NAME $DS DATA.PRED_LEN 96 MODEL.NAME RCFDLinear MODEL.pred_len 96 MODEL.cycle $CYC \
      TRAIN.ENABLE True TRAIN.CHECKPOINT_DIR $CKPT TTA.ENABLE False TEST.ENABLE False \
      NORM_MODULE.ENABLE False DATA_LOADER.NUM_WORKERS 0 SEED $SEED 2>&1 | tail -1 | sed "s/^/TRAIN|$DS|$SEED|/" >> $LOG
  fi
done
done
echo "TRAINING DONE $(date)" >> $LOG

bb() {  # $1=ds $2=cyc $3=ckptdir $4=seed $5=tag  rest: main.py args (env vars set at call site)
  local DS=$1 CYC=$2 CKPT=$3 SEED=$4 TAG=$5; shift 5
  python main.py DATA.NAME $DS DATA.PRED_LEN 96 MODEL.NAME RCFDLinear MODEL.pred_len 96 MODEL.cycle $CYC \
    TRAIN.ENABLE False TRAIN.CHECKPOINT_DIR $CKPT TTA.ENABLE True TEST.ENABLE False \
    RESULT_DIR ./results/RCF_$TAG/ NORM_MODULE.ENABLE False DATA_LOADER.NUM_WORKERS 0 SEED $SEED "$@" \
    2>&1 | grep "^Test MSE" | grep -oE "0\.[0-9]+" | head -1
}

for SEED in 2024 2025 2026 2027 2028; do
for ENTRY in "ETTh2 24" "ETTm2 96" "electricity 168" "traffic 168" "PEMS03 288" "PEMS04 288"; do
  set -- $ENTRY; DS=$1; CYC=$2
  if [ "$SEED" = "2024" ]; then CKPT=./checkpoints/RCFDLinear/${DS}_96_nonorm/; else CKPT=./checkpoints/RCFDLinear/${DS}_96_nonorm_s${SEED}/; fi
  V=$(python main.py DATA.NAME $DS DATA.PRED_LEN 96 MODEL.NAME RCFDLinear MODEL.pred_len 96 MODEL.cycle $CYC \
      TRAIN.ENABLE False TRAIN.CHECKPOINT_DIR $CKPT TTA.ENABLE False TEST.ENABLE True TEST.BATCH_SIZE 20000 TEST.DROP_LAST False \
      NORM_MODULE.ENABLE False DATA_LOADER.NUM_WORKERS 0 SEED $SEED 2>&1 | grep test_mse | grep -oE "0\.[0-9]+" | head -1)
  T=$(bb $DS $CYC $CKPT $SEED TAFAS TTA.SOLVER.BASE_LR 0.001 TTA.SOLVER.WEIGHT_DECAY 0.0001 TTA.TAFAS.GATING_INIT 0.01)
  P=$(bb $DS $CYC $CKPT $SEED PETSA TTA.SOLVER.BASE_LR 0.005 TTA.SOLVER.WEIGHT_DECAY 0.0001)
  DA=$(bb $DS $CYC $CKPT $SEED DAdapter)
  RC=$(RECYCLE_COMB=1 RECYCLE_POOL=1 RECYCLE_RPOOL=1 RECYCLE_COMBETA=200 bb $DS $CYC $CKPT $SEED Recycle)
  echo "$SEED,$DS,$V,$T,$P,$DA,$RC" >> $CSV
  echo "DONE s$SEED $DS | V $V | T $T | P $P | DA $DA | RC $RC" >> $LOG
done
done
echo "BB5SEED_ALL_DONE $(date)" >> $LOG
