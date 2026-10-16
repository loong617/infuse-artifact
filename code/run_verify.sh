#!/bin/bash
# 验证性重跑:消融 + 附录数据集 + 计时,全部 test-only NONORM 漂移域
cd "$(dirname "$0")" || exit 1
export NONORM=1

run() {  # $1=ds $2=cyc $3=ckpt $4=tag
  python main.py DATA.NAME $1 DATA.PRED_LEN 96 MODEL.NAME CycleNet MODEL.pred_len 96 MODEL.cycle $2 \
    TRAIN.ENABLE False TRAIN.CHECKPOINT_DIR ./checkpoints/CycleNet/$3/ TTA.ENABLE True TEST.ENABLE False \
    RESULT_DIR ./results/Recycle_nonorm/ NORM_MODULE.ENABLE False DATA_LOADER.NUM_WORKERS 0 SEED 2024 \
    2>&1 | grep -E "Test MSE|adaptations|update time" | sed "s/^/ABL|$4|$1|/"
}

echo "===== ABLATION ETTh2 s2024 ====="
unset RECYCLE_COMB RECYCLE_POOL RECYCLE_RPOOL RECYCLE_COMBETA
run ETTh2 24 ETTh2_96_nonorm_s2024 ridge
RECYCLE_COMB=1 run ETTh2 24 ETTh2_96_nonorm_s2024 comb
RECYCLE_COMB=1 RECYCLE_POOL=1 run ETTh2 24 ETTh2_96_nonorm_s2024 pool
RECYCLE_COMB=1 RECYCLE_POOL=1 RECYCLE_RPOOL=1 run ETTh2 24 ETTh2_96_nonorm_s2024 rpool
RECYCLE_COMB=1 RECYCLE_POOL=1 RECYCLE_RPOOL=1 RECYCLE_COMBETA=200 run ETTh2 24 ETTh2_96_nonorm_s2024 final

echo "===== ABLATION electricity s2024 ====="
unset RECYCLE_COMB RECYCLE_POOL RECYCLE_RPOOL RECYCLE_COMBETA
run electricity 168 electricity_96_nonorm_s2024 ridge
RECYCLE_COMB=1 run electricity 168 electricity_96_nonorm_s2024 comb
RECYCLE_COMB=1 RECYCLE_POOL=1 run electricity 168 electricity_96_nonorm_s2024 pool
RECYCLE_COMB=1 RECYCLE_POOL=1 RECYCLE_RPOOL=1 run electricity 168 electricity_96_nonorm_s2024 rpool
RECYCLE_COMB=1 RECYCLE_POOL=1 RECYCLE_RPOOL=1 RECYCLE_COMBETA=200 run electricity 168 electricity_96_nonorm_s2024 final

echo "===== APPENDIX: final frozen config ====="
FINAL="RECYCLE_COMB=1 RECYCLE_POOL=1 RECYCLE_RPOOL=1 RECYCLE_COMBETA=200"
for SEED in 2024 2025 2026 2027 2028; do
  env $FINAL python main.py DATA.NAME ETTh1 DATA.PRED_LEN 96 MODEL.NAME CycleNet MODEL.pred_len 96 MODEL.cycle 24 \
    TRAIN.ENABLE False TRAIN.CHECKPOINT_DIR ./checkpoints/CycleNet/ETTh1_96_nonorm_s${SEED}/ TTA.ENABLE True TEST.ENABLE False \
    RESULT_DIR ./results/Recycle_nonorm/ NORM_MODULE.ENABLE False DATA_LOADER.NUM_WORKERS 0 SEED $SEED \
    2>&1 | grep -E "Test MSE" | sed "s/^/APP|ETTh1|s${SEED}|/"
done
env $FINAL python main.py DATA.NAME weather DATA.PRED_LEN 96 MODEL.NAME CycleNet MODEL.pred_len 96 MODEL.cycle 144 \
  TRAIN.ENABLE False TRAIN.CHECKPOINT_DIR ./checkpoints/CycleNet/weather_96_nonorm/ TTA.ENABLE True TEST.ENABLE False \
  RESULT_DIR ./results/Recycle_nonorm/ NORM_MODULE.ENABLE False DATA_LOADER.NUM_WORKERS 0 SEED 2024 \
  2>&1 | grep -E "Test MSE" | sed "s/^/APP|weather|single|/"
env $FINAL python main.py DATA.NAME PEMS07 DATA.PRED_LEN 96 MODEL.NAME CycleNet MODEL.pred_len 96 MODEL.cycle 288 \
  TRAIN.ENABLE False TRAIN.CHECKPOINT_DIR ./checkpoints/CycleNet/PEMS07_96_nonorm/ TTA.ENABLE True TEST.ENABLE False \
  RESULT_DIR ./results/Recycle_nonorm/ NORM_MODULE.ENABLE False DATA_LOADER.NUM_WORKERS 0 SEED 2024 \
  2>&1 | grep -E "Test MSE" | sed "s/^/APP|PEMS07|single|/"
env $FINAL python main.py DATA.NAME PEMS08 DATA.PRED_LEN 96 MODEL.NAME CycleNet MODEL.pred_len 96 MODEL.cycle 288 \
  TRAIN.ENABLE False TRAIN.CHECKPOINT_DIR ./checkpoints/CycleNet/PEMS08_96_nonorm/ TTA.ENABLE True TEST.ENABLE False \
  RESULT_DIR ./results/Recycle_nonorm/ NORM_MODULE.ENABLE False DATA_LOADER.NUM_WORKERS 0 SEED 2024 \
  2>&1 | grep -E "Test MSE" | sed "s/^/APP|PEMS08|single|/"

echo "===== TIMING (fresh, same conditions) ====="
timing() {  # $1=ds $2=cyc $3=ckpt $4=method tag from RESULT_DIR
  python main.py DATA.NAME $1 DATA.PRED_LEN 96 MODEL.NAME CycleNet MODEL.pred_len 96 MODEL.cycle $2 \
    TRAIN.ENABLE False TRAIN.CHECKPOINT_DIR ./checkpoints/CycleNet/$3/ TTA.ENABLE True TEST.ENABLE False \
    RESULT_DIR ./results/$4 NORM_MODULE.ENABLE False DATA_LOADER.NUM_WORKERS 0 SEED 2024 \
    $5 2>&1 | grep -E "Test MSE|adaptations|update time" | sed "s/^/TIME|$1|$4|/"
}
timing ETTh2 24 ETTh2_96_nonorm_s2024 TAFAS_nonorm/ "TTA.SOLVER.BASE_LR 0.001 TTA.SOLVER.WEIGHT_DECAY 0.0001 TTA.TAFAS.GATING_INIT 0.01"
timing ETTh2 24 ETTh2_96_nonorm_s2024 PETSA_nonorm/ "TTA.SOLVER.BASE_LR 0.005 TTA.SOLVER.WEIGHT_DECAY 0.0001"
timing ETTh2 24 ETTh2_96_nonorm_s2024 COSA_nonorm/ ""
RECYCLE_COMB=1 RECYCLE_POOL=1 RECYCLE_RPOOL=1 RECYCLE_COMBETA=200 timing ETTh2 24 ETTh2_96_nonorm_s2024 Recycle_nonorm/ ""
timing electricity 168 electricity_96_nonorm_s2024 TAFAS_nonorm/ "TTA.SOLVER.BASE_LR 0.001 TTA.SOLVER.WEIGHT_DECAY 0.0001 TTA.TAFAS.GATING_INIT 0.01"
timing electricity 168 electricity_96_nonorm_s2024 PETSA_nonorm/ "TTA.SOLVER.BASE_LR 0.005 TTA.SOLVER.WEIGHT_DECAY 0.0001"
timing electricity 168 electricity_96_nonorm_s2024 COSA_nonorm/ ""
RECYCLE_COMB=1 RECYCLE_POOL=1 RECYCLE_RPOOL=1 RECYCLE_COMBETA=200 timing electricity 168 electricity_96_nonorm_s2024 Recycle_nonorm/ ""

echo "===== ALL DONE ====="
