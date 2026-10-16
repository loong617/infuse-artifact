#!/bin/bash
#
# COSA with the 2-layer MLP output adapter (ADAPTER_TYPE=MLP).
# Stack: Linear(pred_len + buffer_context_size -> HIDDEN_DIM) -> Tanh -> Dropout(0.1) -> Linear(-> pred_len)
# The Linear counterpart is scripts/cosa.sh.

export CUDA_VISIBLE_DEVICES=0

MODELS=("DLinear" "FreTS" "iTransformer" "MICN" "OLS" "PatchTST")
DATASETS=("ETTh1" "ETTh2" "ETTm1" "ETTm2" "exchange_rate" "weather")
PRED_LENS=(96 192 336 720)

# COSA Adapter Basic Configuration
BUFFER_CONTEXT_SIZE=10
STEPS=3
BATCH_SIZE=48

# Adapter Architecture (the point of this script)
ADAPTER_TYPE=MLP
HIDDEN_DIM=64

# PAAS Configuration
PAAS=True
PERIOD_N=1

# Fast Adaptation Optimization Settings
FAST_ADAPTATION=True
ADAPTIVE_LR=True
PER_BATCH_LR_RESET=True
MAX_LR=0.005
MIN_LR=0.0001

POGT=True
PARTIAL_ADAPT=True
ADJUST_PRED=True

for MODEL in "${MODELS[@]}"; do
  for DATASET in "${DATASETS[@]}"; do
    for PRED_LEN in "${PRED_LENS[@]}"; do
      CHECKPOINT_DIR="./checkpoints/${MODEL}/${DATASET}_${PRED_LEN}/"
      RESULT_DIR="./results/COSA_MLP/"

      OUTPUT="./results/summary/COSA_MLP/${MODEL}/${DATASET}/${PRED_LEN}.txt"
      mkdir -p "$(dirname "${OUTPUT}")"

      ../.venv/bin/python main.py \
        DATA.BASE_DIR ../data/ \
        DATA.NAME ${DATASET} \
        DATA.PRED_LEN ${PRED_LEN} \
        MODEL.NAME ${MODEL} \
        MODEL.pred_len ${PRED_LEN} \
        TRAIN.ENABLE False \
        TRAIN.CHECKPOINT_DIR ${CHECKPOINT_DIR} \
        TTA.ENABLE True \
        TTA.SOLVER.BASE_LR 0.001 \
        TTA.SOLVER.WEIGHT_DECAY 0.0001 \
        TTA.COSA.BATCH_SIZE ${BATCH_SIZE} \
        TTA.COSA.STEPS ${STEPS} \
        TTA.COSA.BUFFER_CONTEXT_SIZE ${BUFFER_CONTEXT_SIZE} \
        TTA.COSA.ADAPTER_TYPE ${ADAPTER_TYPE} \
        TTA.COSA.HIDDEN_DIM ${HIDDEN_DIM} \
        TTA.COSA.FAST_ADAPTATION ${FAST_ADAPTATION} \
        TTA.COSA.PER_BATCH_LR_RESET ${PER_BATCH_LR_RESET} \
        TTA.COSA.ADAPTIVE_LR ${ADAPTIVE_LR} \
        TTA.COSA.PAAS ${PAAS} \
        TTA.COSA.PERIOD_N ${PERIOD_N} \
        TTA.COSA.POGT ${POGT} \
        TTA.COSA.PARTIAL_ADAPT ${PARTIAL_ADAPT} \
        TTA.COSA.ADJUST_PRED ${ADJUST_PRED} \
        RESULT_DIR ${RESULT_DIR} > ${OUTPUT}
    done
  done
done
