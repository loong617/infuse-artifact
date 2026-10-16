# Copyright (c) 2025-present, Royal Bank of Canada.
# Copyright (c) 2025-present, Kim et al.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

##########################################################################################
# Code is originally from the TAFAS (https://arxiv.org/pdf/2501.04970.pdf) implementation
# from https://github.com/kimanki/TAFAS by Kim et al. which is licensed under 
# Modified MIT License (Non-Commercial with Permission).
# You may obtain a copy of the License at
#
#    https://github.com/kimanki/TAFAS/blob/master/LICENSE
#
###########################################################################################

#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=30G

echo $CUDA_VISIBLE_DEVICES

export CUDA_VISIBLE_DEVICES=0

DATASET="ETTm1"
PRED_LEN=192
MODEL="iTransformer"
CHECKPOINT_DIR="./checkpoints/${MODEL}/${DATASET}_${PRED_LEN}/"
OUTPUT="${MODEL}_${DATASET}_${PRED_LEN}.txt"
BASE_LR=0.001
WEIGHT_DECAY=0.0
GATING_INIT=0.01

python main.py DATA.NAME ${DATASET} DATA.PRED_LEN ${PRED_LEN} MODEL.NAME ${MODEL} MODEL.pred_len ${PRED_LEN} TRAIN.ENABLE True TRAIN.CHECKPOINT_DIR ${CHECKPOINT_DIR} TTA.ENABLE True TTA.SOLVER.BASE_LR ${BASE_LR} TTA.SOLVER.WEIGHT_DECAY ${WEIGHT_DECAY} TTA.TAFAS.GATING_INIT ${GATING_INIT}