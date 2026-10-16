#!/bin/bash
# FFT数据源修复(val尾部替代测试流前缀)后的复现验证
# 已知: 6数据集两种窗口的FFT周期完全一致 -> 数字应逐位复现已发布值
# 本脚本: ReCycle(η=200冻结配置) seed2024 全6数据集, 与 results_sota.csv 对账
# 已手动验证: ETTh2 .2710/33次 ✓, COSA .2901/35次 ✓
cd "$(dirname "$0")" || exit 1
export NONORM=1

declare -A EXP=( [ETTh2]=0.2710 [ETTm2]=0.1702 [electricity]=0.2221 [traffic]=0.7998 [PEMS03]=0.4503 [PEMS04]=0.4764 )
for ENTRY in "ETTh2 24" "ETTm2 96" "electricity 168" "traffic 168" "PEMS03 288" "PEMS04 288"; do
  DS=$(echo $ENTRY | cut -d' ' -f1); CY=$(echo $ENTRY | cut -d' ' -f2)
  CKPT=./checkpoints/CycleNet/${DS}_96_nonorm_s2024/
  COMMON="DATA.NAME $DS DATA.PRED_LEN 96 MODEL.NAME CycleNet MODEL.pred_len 96 MODEL.cycle $CY TRAIN.CHECKPOINT_DIR $CKPT NORM_MODULE.ENABLE False DATA_LOADER.NUM_WORKERS 0 TRAIN.ENABLE False SEED 2024"
  OUT=$(RECYCLE_COMB=1 RECYCLE_POOL=1 RECYCLE_RPOOL=1 RECYCLE_COMBETA=200 python main.py $COMMON TTA.ENABLE True TEST.ENABLE False RESULT_DIR ./results/Recycle_fftfix/ 2>&1 | grep -E "^Test MSE|adaptations")
  GOT=$(echo "$OUT" | grep -oE "0\.[0-9]+" | head -1)
  echo "VERIFY $DS expected=${EXP[$DS]} got=$GOT | $(echo $OUT | tr '\n' ' ')"
done
echo VERIFY_DONE
