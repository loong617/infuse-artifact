# [PUT ICML 2025] Accurate Parameter-Efficient Test-Time Adaptation for Time Series Forecasting

## Abstract
Real-world time series often exhibit a non-stationary nature, degrading the performance of
pre-trained forecasting models. Test-Time Adaptation (TTA) addresses this by adjusting models
during inference, but existing methods typically update the full model, increasing memory and
compute costs. We propose PETSA, a parameter-efficient method that adapts forecasters at test
time by only updating small calibration modules on the input and output. PETSA uses low-rank
adapters and dynamic gating to adjust representations without retraining. To maintain accuracy despite limited adaptation capacity, we introduce a specialized loss combining three components: (1) a robust term, (2) a frequency-domain term to preserve periodicity, and (3) a patch-wise structural term for structural alignment. PETSA improves the adaptability of various forecasting backbones while requiring fewer parameters than baselines. Experimental results on benchmark datasets show that PETSA achieves competitive or better performance across all horizons.

![Overview](./overview.png)

The figure above illustrates the overview pipeline of our proposed PETSA.

## Prepare Datasets
The datasets can be downloaded from the [Time-Series-Library](https://github.com/thuml/Time-Series-Library).

### Datasets
- ETTh1
- ETTm1
- ETTh2
- ETTm2
- exchange_rate
- weather

Place the downloaded datasets in `data/{dataset}/{dataset}.csv`.

**Example:** For ETTh1, place it in `data/ETTh1/ETTh1.csv`.


### Models
- iTransformer
- PatchTST
- DLinear
- OLS
- FreTS
- MICN

### Datasets
- ETTh1
- ETTm1
- ETTh2
- ETTm2
- exchange_rate
- weather

## Libraries Version


```bash
einops==0.8.0
local-attention==1.9.14
matplotlib==3.7.0
numpy==1.23.5
pandas==1.5.3
reformer-pytorch==1.4.4
scikit-learn==1.2.2
scipy==1.10.1
sympy==1.11.1
torch==1.7.1
tqdm==4.64.1
```


## Install Requirements

```bash
pip install -r requirements.txt
```


## Example Execution Code
The script files are located in the `scripts/` directory. Run the following script to execute the model:

```bash
bash scripts/{model}/{dataset}_{pred_len}/run_petsa.sh
```

**Example:** For model `iTransformer` and dataset `ETTh1`, run:

```bash
# For example: (gpu_number low_rank loss_alpha gating_init)
bash scripts/iTransformer/ETTh1_96/run_petsa.sh  3 16 0.2 0.02
```

For settings where checkpoints are not provided, you can train the model with: 

```bash
bash scripts/iTransformer/ETTh1_96/train.sh
```

## Models and Layers Packages
The implementation references the [Time-Series-Library](https://github.com/thuml/Time-Series-Library).



## Acknowledgements

Thanks for the [Time-Series-Library](https://github.com/thuml/Time-Series-Library) and [TAFAS](https://github.com/kimanki/TAFAS) libraries.