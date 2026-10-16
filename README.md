# Revising In-Flight Forecasts — Anonymous Artifact

Code, data, and experiment records for the submission
*"Revising In-Flight Forecasts: Sequential Innovation Conditioning and
Cone-Coherent Fusion for Online Forecasting Services"*.

**INFUSE** is a closed-form serving framework that revises *in-flight*
multi-step forecasts — trajectories that were already emitted and are
partially verified by newly arrived observations — without model calls or
gradients. It combines (i) sequential innovation conditioning via a single
Cholesky factorization over progressive monotone-prefix covariances,
(ii) cone-coherent revision governed by an Evidence-vs.-Staleness variance
law, and (iii) co-target excess-precision fusion with posterior-calibrated
margins.

## Repository layout

```
code/         main codebase (INFUSE core + evaluation harness + results)
  inflight.py            INFUSE implementation (revision, fusion, serving)
  main.py                backbone train/eval entry (TAFAS framework)
  exp_runner.py          job-file driven training/eval runner
  dump_preds.py          dump frozen-backbone forecasts per test origin
  dump_fm.py             zero-shot Chronos-Bolt foundation-model dumps
  eval_all.py            evaluate frozen/refresh/INFUSE-U/INFUSE + fusion
  eval_revise.py         training-free revision baselines (last/mean/AR1)
  eval_causal.py         causal-gain predictions (calibration protocol)
  run_elf.py             ELF baseline (re-implemented from Algorithms 1-5)
  make_naive.py          naive last-value reference dumps
  autoscale.py           Azure Functions predictive-autoscaler simulation
  checkpoints/           trained backbone checkpoints
  results/               per-run outputs (metrics, configs)
  res_*.jsonl, *.csv     aggregated metrics reported in the paper
  *.log                  raw experiment logs
baselines/    code-only copies of compared methods (see baselines/README.md)
data/         benchmark datasets (see data/DATA.md for large/public ones)
```

## Setup

```bash
cd code
pip install -r requirements.txt
```

Datasets are read from `../data/` relative to `code/` (default `BASE_DIR`);
the bundled `data/` folder already sits in that position — keep the layout.
Prepare the zipped datasets and the Azure build first:

```bash
python ../data/prepare_data.py     # extracts data/*/  *.zip archives
```

## Reproduce

1. **Train backbones** (or use `checkpoints/`):
   `bash scripts/<model>/<dataset>_<horizon>/train.sh`
   or batch jobs: `python exp_runner.py jobs.json out.csv`
2. **Dump forecasts** (stride-1 origins over the test stream):
   `python dump_preds.py <ckpt_dir> <dataset> <model> <H> out.npz [norm]`
   Foundation-model forecasts:
   `python dump_fm.py <dataset> chronos-bolt <H> <ctx> out.npz`
3. **Evaluate in-flight revision/fusion** (CPU, strictly causal):
   `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python eval_all.py out.jsonl "dumps/grid/*.npz"`
4. **Extra analyses**: `eval_revise.py`, `eval_causal.py`, `run_elf.py`,
   `make_naive.py`, `autoscale.py`
5. **Official baselines** (TAFAS / PETSA / COSA / delta-Adapter grid):
   `python ../baselines/COSA_official/run_grid.py <out.csv> <models> <datasets> <horizons> [seed]`

All reported numbers are already computed in `code/res_*.jsonl`,
`code/results_*.csv`, and the `*.log` files; `code/results/` contains the
per-run artifacts.

## License

CC BY-NC-SA 4.0 (see `code/LICENSE`). Subfolders under `baselines/` retain
their upstream licenses and attribution.
