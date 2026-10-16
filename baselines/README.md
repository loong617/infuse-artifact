# Baselines

Code-only copies of the compared open-source methods (training checkpoints,
bundled datasets and generated outputs were stripped; see `data/DATA.md`
for datasets and `code/res_*.jsonl` for reported metrics). Upstream
sources:

- `TAFAS/` — TAFAS (github.com/kimanki/TAFAS)
- `COSA_official/` — COSA, on the Time-Series-Library / TAFAS harness
  (github.com/thuml/Time-Series-Library); `run_grid.py` is the benchmark
  grid runner used for the paper
- `AdapterOfficial/` — calibration-adapter family (AdaCali, AdaIntpX,
  Adapter-X+Y, QuanCali, ConfCali)
- `ADAPTZ_official/` — AdaptZ / DSoF (github.com/yyalau/iclr2025_dsof)
- `CycleNet/` — CycleNet (github.com/ACAT-SCUT/CycleNet)
- `FSNet_official/`, `OneNet_official/`, `SOLID_official/` — continual/online
  forecasting baselines on the OneNet harness (github.com/yfzhang114/OneNet)
- `OnlineTSF_official/` — OnlineTSF suite (github.com/SJTU-DMTai/OnlineTSF)

Each subfolder keeps its upstream license and file headers.
