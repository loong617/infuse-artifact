# Datasets

Expected layout: `data/<name>/<name>.{csv,npz,txt}`, read by
`code/datasets/` with `BASE_DIR = ../data/` (relative to `code/`).

## Bundled

| Dataset | File | Note |
|---|---|---|
| ETTh1, ETTh2 | `ETT*/ETT*.csv` | raw CSV |
| ETTm1, ETTm2 | `ETT*/ETT*.csv` | raw CSV |
| weather | `weather/weather.csv` | raw CSV |
| exchange_rate | `exchange_rate/exchange_rate.csv` | raw CSV |
| electricity | `electricity/electricity.csv` | raw CSV |
| PEMS03/04/07/08 | `PEMS*/PEMS*.npz` | raw NPZ |
| traffic | `traffic/traffic.zip` | zipped (extract with `prepare_data.py`) |
| solar | `solar/solar_AL.zip` | zipped (extract with `prepare_data.py`) |
| Azure Functions | `azure_functions/azure_functions.csv` | prebuilt series |

Run `python data/prepare_data.py` to extract the zipped archives in place.

## Requires download (too large to bundle)

- **Azure Functions raw trace** (`azure_raw/`): the 2019 Azure Functions
  invocation trace (public release). Place the per-day
  `invocations_per_function_md.anon.d*.csv` (and the
  `*_durations_*/` `*_memory_*` percentiles files if needed) into
  `data/azure_raw/`, then run `python data/azure_raw/build_azure.py`
  to regenerate `data/azure_functions/azure_functions.csv`.
  The bundled `azure_functions/azure_functions.csv` is already the
  built artifact, so this step is only needed for a full rebuild.

The public benchmark datasets above are also available from the
Time-Series-Library collection and the original dataset releases; the
bundled copies match those sources unchanged.
