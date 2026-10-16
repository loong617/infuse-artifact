# Queue B (GPU): delta-Adapter for PatchTST (benchmark and Azure Functions), then PatchTST seeds 1, 2 at H = 96
# (official adapters + delta-Adapter, then adaptation-only runs). iTransformer / DLinear delta-Adapter runs on the
# CPU (grid_dad_w*.csv), because many small sequential steps are slow on a shared GPU.
# Starts when the Chronos-Bolt dumps are finished, to keep GPU memory below its capacity.
$ErrorActionPreference = 'Continue'
Set-Location $PSScriptRoot
while (Get-CimInstance Win32_Process -Filter "Name='python.exe'" | ? { $_.CommandLine -match 'dump_fm.py' }) { Start-Sleep 60 }
$DS = 'ETTh1,ETTh2,ETTm1,ETTm2,exchange_rate,weather'
python -u run_grid.py grid_dad_patch.csv PatchTST $DS 96,192,336,720 0 only:DADAPTER
foreach ($s in 1, 2) {
    python -u run_grid.py grid_seeds.csv PatchTST $DS 96 $s nodad
    python -u run_grid.py grid_seeds.csv PatchTST $DS 96 $s only:DADAPTER
    python -u run_grid.py grid_seeds_noadj.csv PatchTST $DS 96 $s noadj
}
# Azure Functions: PatchTST checkpoints appear only after its (long) training
while (Get-CimInstance Win32_Process -Filter "Name='python.exe'" | ? { $_.CommandLine -match 'grid_azure(_noadj)?(_patch)?\.csv PatchTST' }) { Start-Sleep 120 }
python -u run_grid.py grid_dad_patch.csv PatchTST azure_functions 96,192,336,720 0 only:DADAPTER
'QUEUE_B_DONE'
