# Queue A (GPU): seed $args[0] for iTransformer / DLinear on all datasets at H = 192, 336, 720 (official adapters;
# delta-Adapter for the seeds runs on the CPU, queue_dad_seeds.ps1). Run one instance per seed, each with its own CSV.
$ErrorActionPreference = 'Continue'
Set-Location $PSScriptRoot
$DS = 'ETTh1,ETTh2,ETTm1,ETTm2,exchange_rate,weather'
$s = $args[0]
$csv = if ($s -eq '1') { 'grid_seeds.csv' } else { "grid_seeds_$s.csv" }     # seed 1 resumes the earlier CSV
python -u run_grid.py $csv iTransformer,DLinear $DS 192,336,720 $s nodad
'QUEUE_A_DONE'
