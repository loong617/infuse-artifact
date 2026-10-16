# CPU queue: delta-Adapter for the iTransformer / DLinear seed runs (seeds 1, 2, all horizons). Run with
# CUDA_VISIBLE_DEVICES=-1; settings whose checkpoint or dump does not exist yet are skipped, so the loop repeats
# until queue A has finished all seed trainings.
$ErrorActionPreference = 'Continue'
Set-Location $PSScriptRoot
$DS = 'ETTh1,ETTh2,ETTm1,ETTm2,exchange_rate,weather'
do {
    $running = [bool](Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" | ? { $_.CommandLine -match 'queue_a.ps1' })
    foreach ($s in 1, 2) {
        python -u run_grid.py "grid_seeds_dad_$($args[0]).csv" $args[0] $DS 96,192,336,720 $s only:DADAPTER
    }
    if ($running) { Start-Sleep 300 }
} while ($running)
'QUEUE_DAD_SEEDS_DONE'
