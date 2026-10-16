# delta-Adapter for one backbone and one seed on all datasets and horizons, after the processes matching the wait
# pattern have exited. Usage: queue_dad_one.ps1 <model> <seed> <wait-regex>   (device from CUDA_VISIBLE_DEVICES)
$ErrorActionPreference = 'Continue'
Set-Location $PSScriptRoot
$m, $s, $wait = $args[0], $args[1], $args[2]
while (Get-CimInstance Win32_Process | ? { $_.ProcessId -ne $PID -and $_.CommandLine -match $wait }) { Start-Sleep 120 }
$DS = 'ETTh1,ETTh2,ETTm1,ETTm2,exchange_rate,weather'
python -u run_grid.py "grid_seeds_dad_${m}_s$s.csv" $m $DS 96,192,336,720 $s only:DADAPTER
"QUEUE_DAD_ONE_DONE $m $s"
