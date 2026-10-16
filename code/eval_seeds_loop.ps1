# Evaluate seed dumps (refresh / INFUSE-U / INFUSE) as queue A and B produce them; stops when both queues are done.
$env:OMP_NUM_THREADS = '1'; $env:MKL_NUM_THREADS = '1'; $env:OPENBLAS_NUM_THREADS = '1'
Set-Location $PSScriptRoot
do {
    $running = [bool](Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" | ? { $_.CommandLine -match 'queue_[ab].ps1' })
    python -u eval_all.py res_grid.jsonl "dumps/grid/*_s[12].npz" --workers 2
    if ($running) { Start-Sleep 600 }
} while ($running)
'EVAL_SEEDS_DONE'
