# Re-run the adaptation-only pass every 15 minutes until all 72 settings have it (it skips finished ones and
# settings whose frozen dump does not exist yet).
while ($true) {
    python -u run_grid.py grid_noadj.csv iTransformer,DLinear,PatchTST ETTh1,ETTh2,ETTm1,ETTm2,exchange_rate,weather 96,192,336,720 0 noadj 2>&1 | Out-File -Append -Encoding utf8 grid_noadj.log
    $n = (Get-Content grid_noadj.csv | Measure-Object).Count
    if ($n -ge 216) { break }
    Start-Sleep -Seconds 900
}
"NOADJ_ALL_DONE" | Out-File -Append -Encoding utf8 grid_noadj.log
