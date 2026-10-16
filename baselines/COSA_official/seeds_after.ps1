# After the iTransformer/DLinear grid is complete, rerun the H=96 settings with two more seeds (variance check).
while ((Get-Content grid_itdl.csv | Measure-Object).Count -lt 192) { Start-Sleep -Seconds 120 }
foreach ($s in 1, 2) {
    python -u run_grid.py grid_seeds.csv iTransformer,DLinear ETTh1,ETTh2,ETTm1,ETTm2,exchange_rate,weather 96 $s 2>&1 | Out-File -Append -Encoding utf8 grid_seeds.log
}
"SEEDS_DONE" | Out-File -Append -Encoding utf8 grid_seeds.log
