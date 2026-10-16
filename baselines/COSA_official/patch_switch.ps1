# Wait until the main PatchTST runner has finished weather-96, then stop it and continue PatchTST at H=192 only
# (H=336/720 for the large datasets run in grid_patch_long; the small datasets ran in grid_patch_small).
while (-not (Select-String -Path grid_patch.csv -Pattern '^PatchTST,weather,96,0,COSA' -Quiet)) { Start-Sleep -Seconds 60 }
$runner = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'run_grid.py grid_patch.csv' }
foreach ($r in $runner) {
    Get-CimInstance Win32_Process -Filter "ParentProcessId=$($r.ProcessId)" | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
    Stop-Process -Id $r.ProcessId -Force
}
"switched at $(Get-Date)" | Out-File -Append patch_switch.log
python -u run_grid.py grid_patch192.csv PatchTST ETTm1,ETTm2,weather 192 0 2>&1 | Out-File -Append -Encoding utf8 grid_patch192.log
"PATCH192_DONE" | Out-File -Append patch_switch.log
