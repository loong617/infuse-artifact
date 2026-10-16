# After every experiment queue has finished: evaluate all new dumps, rebuild tables / figures / numbers, compile.
$env:OMP_NUM_THREADS = '1'; $env:MKL_NUM_THREADS = '1'; $env:OPENBLAS_NUM_THREADS = '1'
Set-Location $PSScriptRoot
$busy = 'queue_[ab]\.ps1|queue_dad_|run_grid\.py|dump_fm\.py'
while (Get-CimInstance Win32_Process | ? { $_.ProcessId -ne $PID -and $_.CommandLine -match $busy }) { Start-Sleep 300 }
"experiments finished $(Get-Date)"
python -u eval_all.py res_grid.jsonl "dumps/grid/*_s[12].npz" --workers 8
python -u eval_all.py res_grid.jsonl "dumps/grid/*.npz" --service --ablate --workers 8
python -u eval_all.py res_fm.jsonl "dumps/fm/*.npz" --service --ablate --workers 8
python -u eval_all.py res_tta.jsonl "dumps/grid/tta/*.npz" --workers 8
python -u eval_revise.py res_revise.jsonl "dumps/grid/*.npz" "dumps/fm/*.npz" --workers 8
python -u make_paper_assets.py
Set-Location ..\paper
Push-Location figs; pdflatex -interaction=nonstopmode framework.tex | Out-Null; Pop-Location
latexmk -pdf -interaction=nonstopmode main_www.tex | Out-Null
pdflatex -interaction=nonstopmode main_www.tex | Out-Null
"FINISH_ALL_DONE $(Get-Date)"
