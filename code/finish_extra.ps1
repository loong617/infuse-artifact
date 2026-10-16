# After finish_all.ps1: extra analyses (Figure 7b/c) for the dumps that appeared late, then assets and PDF again.
$env:OMP_NUM_THREADS = '1'; $env:MKL_NUM_THREADS = '1'; $env:OPENBLAS_NUM_THREADS = '1'
Set-Location $PSScriptRoot
while (Get-CimInstance Win32_Process | ? { $_.ProcessId -ne $PID -and $_.CommandLine -match 'finish_all\.ps1|analysis_extra\.py' }) { Start-Sleep 300 }
python -u analysis_extra.py "dumps/grid/*_96.npz" "dumps/fm/*_96.npz" --gain-k --workers 8
python -u analysis_extra.py "dumps/grid/*.npz" "dumps/fm/*.npz" --workers 8
python -u make_paper_assets.py
Set-Location ..\paper
latexmk -pdf -interaction=nonstopmode main_www.tex | Out-Null
pdflatex -interaction=nonstopmode main_www.tex | Out-Null
"FINISH_EXTRA_DONE $(Get-Date)"
