# Compact progress summary of the remaining experiments.
Set-Location $PSScriptRoot
function rows($pat, $m) { (Get-ChildItem $pat -ErrorAction SilentlyContinue | % { Get-Content $_ } | ? { $_ -match $m }).Count }
"{0:HH:mm}  dAdapter seed0: {1}/84   seeds(off.) s1: {2}/36  s2: {3}/36  PatchTST seeds: {4}/12   dAdapter seeds: {5}/108   Azure PatchTST: {6}/4" -f (Get-Date),
    (rows 'grid_dad*.csv' ',DADAPTER,'),
    (rows 'grid_seeds.csv' '^(iTransformer|DLinear),[^,]+,(192|336|720),1,frozen'),
    (rows 'grid_seeds_2.csv' '^(iTransformer|DLinear),[^,]+,(192|336|720),2,frozen'),
    (rows 'grid_seeds.csv' '^PatchTST,[^,]+,96,[12],frozen'),
    ((rows 'grid_seeds_dad*.csv' ',DADAPTER,') + (rows 'grid_seeds.csv' '^PatchTST,.*,DADAPTER,')),
    (rows 'grid_azure_patch.csv' ',frozen,')
