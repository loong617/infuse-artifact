"""Naive (last observed value) forecasts in the dump format, for every benchmark dataset and horizon (and Azure
Functions): pred[i, j] = y[L + i - 1]. Written to dumps/naive/naive_<ds>_<H>.npz; evaluate with eval_all.py."""
import os

import numpy as np

DSETS = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'exchange_rate', 'weather', 'azure_functions']
os.makedirs('dumps/naive', exist_ok=True)
for ds in DSETS:
    for H in (96, 192, 336, 720):
        out = 'dumps/naive/naive_%s_%d.npz' % (ds, H)
        src = 'dumps/grid/iTransformer_%s_%d.npz' % (ds, H)
        if os.path.exists(out) or not os.path.exists(src):
            continue
        D = np.load(src)
        y, L, n = D['y'], int(D['L']), D['pred'].shape[0]
        pred = np.repeat(y[L - 1:L - 1 + n][:, None, :], H, axis=1).astype(np.float32)
        np.savez(out, y=y, pred=pred, L=L, H=H)
        print(out, pred.shape)
