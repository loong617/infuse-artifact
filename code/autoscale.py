"""Downstream autoscaling on Azure Functions (H = 96). At every minute t, a predictive autoscaler provisions the
capacity of each application for minute t + d (provisioning delay d) from the currently served forecast and its
closed-form posterior error standard deviation scale s_post:
    capacity = max(0, mu + sigma * (yhat_z + h * s_post))
Sweeping h traces the trade-off between the SLA-violation rate and over-provisioning on raw invocation counts.
Results to autoscale.json.
"""
import json
import os

import numpy as np
import pandas as pd

import inflight as I

H, HS = 96, 48
DELAYS = (1, 5, 10, 15)                 # minutes ahead
R_LIST = (1, 12, 24)
MARGINS = np.linspace(-1.0, 6.0, 141)
TARGETS = (0.01, 0.05)
FC = [('iTransformer', 'dumps/grid'), ('PatchTST', 'dumps/grid'), ('DLinear', 'dumps/grid'),
      ('bolt-small', 'dumps/fm'), ('bolt-base', 'dumps/fm')]


def raw_stats():
    X = pd.read_csv('../data/azure_functions/azure_functions.csv').iloc[:, 1:].values.astype(np.float64)
    S = np.load('dumps/series_azure_functions.npz')
    n_tr = int(S['n_train'])
    mu, sd = X[:n_tr].mean(0), X[:n_tr].std(0)
    ts = int(S['test_start'])
    z = (X[ts:ts + 50] - mu) / sd
    assert np.abs(z - S['x'][ts:ts + 50]).max() < 1e-3, 'standardisation mismatch'
    return mu, sd


def curve(e_served, gt_z, mu, sd, d, scale=None):
    """e_served, gt_z: [n, HS, C] served errors and standardised targets (row t, column j -> minute t + j + 1)."""
    e = e_served[:, d - 1]                                   # [n, C]: lead d - 1 -> minute t + d
    y = gt_z[:, d - 1]
    yhat = y - e
    sc = 1.0 if scale is None else scale[:, d - 1]
    D = np.maximum(mu + sd * y, 0.0)
    viol, over = [], []
    for h in MARGINS:
        cap = np.maximum(mu + sd * (yhat + h * sc), 0.0)
        viol.append(float((D > cap + 1e-9).mean()))
        over.append(float(np.maximum(cap - D, 0).sum() / D.sum()))
    return viol, over


def at_target(viol, over, target):
    v, o = np.array(viol), np.array(over)
    if v.min() > target or v.max() < target:
        return float('nan')
    order = np.argsort(v)
    return float(np.interp(target, v[order], o[order]))


def main():
    mu, sd = raw_stats()
    res = {}
    for name, base in FC:
        path = '%s/%s_azure_functions_%d.npz' % (base, name, H)
        y, pred, gt, E, L, _ = I.load(path)
        variants = {}
        scales = {}
        for r in R_LIST:
            for mode in (('hold',) if r == 1 else ('hold', 'update', 'fuse')):
                _, _, es, sc = I.service(E, pred, gt, H, r, HS, mode=mode, return_served=True, return_std=True)
                variants['%s_r%d' % (mode, r)] = es
                if mode in ('update', 'fuse'):
                    scales['%s_r%d' % (mode, r)] = sc
                if mode == 'fuse' and r == 12:
                    variants['fuse_pt_r12'] = es            # point-only (uncalibrated margin) for ablation
        _, _, es1, sc1 = I.service(E, pred, gt, H, 1, HS, mode='fuse', return_served=True, return_std=True)
        variants['fuse_r1'] = es1; scales['fuse_r1'] = sc1
        extra = [('ELF', '%s/elf/ELF_%s_azure_functions_%d.npz' % (base, name, H))]
        if base == 'dumps/grid':
            extra += [(t, 'dumps/grid/tta/%s_%s_azure_functions_%d.npz' % (t, name, H))
                      for t in ('DADAPTER', 'TAFAS-noadj', 'PETSA-noadj', 'COSA-noadj')]
        for tag, p in extra:
            if os.path.exists(p):
                _, pr2, gt2, E2, _, _ = I.load(p)
                variants['%s_r1' % tag] = I.service(E2, pr2, gt2, H, 1, HS, mode='hold', return_served=True)[2]
                if tag == 'ELF':
                    for r in R_LIST[1:]:
                        variants['ELFhold_r%d' % r] = I.service(E2, pr2, gt2, H, r, HS, mode='hold', return_served=True)[2]
                        _, _, es_e, sc_e = I.service(E2, pr2, gt2, H, r, HS, mode='fuse', return_served=True, return_std=True)
                        variants['ELFfuse_r%d' % r] = es_e; scales['ELFfuse_r%d' % r] = sc_e
                    _, _, es_e1, sc_e1 = I.service(E2, pr2, gt2, H, 1, HS, mode='fuse', return_served=True, return_std=True)
                    variants['ELFfuse_r1'] = es_e1; scales['ELFfuse_r1'] = sc_e1
        n = E.shape[0]
        idx = np.arange(n)[:, None] + np.arange(HS)[None, :] + L
        gt_z = y[idx]
        out = {}
        for vname, es in variants.items():
            out[vname] = {}
            sc = scales.get(vname)
            for d in DELAYS:
                v, o = curve(es, gt_z, mu, sd, d, scale=sc)
                out[vname][str(d)] = {'viol': v, 'over': o, 'mse': float((es[:, d - 1] ** 2).mean()),
                                      **{'over_at_%g' % tg: at_target(v, o, tg) for tg in TARGETS}}
        res[name] = out
        print(name, {k: round(v['10']['over_at_0.05'], 4) for k, v in out.items()}, flush=True)
    json.dump({'margins': MARGINS.tolist(), 'delays': DELAYS, 'res': res}, open('autoscale.json', 'w'))


if __name__ == '__main__':
    main()
