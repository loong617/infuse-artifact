"""Causal check of the two gain predictions (Prop. 4.1 for refresh, Prop. 5.1 for INFUSE-U) and of the stability of the
error correlation. The first part of the test stream acts as a calibration period (the first days of deployment):
the lead-wise MSE m(j) and the pooled error covariance are estimated only from forecasts fully revealed by its end,
and the gains are predicted for, and measured on, the remaining forecasts.

calibration end n_cal = max(round(FRAC * n), H + MIN_WIN); settings in which the evaluation part would be shorter than
half of the stream are skipped (short series with long horizons).

Per dump (results to res_causal.jsonl):
  pred_ref / act_ref   predicted / measured relative MSE change of refresh on the evaluation part
  pred_upd / act_upd   predicted (calibration covariance) / measured (online INFUSE-U) relative change
  act_upd_frozen       INFUSE-U with the covariance frozen at the end of the calibration period

Usage: python eval_causal.py <out.jsonl> <glob> [<glob> ...] [--workers N]
"""
import glob
import json
import os
import sys
import time
from collections import Counter
from multiprocessing import Pool

import numpy as np

import inflight as I

FRAC = 0.2
MIN_WIN = 64


def explained(S, k):
    A = S[:k, :k] + 1e-6 * (np.trace(S[:k, :k]) / k + 1e-8) * np.eye(k)
    G = np.linalg.solve(A, S[:k, k:]).T
    return np.einsum('ij,ji->i', G, S[:k, k:])


def evaluate(path):
    t0 = time.time()
    y, pred, gt, E, L, H = I.load(path)
    n, _, C = E.shape
    n_cal = max(int(round(FRAC * n)), H + MIN_WIN)
    rec = {'path': path.replace('\\', '/'), 'n': n, 'H': H, 'n_cal': n_cal}
    if n - n_cal < n / 2:
        rec['skipped'] = True
        return rec
    B = I.tafas_batches(y, n, L)
    cal = E[:n_cal - H]                                     # forecasts fully revealed by the end of calibration
    m = (cal ** 2).mean((0, 2))
    s = np.sqrt((cal ** 2).mean((0, 1)))
    Z = cal / s
    S = np.einsum('nhc,ngc->hg', Z, Z, optimize=True) / (Z.shape[0] * C)
    d = np.diag(S)
    tot, cnt, ks = 0.0, 0, Counter()
    for i0, b_end, p in B:
        for i in range(max(i0, n_cal), b_end):
            k = min(i0 + p - i, H)
            if 0 < k < H and i + k < n:
                tot += m[:k].sum() + m[:H - k].sum(); ks[k] += 1
            else:
                tot += m.sum(); ks[0] += 1
            cnt += H
    rec['pred_ref'] = 100 * (tot / cnt - m.mean()) / m.mean()
    # Predicted variance reduction under Prop. 5.1 + Prop. 5.2 (in-flight cone with anchor k0)
    k0 = min(max((p for _, _, p in B), default=24), H - 1)
    k0 = max(k0, min(24, H - 1))
    v0 = np.maximum(d[k0:] - explained(S, k0), 1e-12) if 0 < k0 < H else None
    exp_sum = 0.0
    for k, c in ks.items():
        if k <= 0:
            continue
        ek = explained(S, k)
        vk = np.maximum(d[k:] - ek, 1e-12)
        if v0 is not None and k < k0:
            Lsh = H - k0
            v1, v2 = vk[:Lsh], v0[:Lsh]
            ok2 = v2 < (1.0 + I.CONE_TAU) * v1
            v_irr = np.where(ok2, I.GAMMA * np.minimum(v1, v2), 0.0)
            tv1 = np.maximum(v1 - v_irr, 1e-12); tv2 = np.maximum(v2 - v_irr, 1e-12)
            w0 = np.where(ok2, tv1 / (tv1 + tv2), 0.0)
            vc = np.where(ok2, v_irr + (1 - w0) ** 2 * tv1 + w0 ** 2 * tv2, v1)
            ek = ek.copy()
            ek[:Lsh] = np.maximum(d[k:k + Lsh] - vc, 0.0)
        exp_sum += c * float(ek.sum())
    rec['pred_upd'] = -100 * exp_sum / (sum(ks.values()) * d.sum())
    fr = (E[n_cal:] ** 2).mean()
    for mode, key, kw in (('refresh', 'act_ref', {}), ('update', 'act_upd', {}), ('update', 'act_upd_frozen',
                                                                                  {'cov_stop': n_cal})):
        w = I.tafas(y, pred, gt, E, L, H, mode=mode, batches=B, return_window=True, **kw)
        rec[key] = 100 * (w[n_cal:].mean() - fr) / fr
    rec['seconds'] = time.time() - t0
    return rec


def main():
    args = sys.argv[1:]
    workers = 6
    if '--workers' in args:
        i = args.index('--workers'); workers = int(args[i + 1]); del args[i:i + 2]
    out, pats = args[0], args[1:]
    done = {json.loads(l)['path'] for l in open(out)} if os.path.exists(out) else set()
    todo = sorted({p for pat in pats for p in glob.glob(pat) if p.replace('\\', '/') not in done}, key=os.path.getsize)
    with Pool(workers) as pool:
        for rec in pool.imap_unordered(evaluate, todo):
            with open(out, 'a') as f:
                f.write(json.dumps(rec) + '\n')
            if rec.get('skipped'):
                print('%-48s skipped (n=%d, n_cal=%d)' % (os.path.basename(rec['path']), rec['n'], rec['n_cal']), flush=True)
            else:
                print('%-48s ref pred %+.1f act %+.1f | upd pred %+.1f act %+.1f frozen-cov %+.1f (%.0fs)' % (
                    os.path.basename(rec['path']), rec['pred_ref'], rec['act_ref'], rec['pred_upd'], rec['act_upd'],
                    rec['act_upd_frozen'], rec['seconds']), flush=True)


if __name__ == '__main__':
    main()
