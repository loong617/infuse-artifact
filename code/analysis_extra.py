"""Extra analyses for the paper (results to analysis_extra.json, keyed by dump name).

pred_update: MSE reduction of INFUSE-U under the TAFAS protocol predicted by the Gaussian model alone (Prop. 2), from
             the pooled error covariance of the setting: sum over forecasts of the explained pending variance,
             tr(S_pr S_rr^-1 S_rp), divided by the total error variance.
gain_k:      relative MSE reduction of the pending leads after k revealed targets, for k in K, with the covariance
             estimated online from fully revealed forecasts (re-estimated every 50 origins).
Usage: python analysis_extra.py <glob> [<glob> ...] [--workers N] [--gain-k]
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

K = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64)
OUT = 'analysis_extra.json'


def explained(S, k):
    A = S[:k, :k] + 1e-6 * (np.trace(S[:k, :k]) / k + 1e-8) * np.eye(k)
    G = np.linalg.solve(A, S[:k, k:]).T                                   # [H-k, k]
    return np.einsum('ij,ji->i', G, S[:k, k:])                            # explained variance per pending lead


def pred_update(y, E, L, H):
    n, _, C = E.shape
    s = np.sqrt((E ** 2).mean((0, 1)))
    Z = E / s
    S = np.einsum('nhc,ngc->hg', Z, Z, optimize=True) / (n * C)
    d = np.diag(S)
    cnt = Counter()
    for i0, b_end, p in I.tafas_batches(y, n, L):
        for i in range(i0, b_end):
            k = min(i0 + p - i, H)
            if 0 < k < H and i + k < n:
                cnt[k] += 1
            else:
                cnt[0] += 1
    num = sum(c * explained(S, k).sum() for k, c in cnt.items() if k > 0)
    return float(100 * num / (sum(cnt.values()) * d.sum()))


def gain_k(E, H, every=50, warm=64, k0=24):
    n, _, C = E.shape
    cov = I.ErrorCov(H, C)
    nxt = 0
    acc = {k: [0.0, 0.0] for k in K if k < H}
    acc_c = {k: [0.0, 0.0] for k in K if k < H}
    for i0 in range(warm + H, n, every):
        new = []
        while nxt + H <= i0:
            new.append(nxt); nxt += 1
        if new:
            cov.add_many(E[new[0]:new[-1] + 1])
        Sig = cov.sigma()
        d = np.diag(Sig)
        Kmax = max(max(acc), k0)
        L_mat, U, csum_U2 = I.chol_factors(Sig, Kmax)
        G_map = {k: cov.gain(k, Sig) for k in set(acc) | ({k0} if k0 < H else set())}
        v_map = {k: np.maximum(d[k:] - csum_U2[k - 1, k:], 1e-12) for k in G_map}
        m_end = min(i0 + every, n)
        blk = E[i0:m_end]
        # Anchor origins i - (k0 - k) for cone revision when k < k0
        for k in acc:
            e_upd = blk[:, k:] - np.einsum('jk,nkc->njc', G_map[k], blk[:, :k])
            acc[k][0] += float((blk[:, k:] ** 2).sum())
            acc[k][1] += float((e_upd ** 2).sum())
            eu_c = e_upd.copy()
            if k < k0 and k0 in G_map:
                sh = k0 - k
                blk0 = E[i0 - sh:m_end - sh]
                eu0 = blk0[:, k0:] - np.einsum('jk,nkc->njc', G_map[k0], blk0[:, :k0])
                Lsh = H - k0
                v1, v2 = v_map[k][:Lsh], v_map[k0][:Lsh]
                ok2 = v2 < (1.0 + I.CONE_TAU) * v1
                v_irr = np.where(ok2, I.GAMMA * np.minimum(v1, v2), 0.0)
                tv1 = np.maximum(v1 - v_irr, 1e-12); tv2 = np.maximum(v2 - v_irr, 1e-12)
                w0 = np.where(ok2, tv1 / (tv1 + tv2), 0.0)[None, :, None]
                eu_c[:, :Lsh] = (1 - w0) * eu_c[:, :Lsh] + w0 * eu0
            acc_c[k][0] += float((blk[:, k:] ** 2).sum())
            acc_c[k][1] += float((eu_c ** 2).sum())
    return ({str(k): 100 * (v[1] - v[0]) / v[0] for k, v in acc.items()},
            {str(k): 100 * (v[1] - v[0]) / v[0] for k, v in acc_c.items()})


def run(args):
    path, do_k = args
    t0 = time.time()
    y, pred, gt, E, L, H = I.load(path)
    rec = {'pred_update': pred_update(y, E, L, H)}
    if do_k:
        rec['gain_k'], rec['gain_k_cone'] = gain_k(E, H)
    rec['seconds'] = time.time() - t0
    return os.path.basename(path)[:-4], rec


def main():
    args = sys.argv[1:]
    workers = 4
    if '--workers' in args:
        i = args.index('--workers'); workers = int(args[i + 1]); del args[i:i + 2]
    do_k = '--gain-k' in args
    args = [a for a in args if a != '--gain-k']
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    todo = sorted({p for pat in args for p in glob.glob(pat)
                   if os.path.basename(p)[:-4] not in res or (do_k and 'gain_k' not in res[os.path.basename(p)[:-4]])},
                  key=os.path.getsize)
    with Pool(workers) as pool:
        for name, rec in pool.imap_unordered(run, [(p, do_k) for p in todo]):
            res = json.load(open(OUT)) if os.path.exists(OUT) else {}
            res.setdefault(name, {}).update(rec)
            json.dump(res, open(OUT, 'w'))
            print('%-45s pred_update %.1f%%%s (%.0fs)' % (name, rec['pred_update'],
                  ('  gain_k ' + ' '.join('%s:%.1f' % kv for kv in rec['gain_k'].items())) if 'gain_k' in rec else '',
                  rec['seconds']), flush=True)


if __name__ == '__main__':
    main()
