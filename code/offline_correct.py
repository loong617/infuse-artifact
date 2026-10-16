"""Offline headroom study: strictly causal, closed-form error correctors on dumped frozen predictions.

Window i has origin o = L + i and forecasts y[o + j], j = 0..H-1. Its error e[i, j] = y[o+j] - pred[i, j].
At window i only y[: o] is known, i.e. errors e[i', j'] with i' + j' <= i - 1; windows i' <= i - H are
fully revealed.

Usage: python offline_correct.py <npz> [more npz ...]
"""
import sys

import numpy as np


def latest_column(E, i, K):
    """Errors at the latest observed target time (o-1) made by the K most recent origins.
    z_k = e[i-k, k-1], k = 1..K  -> shape [K, C] (zeros where unavailable)."""
    C = E.shape[2]
    z = np.zeros((K, C), dtype=np.float64)
    for k in range(1, K + 1):
        if i - k >= 0 and k - 1 < E.shape[1]:
            z[k - 1] = E[i - k, k - 1]
    return z


def run(path):
    D = np.load(path)
    y, pred, L, H = D['y'].astype(np.float64), D['pred'].astype(np.float64), int(D['L']), int(D['H'])
    n, _, C = pred.shape
    gt = np.stack([y[L + i:L + i + H] for i in range(n)])
    E = gt - pred
    base = (E ** 2).mean()
    res = {'frozen': base}

    # C1: lead-wise bias from the last N fully revealed windows (per channel)
    for N in (H, 4 * H):
        cs = np.cumsum(np.concatenate([np.zeros((1, H, C)), E]), axis=0)
        se = 0.0
        for i in range(n):
            hi = i - H + 1          # windows 0..i-H are revealed -> slice [lo, hi)
            lo = max(0, hi - N)
            b = (cs[hi] - cs[lo]) / (hi - lo) if hi > lo else 0.0
            se += ((E[i] - b) ** 2).sum()
        res['C1 lead-bias N=%d' % N] = se / E.size

    # C3: ridge regression of the whole error trajectory on the latest error column (closed-form RLS)
    for K in (1, 4, 16):
        for lam in (1.0, 0.995):
            for pooled in (True, False):
                d = K + 1
                if pooled:
                    A = np.zeros((d, d)); B = np.zeros((d, H))
                else:
                    A = np.zeros((C, d, d)); B = np.zeros((C, d, H))
                Z = [None] * n
                se = 0.0
                for i in range(n):
                    z = latest_column(E, i, K)                       # [K, C]
                    Z[i] = np.concatenate([z, np.ones((1, C))], 0)   # [d, C]
                    j = i - H                                        # newly fully revealed window
                    if j >= 0:
                        zj, ej = Z[j], E[j]                          # [d,C], [H,C]
                        if pooled:
                            A = lam * A + zj @ zj.T
                            B = lam * B + zj @ ej.T
                        else:
                            A = lam * A + np.einsum('dc,ec->cde', zj, zj)
                            B = lam * B + np.einsum('dc,hc->cdh', zj, ej)
                    if j >= 4 * d:
                        if pooled:
                            reg = 1e-3 * np.trace(A) / d + 1e-8
                            W = np.linalg.solve(A + reg * np.eye(d), B)         # [d, H]
                            ehat = (Z[i].T @ W).T                                # [H, C]
                        else:
                            reg = 1e-3 * np.trace(A, axis1=1, axis2=2)[:, None, None] / d + 1e-8
                            W = np.linalg.solve(A + reg * np.eye(d)[None], B)  # [C, d, H]
                            ehat = np.einsum('dc,cdh->hc', Z[i], W)
                    else:
                        ehat = 0.0
                    se += ((E[i] - ehat) ** 2).sum()
                res['C3 K=%d lam=%.3f %s' % (K, lam, 'pooled' if pooled else 'per-ch')] = se / E.size

    print('==', path, 'n=%d C=%d H=%d' % (n, C, H))
    for k, v in res.items():
        print('  %-36s %.4f  (%+.1f%%)' % (k, v, 100 * (v - base) / base))


if __name__ == '__main__':
    for p in sys.argv[1:]:
        run(p)
