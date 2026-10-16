"""Offline replica of the TAFAS batch protocol with partially observed ground truth (POGT).

Batches of p+1 consecutive origins, p recomputed per batch from the first window's lookback exactly as
TAFAS's PAAS does. At batch end the first k = p - b leads of window b are revealed; leads >= k may be
replaced (the last window, k = 0, is never adjusted). Compared estimators for the unrevealed leads:
  refresh : the frozen model's forecast issued at the batch end (newest lookback)
  krig    : conditional-Gaussian update of the old forecast given its revealed error prefix
  fuse    : per-lead closed-form combination of krig and refresh (weights from past revealed windows)
Usage: python offline_pogt.py <npz> [...]
"""
import sys

import numpy as np


def paas_period(lookback):
    x = lookback - lookback.mean(0)
    amp = np.abs(np.fft.rfft(x, axis=0))
    power = (amp ** 2).mean(0)
    f = int(np.argmax(amp[:, int(np.argmax(power))]))
    return lookback.shape[0] // f if f > 0 else 24


def batches(y, n, L):
    out, i0 = [], 0
    while i0 < n:
        p = paas_period(y[i0:i0 + L])
        b_end = min(i0 + p + 1, n)
        out.append((i0, b_end, p))
        i0 = b_end
    return out


def run(path):
    D = np.load(path)
    y, pred, L, H = D['y'].astype(np.float64), D['pred'].astype(np.float64), int(D['L']), int(D['H'])
    n, _, C = pred.shape
    gt = np.stack([y[L + i:L + i + H] for i in range(n)])
    E = gt - pred
    base = (E ** 2).mean()
    B = batches(y, n, L)
    ps = [p for _, _, p in B]
    res = {}
    for mode in ('refresh', 'krig', 'fuse'):
        S = np.zeros((H, H)); cnt = 0.0; nxt = 0
        # fusion statistics per lead: sums of e_k^2, e_r^2, e_k*e_r over revealed history
        F = np.zeros((3, H))
        pend = []                       # (window, k, ehat_krig, e_ref) awaiting full revelation
        se = 0.0
        for i0, b_end, p in B:
            T = i0 + p                  # ground truth known up to origin index T (exclusive of y[L+T:])
            while nxt < n and nxt + H <= T:
                S += E[nxt] @ E[nxt].T; cnt += C; nxt += 1
            keep = []
            for (w, k, ek, er) in pend:
                if w + H <= T:          # fully revealed: update fusion stats on leads k..H-1
                    F[0, k:] += (ek ** 2).sum(1); F[1, k:] += (er ** 2).sum(1); F[2, k:] += (ek * er).sum(1)
                else:
                    keep.append((w, k, ek, er))
            pend = keep
            for i in range(i0, b_end):
                k = min(T - i, H)
                e = E[i].copy()
                if 0 < k < H and i + k < n:
                    e_ref = gt[i, k:] - pred[i + k, :H - k]
                    e_kr = e[k:].copy()
                    if cnt > 10 * C:
                        Sig = S / cnt
                        A = Sig[:k, :k] + 1e-6 * np.trace(Sig[:k, :k]) / k * np.eye(k)
                        G = np.linalg.solve(A, Sig[:k, k:]).T
                        e_kr = e[k:] - G @ E[i, :k]
                    if mode == 'refresh':
                        e[k:] = e_ref
                    elif mode == 'krig':
                        e[k:] = e_kr
                    else:
                        # optimal per-lead weight w on the refresh: minimizes E[(1-w) e_k + w e_r]^2
                        a, b_, c = F[0, k:], F[1, k:], F[2, k:]
                        den = a + b_ - 2 * c
                        wgt = np.where(den > 1e-9, np.clip((a - c) / np.maximum(den, 1e-9), 0, 1), 0.5)
                        e[k:] = (1 - wgt)[:, None] * e_kr + wgt[:, None] * e_ref
                    pend.append((i, k, e_kr, e_ref))
                se += (e ** 2).sum()
        res[mode] = se / E.size
    print('== %s  n=%d C=%d H=%d  periods=%s  frozen=%.4f' % (path, n, C, H, sorted(set(ps)), base))
    for k_, v in res.items():
        print('   %-8s %.4f (%+.1f%%)' % (k_, v, 100 * (v - base) / base))


if __name__ == '__main__':
    for p_ in sys.argv[1:]:
        run(p_)
