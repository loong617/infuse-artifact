"""Direct per-batch CPU time of INFUSE-U / INFUSE under the TAFAS protocol (single-threaded NumPy): progressive
online covariance update, Ledoit-Wolf shrinkage, Cholesky prefix innovation factorization, in-flight cone revision,
and excess-precision fusion of every forecast in the batch. Results to timing_h.json.
Usage: python time_infuse2.py <npz> [...]   (run with OMP/MKL/OPENBLAS_NUM_THREADS=1)"""
import json
import os
import sys
import time

import numpy as np
from scipy.linalg import solve_triangular

import inflight as I


def run(path, fuse):
    y, pred, gt, E, L, H = I.load(path)
    n, _, C = pred.shape
    B = I.tafas_batches(y, n, L)
    blocks = sorted({min(H, 96), min(H, 192), min(H, 336), H})
    h0 = blocks[0]
    covs = {hb: I.ErrorCov(hb, C) for hb in blocks}
    nxts = {hb: 0 for hb in blocks}
    tot, nb = 0.0, 0
    for i0, b_end, p_len in B:
        T = i0 + p_len
        t0 = time.perf_counter()
        Kmax = min(T, max(p_len, 24), H - 1)
        facs = {}
        for hb in blocks:
            stop = min(n, max(nxts[hb], T - hb + 1))
            if stop > nxts[hb]:
                covs[hb].add_many(E[nxts[hb]:stop, :hb]); nxts[hb] = stop
            if covs[hb].ready(I.WARMUP) and hb > 1:
                Sig = covs[hb].sigma()
                if hb > h0 and h0 in facs:
                    Sig = Sig * float(facs[h0][0][:h0].mean() / max(np.diag(Sig)[:h0].mean(), 1e-12))
                Kh = min(max(Kmax, 1), hb - 1)
                facs[hb] = (np.diag(Sig),) + I.chol_factors(Sig, Kh)
        if facs:
            eu_k, vu_k, vn_k = {}, {}, {}
            k_need = {min(T - i, H) for i in range(i0, b_end) if 0 < min(T - i, H) < H and i + min(T - i, H) < n}
            if 0 < Kmax < H and T - Kmax >= 0 and T < n:
                k_need.add(Kmax)
            for k in sorted(k_need):
                i = T - k
                eu = pred[i, k:].copy()
                vu = np.full(H - k, np.inf); vn = np.ones(H - k)
                prev_hb = k
                for hb in blocks:
                    if hb <= prev_hb or hb not in facs:
                        continue
                    d, L_mat, U, csum_U2 = facs[hb]
                    if k > L_mat.shape[0]:
                        continue
                    nu = solve_triangular(L_mat[:k, :k], E[i, :k], lower=True)
                    sl_p = slice(prev_hb - k, hb - k)
                    eu[sl_p] = pred[i, prev_hb:hb] + U[:k, prev_hb:hb].T @ nu
                    vu[sl_p] = np.maximum(d[prev_hb:hb] - csum_U2[k - 1, prev_hb:hb], 1e-12)
                    vn[sl_p] = d[prev_hb - k:hb - k]
                    prev_hb = hb
                eu_k[k] = eu; vu_k[k] = vu; vn_k[k] = vn
            k0 = Kmax
            for i in range(i0, b_end):
                k = min(T - i, H)
                if not (0 < k < H and i + k < n):
                    continue
                upd = eu_k[k].copy(); vu = vu_k[k].copy(); vn = vn_k[k]
                if k < k0 and k0 in eu_k:
                    Lsh = H - k0
                    v1, v2 = vu[:Lsh], vu_k[k0][:Lsh]
                    ok2 = np.isfinite(v2) & (v2 < (1.0 + I.CONE_TAU) * v1)
                    v_irr = np.where(ok2, I.GAMMA * np.minimum(v1, v2), 0.0)
                    tv1 = np.maximum(v1 - v_irr, 1e-12); tv2 = np.maximum(v2 - v_irr, 1e-12)
                    w0 = np.where(ok2, tv1 / (tv1 + tv2), 0.0)
                    upd[:Lsh] = (1 - w0)[:, None] * upd[:Lsh] + w0[:, None] * eu_k[k0][:Lsh]
                    vu[:Lsh] = np.where(ok2, v_irr + (1 - w0) ** 2 * tv1 + w0 ** 2 * tv2, v1)
                if fuse:
                    v_irr = I.GAMMA * np.minimum(vu, vn)
                    tvu = np.maximum(vu - v_irr, 1e-12); tvn = np.maximum(vn - v_irr, 1e-12)
                    w = (tvu / (tvu + tvn))[:, None]
                    upd = (1 - w) * upd + w * pred[i + k, :H - k]
            tot += time.perf_counter() - t0; nb += 1
    return 1e3 * tot / max(nb, 1), H, C


out = json.load(open('timing_h.json')) if os.path.exists('timing_h.json') else {}
for path in sys.argv[1:]:
    u, H, C = run(path, False)
    f, _, _ = run(path, True)
    out[os.path.basename(path)[:-4]] = {'H': H, 'C': C, 'update_ms': u, 'fuse_ms': f}
    print('%s H=%d C=%d  INFUSE-U %.2f ms/batch  INFUSE %.2f ms/batch' % (os.path.basename(path), H, C, u, f), flush=True)
json.dump(out, open('timing_h.json', 'w'), indent=1)
