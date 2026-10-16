"""In-flight forecast updating and cone-coherent fusion:
  1. Sequential innovation factorization (Cholesky prefix orthogonalization) + horizon-progressive prefix covariance.
  2. In-flight trajectory cone revision governed by the Evidence-vs.-Staleness variance law.
  3. Co-target excess-precision fusion (shared post-decision innovation floor) and posterior-calibrated serving.
Strictly causal evaluation on dumped predictions.
"""
import os

import numpy as np
from scipy.linalg import solve_triangular

WARMUP = 32          # completed prefix trajectories needed before the first update
GAMMA = 0.75         # shared post-decision target innovation floor ratio
CONE_TAU = 0.15      # anchor tolerance in Evidence-vs.-Staleness gating

DATASETS = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'exchange_rate', 'weather', 'electricity', 'traffic', 'azure_functions']
_BOUNDS = {}


def plausible_bounds(path):
    """Per-channel plausibility bound for zero-shot foundation-model forecasts unrolled beyond the native 64-step
    horizon: training range extended by its width on both sides."""
    name = os.path.basename(path)
    ds = next((d for d in DATASETS if '_%s_' % d in name), None)
    if ds is None or not name.startswith('bolt-'):
        return None
    if ds not in _BOUNDS:
        f = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dumps', 'series_%s.npz' % ds)
        if not os.path.exists(f):
            return None
        S = np.load(f)
        x = S['x'][:int(S['n_train'])]
        lo, hi = x.min(0), x.max(0)
        _BOUNDS[ds] = (lo - (hi - lo), hi + (hi - lo))
    return _BOUNDS[ds]


def load(path):
    D = np.load(path)
    pred = D['pred'].astype(np.float32)
    b = plausible_bounds(path)
    if b is not None and b[0].shape[0] == pred.shape[2]:
        pred = np.clip(pred, b[0][None, None, :], b[1][None, None, :]).astype(np.float32)
    if 'y' not in D:
        base = os.path.basename(path).split('_', 1)[1]
        D = np.load(os.path.join(os.path.dirname(os.path.dirname(path)), base))
    y = D['y'].astype(np.float32)
    L, H = int(D['L']), int(D['H'])
    n = pred.shape[0]
    idx = np.arange(n)[:, None] + np.arange(H)[None, :] + L
    gt = y[idx]                                    # [n, H, C]
    return y, pred, gt, gt - pred, L, H


def paas_period(lookback):
    x = lookback - lookback.mean(0)
    amp = np.abs(np.fft.rfft(x, axis=0))
    power = (amp ** 2).mean(0)
    f = int(np.argmax(amp[:, int(np.argmax(power))]))
    return lookback.shape[0] // f if f > 0 else 24


def tafas_batches(y, n, L):
    out, i0 = [], 0
    while i0 < n:
        p = paas_period(y[i0:i0 + L])
        out.append((i0, min(i0 + p + 1, n), p))
        i0 = min(i0 + p + 1, n)
    return out


def chol_factors(Sig, K):
    """Cholesky prefix factorization of the leading K x K block of Sig:
      L L^T = Sig[:K, :K] + ridge * I
      U     = L^{-1} Sig[:K, :]       ([K, H] innovation-to-lead projection)
      csum  = cumulative sum of U^2   ([K, H] explained variance after k revealed steps).
    """
    A = Sig[:K, :K] + 1e-6 * (np.trace(Sig[:K, :K]) / K + 1e-8) * np.eye(K)
    L = np.linalg.cholesky(A)
    U = solve_triangular(L, Sig[:K, :], lower=True)
    return L, U, np.cumsum(U ** 2, axis=0)


class ErrorCov:
    """Online estimate of the lead-by-lead error covariance from revealed error trajectories.
    Channel errors are scaled by a running per-channel RMS before pooling."""

    def __init__(self, H, C, rho=1.0, shrink='lw', per_channel=0.0, stride=1):
        self.H, self.C, self.rho, self.shrink, self.kappa = H, C, rho, shrink, per_channel
        self.S = np.zeros((H, H)); self.w = 0.0
        self.ss = np.zeros(C); self.wc = 0.0
        self.q4 = 0.0
        self.Sc = np.zeros((C, H, H)) if per_channel > 0 else None
        self.q4c = np.zeros(C)
        self.wn = 0.0
        self.nwin = 0
        self.overlap = min(1.0, stride / H)
        self.last_lambda = 0.0

    def add(self, e):                              # e: [H, C]
        self.m2 = self.rho * getattr(self, 'm2', 0.0) + e ** 2
        self.nm = self.rho * getattr(self, 'nm', 0.0) + 1.0
        self.ss = self.rho * self.ss + (e ** 2).mean(0)
        self.wc = self.rho * self.wc + 1.0
        s = np.sqrt(self.ss / self.wc) + 1e-8
        z = e / s
        self.S = self.rho * self.S + z @ z.T
        self.w = self.rho * self.w + self.C
        self.q4 = self.rho * self.q4 + float(((z ** 2).sum(0) ** 2).sum())
        if self.Sc is not None:
            self.Sc = self.rho * self.Sc + np.einsum('hc,gc->chg', z, z)
            self.q4c = self.rho * self.q4c + (z ** 2).sum(0) ** 2
        self.wn = self.rho * self.wn + 1.0
        self.nwin += 1

    def add_many(self, Eb):
        """Vectorized addition of Eb [m, H, C] in order; exact match to sequential add(e) when rho == 1."""
        if self.rho != 1.0 or self.Sc is not None or len(Eb) == 0:
            for e in Eb:
                self.add(e)
            return
        m = len(Eb)
        sq = Eb.astype(np.float64) ** 2
        self.m2 = getattr(self, 'm2', 0.0) + sq.sum(0)
        self.nm = getattr(self, 'nm', 0.0) + m
        ss = self.ss[None, :] + np.cumsum(sq.mean(1), axis=0)
        wc = self.wc + np.arange(1, m + 1)[:, None]
        s = np.sqrt(ss / wc) + 1e-8
        Z = Eb / s[:, None, :]
        Zc = np.transpose(Z, (1, 0, 2)).reshape(self.H, -1)
        self.S = self.S + Zc @ Zc.T
        self.w += m * self.C
        self.q4 += float(((Z ** 2).sum(1) ** 2).sum())
        self.ss, self.wc = ss[-1], float(wc[-1, 0])
        self.wn += m
        self.nwin += m

    def gains_per_channel(self, k):
        """Per-channel covariance (no pooling), each shrunk with its own Ledoit-Wolf intensity."""
        N = max(self.wn, 1.0)
        Sig = self.Sc / N
        D = np.einsum('chh->ch', Sig)
        off = ((Sig ** 2).sum((1, 2)) - (D ** 2).sum(1)) + 1e-12
        b2 = np.maximum(self.q4c / N - (Sig ** 2).sum((1, 2)), 0.0) / (N * self.overlap)
        lam = np.minimum(1.0, b2 / off)[:, None, None]
        eye = np.eye(self.H)[None]
        Sig = (1 - lam) * Sig + lam * Sig * eye
        A = Sig[:, :k, :k] + 1e-6 * (np.trace(Sig[:, :k, :k], axis1=1, axis2=2)[:, None, None] / k + 1e-8) * np.eye(k)[None]
        return np.transpose(np.linalg.solve(A, Sig[:, :k, k:]), (0, 2, 1))

    def sigma(self):
        Sig = self.S / max(self.w, 1e-8)
        if self.shrink == 'lw':
            N = max(self.w, 1.0)
            b2 = max(self.q4 / N - float((Sig ** 2).sum()), 0.0) / (N * self.overlap)
            d2 = float(((Sig - np.diag(np.diag(Sig))) ** 2).sum()) + 1e-12
            lam = min(1.0, b2 / d2)
        else:
            lam = float(self.shrink)
        self.last_lambda = lam
        if lam > 0:
            Sig = (1 - lam) * Sig + lam * np.diag(np.diag(Sig))
        return Sig

    def gain(self, k, Sig=None):
        """G_k with e_hat[k:] = G_k @ e[:k] (pooled model)."""
        Sig = self.sigma() if Sig is None else Sig
        A = Sig[:k, :k] + 1e-6 * (np.trace(Sig[:k, :k]) / k + 1e-8) * np.eye(k)
        return np.linalg.solve(A, Sig[:k, k:]).T

    def ready(self, min_windows):
        return self.nwin >= min_windows


class Fusion:
    """Online Bates-Granger weights estimated from completed history (ablation baseline)."""

    def __init__(self, H, rho=1.0):
        self.a = np.zeros(H); self.b = np.zeros(H); self.c = np.zeros(H); self.rho = rho

    def add(self, k, e_upd, e_ref):
        self.a *= self.rho; self.b *= self.rho; self.c *= self.rho
        self.a[k:] += (e_upd ** 2).mean(1); self.b[k:] += (e_ref ** 2).mean(1)
        self.c[k:] += (e_upd * e_ref).mean(1)

    def weight(self, k):
        a, b, c = self.a[k:], self.b[k:], self.c[k:]
        den = a + b - 2 * c
        w = np.where(den > 1e-12, (a - c) / np.maximum(den, 1e-12), 0.5)
        w = np.where(a + b > 0, w, 0.5)
        return np.clip(w, 0.0, 1.0)


def tafas_both(y, pred, gt, E, L, H, min_windows=None, batches=None, cov_stop=None, gamma=GAMMA):
    """Compute both INFUSE-U ('update') and INFUSE ('fuse') in a single pass over the batches."""
    n, _, C = pred.shape
    B = tafas_batches(y, n, L) if batches is None else batches
    min_windows = WARMUP if min_windows is None else min_windows
    blocks = sorted({min(H, 96), min(H, 192), min(H, 336), H})
    h0 = blocks[0]
    covs = {hb: ErrorCov(hb, C) for hb in blocks}
    nxts = {hb: 0 for hb in blocks}
    out_u = E.copy()
    out_f = E.copy()
    for i0, b_end, p_len in B:
        T = i0 + p_len
        Kmax = min(T, max(p_len, 24), H - 1)
        facs = {}
        for hb in blocks:
            new = []
            while nxts[hb] < n and nxts[hb] + hb <= T and (cov_stop is None or nxts[hb] + hb <= cov_stop):
                new.append(nxts[hb]); nxts[hb] += 1
            if new:
                covs[hb].add_many(E[new[0]:new[-1] + 1, :hb])
            if covs[hb].ready(min_windows) and hb > 1:
                Sig = covs[hb].sigma()
                if hb > h0 and h0 in facs:
                    ratio = float(facs[h0][0][:h0].mean() / max(np.diag(Sig)[:h0].mean(), 1e-12))
                    Sig = Sig * ratio
                Kh = min(max(Kmax, 1), hb - 1)
                facs[hb] = (np.diag(Sig),) + chol_factors(Sig, Kh)

        eu_k, vu_k, vn_k = {}, {}, {}
        k_need = {min(T - i, H) for i in range(i0, b_end) if 0 < min(T - i, H) < H and i + min(T - i, H) < n}
        if 0 < Kmax < H and T - Kmax >= 0 and T < n:
            k_need.add(Kmax)
        for k in sorted(k_need):
            i = T - k
            eu = E[i, k:].copy()
            vu = np.full(H - k, np.inf)
            vn = np.ones(H - k)
            prev_hb = k
            for hb in blocks:
                if hb <= prev_hb or hb not in facs:
                    continue
                d, L_mat, U, csum_U2 = facs[hb]
                if k > L_mat.shape[0]:
                    continue
                nu = solve_triangular(L_mat[:k, :k], E[i, :k], lower=True)
                sl_p = slice(prev_hb - k, hb - k)
                eu[sl_p] = E[i, prev_hb:hb] - U[:k, prev_hb:hb].T @ nu
                vu[sl_p] = np.maximum(d[prev_hb:hb] - csum_U2[k - 1, prev_hb:hb], 1e-12)
                vn[sl_p] = d[prev_hb - k:hb - k]
                prev_hb = hb
            eu_k[k] = (eu, prev_hb > k)
            vu_k[k] = vu
            vn_k[k] = vn

        k0 = Kmax
        has_k0 = (k0 in eu_k and eu_k[k0][1])
        for i in range(i0, b_end):
            k = min(T - i, H)
            if k == 0 and i < n and has_k0:
                eu0, _ = eu_k[k0]
                vu0, vn0 = vu_k[k0], vn_k[k0]
                Lsh = H - k0
                ok0 = np.isfinite(vu0) & (vu0 <= 1.05 * vn0)
                v_irr = np.where(ok0, gamma * np.minimum(vu0, vn0), 0.0)
                tv0 = np.where(ok0, np.maximum(vu0 - v_irr, 1e-12), 1.0)
                tvn = np.where(ok0, np.maximum(vn0 - v_irr, 1e-12), 1.0)
                w = np.where(ok0, tv0 / (tv0 + tvn), 1.0)[:, None]
                out_f[i, :Lsh] = (1 - w) * eu0 + w * E[i, :Lsh]
                continue
            if not (0 < k < H and i + k < n):
                continue
            er = E[i + k, :H - k]
            eu, ok = eu_k[k]
            if not ok:
                out_u[i, k:] = eu
                out_f[i, k:] = er
                continue
            vu, vn = vu_k[k].copy(), vn_k[k]
            eu_c = eu.copy()
            vu_c = vu.copy()
            if k < k0 and has_k0:
                eu0, _ = eu_k[k0]
                vu0 = vu_k[k0]
                Lsh = H - k0
                v1, v2 = vu[:Lsh], vu0[:Lsh]
                ok2 = np.isfinite(v2) & (v2 < (1.0 + CONE_TAU) * v1)
                v_irr_u = np.where(ok2, gamma * np.minimum(v1, v2), 0.0)
                tv1 = np.where(ok2, np.maximum(v1 - v_irr_u, 1e-12), 1.0)
                tv2 = np.where(ok2, np.maximum(v2 - v_irr_u, 1e-12), 1.0)
                w0 = np.where(ok2, tv1 / (tv1 + tv2), 0.0)
                eu_c[:Lsh] = (1 - w0)[:, None] * eu[:Lsh] + w0[:, None] * eu0[:Lsh]
                vu_c[:Lsh] = np.where(ok2, v_irr_u + (1 - w0) ** 2 * tv1 + w0 ** 2 * tv2, v1)
            out_u[i, k:] = eu_c
            valid = np.isfinite(vu_c)
            v_irr_f = np.where(valid, gamma * np.minimum(vu_c, vn), 0.0)
            tvu = np.where(valid, np.maximum(vu_c - v_irr_f, 1e-12), 1.0)
            tvn = np.where(valid, np.maximum(vn - v_irr_f, 1e-12), 1.0)
            wf = np.where(valid, tvu / (tvu + tvn), 1.0)[:, None]
            out_f[i, k:] = (1 - wf) * eu_c + wf * er

    res_u = (float((out_u ** 2).mean()), float(np.abs(out_u).mean()), [float(v) for v in (out_u ** 2).mean((0, 2))])
    res_f = (float((out_f ** 2).mean()), float(np.abs(out_f).mean()), [float(v) for v in (out_f ** 2).mean((0, 2))])
    return res_u, res_f


def _progressive_tafas(y, pred, gt, E, L, H, mode, min_windows, batches, return_lead, return_window, cov_stop, gamma):
    """Progressive innovation factorization + in-flight cone revision + co-target excess-precision fusion."""
    n, _, C = pred.shape
    B = tafas_batches(y, n, L) if batches is None else batches
    min_windows = WARMUP if min_windows is None else min_windows
    blocks = sorted({min(H, 96), min(H, 192), min(H, 336), H})
    h0 = blocks[0]
    covs = {hb: ErrorCov(hb, C) for hb in blocks}
    nxts = {hb: 0 for hb in blocks}
    out = E.copy()
    for i0, b_end, p_len in B:
        T = i0 + p_len
        Kmax = min(T, max(p_len, 24), H - 1)
        facs = {}
        for hb in blocks:
            new = []
            while nxts[hb] < n and nxts[hb] + hb <= T and (cov_stop is None or nxts[hb] + hb <= cov_stop):
                new.append(nxts[hb]); nxts[hb] += 1
            if new:
                covs[hb].add_many(E[new[0]:new[-1] + 1, :hb])
            if covs[hb].ready(min_windows) and hb > 1:
                Sig = covs[hb].sigma()
                if hb > h0 and h0 in facs:
                    ratio = float(facs[h0][0][:h0].mean() / max(np.diag(Sig)[:h0].mean(), 1e-12))
                    Sig = Sig * ratio
                Kh = min(max(Kmax, 1), hb - 1)
                facs[hb] = (np.diag(Sig),) + chol_factors(Sig, Kh)

        eu_k, vu_k, vn_k = {}, {}, {}
        k_need = {min(T - i, H) for i in range(i0, b_end) if 0 < min(T - i, H) < H and i + min(T - i, H) < n}
        if 0 < Kmax < H and T - Kmax >= 0 and T < n:
            k_need.add(Kmax)
        for k in sorted(k_need):
            i = T - k
            eu = E[i, k:].copy()
            vu = np.full(H - k, np.inf)
            vn = np.ones(H - k)
            prev_hb = k
            for hb in blocks:
                if hb <= prev_hb or hb not in facs:
                    continue
                d, L_mat, U, csum_U2 = facs[hb]
                if k > L_mat.shape[0]:
                    continue
                nu = solve_triangular(L_mat[:k, :k], E[i, :k], lower=True)
                sl_p = slice(prev_hb - k, hb - k)
                eu[sl_p] = E[i, prev_hb:hb] - U[:k, prev_hb:hb].T @ nu
                vu[sl_p] = np.maximum(d[prev_hb:hb] - csum_U2[k - 1, prev_hb:hb], 1e-12)
                vn[sl_p] = d[prev_hb - k:hb - k]
                prev_hb = hb
            eu_k[k] = (eu, prev_hb > k)
            vu_k[k] = vu
            vn_k[k] = vn

        k0 = Kmax
        has_k0 = (k0 in eu_k and eu_k[k0][1])
        for i in range(i0, b_end):
            k = min(T - i, H)
            if mode == 'fuse' and k == 0 and i < n and has_k0:
                eu0, _ = eu_k[k0]
                vu0, vn0 = vu_k[k0], vn_k[k0]
                Lsh = H - k0
                ok0 = np.isfinite(vu0) & (vu0 <= 1.05 * vn0)
                v_irr = np.where(ok0, gamma * np.minimum(vu0, vn0), 0.0)
                tv0 = np.where(ok0, np.maximum(vu0 - v_irr, 1e-12), 1.0)
                tvn = np.where(ok0, np.maximum(vn0 - v_irr, 1e-12), 1.0)
                w = np.where(ok0, tv0 / (tv0 + tvn), 1.0)[:, None]
                out[i, :Lsh] = (1 - w) * eu0 + w * E[i, :Lsh]
                continue
            if not (0 < k < H and i + k < n):
                continue
            er = E[i + k, :H - k]
            eu, ok = eu_k[k]
            if not ok:
                out[i, k:] = eu if mode == 'update' else er
                continue
            vu, vn = vu_k[k].copy(), vn_k[k]
            eu_c = eu.copy()
            vu_c = vu.copy()
            if k < k0 and has_k0:
                eu0, _ = eu_k[k0]
                vu0 = vu_k[k0]
                Lsh = H - k0
                v1, v2 = vu[:Lsh], vu0[:Lsh]
                ok2 = np.isfinite(v2) & (v2 < (1.0 + CONE_TAU) * v1)
                v_irr_u = np.where(ok2, gamma * np.minimum(v1, v2), 0.0)
                tv1 = np.where(ok2, np.maximum(v1 - v_irr_u, 1e-12), 1.0)
                tv2 = np.where(ok2, np.maximum(v2 - v_irr_u, 1e-12), 1.0)
                w0 = np.where(ok2, tv1 / (tv1 + tv2), 0.0)
                eu_c[:Lsh] = (1 - w0)[:, None] * eu[:Lsh] + w0[:, None] * eu0[:Lsh]
                vu_c[:Lsh] = np.where(ok2, v_irr_u + (1 - w0) ** 2 * tv1 + w0 ** 2 * tv2, v1)
            if mode == 'update':
                out[i, k:] = eu_c
            else:
                valid = np.isfinite(vu_c)
                v_irr_f = np.where(valid, gamma * np.minimum(vu_c, vn), 0.0)
                tvu = np.where(valid, np.maximum(vu_c - v_irr_f, 1e-12), 1.0)
                tvn = np.where(valid, np.maximum(vn - v_irr_f, 1e-12), 1.0)
                wf = np.where(valid, tvu / (tvu + tvn), 1.0)[:, None]
                out[i, k:] = (1 - wf) * eu_c + wf * er

    mse, mae = float((out ** 2).mean()), float(np.abs(out).mean())
    if return_window:
        return (out ** 2).mean((1, 2))
    if return_lead:
        return mse, mae, (out ** 2).mean((0, 2))
    return mse, mae


def tafas(y, pred, gt, E, L, H, mode='update', rho=1.0, shrink='lw', min_windows=None, batches=None,
          refresh_every=1, return_lead=False, per_channel=False, return_window=False, cov_stop=None,
          cone=True, gamma=GAMMA):
    """TAFAS batch protocol evaluation.
    For the default INFUSE-U ('update') and INFUSE ('fuse') configurations (cone=True, rho=1.0, shrink='lw',
    not per_channel, refresh_every=1), uses progressive innovation factorization + cone revision + excess-precision
    fusion. Ablation modes ('single_u', 'single_f', 'fuse_eq', 'fuse_bg', 'last', 'mean', 'ar1', shrink != 'lw',
    rho != 1.0, per_channel=True) isolate individual components."""
    if (mode in ('update', 'fuse') and cone and rho == 1.0 and shrink == 'lw' and not per_channel
            and refresh_every == 1):
        return _progressive_tafas(y, pred, gt, E, L, H, mode, min_windows, batches, return_lead, return_window,
                                  cov_stop, gamma)
    if mode == 'single_u':
        mode = 'update'
    elif mode == 'single_f':
        mode = 'fuse'
    n, _, C = pred.shape
    B = tafas_batches(y, n, L) if batches is None else batches
    cov = ErrorCov(H, C, rho=rho, shrink=shrink, per_channel=1.0 if per_channel else 0.0)
    fus = Fusion(H)
    min_windows = WARMUP if min_windows is None else min_windows
    nxt = 0
    pend = []
    out = np.empty_like(E)
    for bi, (i0, b_end, p) in enumerate(B):
        T = i0 + p
        new = []
        while nxt < n and nxt + H <= T and (cov_stop is None or nxt + H <= cov_stop):
            new.append(nxt); nxt += 1
        if new:
            cov.add_many(E[new[0]:new[-1] + 1])
        keep = []
        for it in pend:
            if it[0] + H <= T:
                fus.add(it[1], it[2], it[3])
            else:
                keep.append(it)
        pend = keep
        can_call = (bi % refresh_every == 0)
        Sig = cov.sigma() if cov.ready(min_windows) else None
        fac = None
        if Sig is not None and not per_channel and mode not in ('frozen', 'refresh', 'last', 'mean', 'ar1'):
            Kmax = min(p, H - 1)
            if Kmax >= 1:
                fac = (np.diag(Sig),) + chol_factors(Sig, Kmax)
        if mode == 'ar1' and Sig is not None:
            phi = float(np.clip(np.trace(Sig, 1) / max(np.trace(Sig[:-1, :-1]), 1e-12), -1.0, 1.0))
            phi_pow = phi ** np.arange(1, H + 1)
        g_pc = {}
        for i in range(i0, b_end):
            k = min(T - i, H)
            e = E[i].copy()
            if mode != 'frozen' and 0 < k < H and i + k < n:
                e_ref = gt[i, k:] - pred[i + k, :H - k]
                if mode == 'last':
                    e_upd = e[k:] - E[i, k - 1][None, :]
                elif mode == 'mean':
                    e_upd = e[k:] - E[i, :k].mean(0)[None, :]
                elif Sig is None:
                    e_upd = e[k:]
                elif mode == 'ar1':
                    e_upd = e[k:] - phi_pow[:H - k, None] * E[i, k - 1][None, :]
                elif per_channel:
                    if k not in g_pc:
                        g_pc[k] = cov.gains_per_channel(k)
                    e_upd = e[k:] - np.einsum('chk,kc->hc', g_pc[k], E[i, :k])
                elif mode in ('refresh', 'frozen') or fac is None:
                    e_upd = e[k:]
                else:
                    d, L_mat, U, csum_U2 = fac
                    nu = solve_triangular(L_mat[:k, :k], E[i, :k], lower=True)
                    e_upd = e[k:] - U[:k, k:].T @ nu
                if mode in ('update', 'last', 'mean', 'ar1') or (mode in ('refresh', 'fuse', 'fuse_eq', 'fuse_bg')
                                                                 and not can_call):
                    e[k:] = e_upd
                elif mode == 'refresh':
                    e[k:] = e_ref
                elif mode == 'fuse_eq':
                    e[k:] = 0.5 * (e_upd + e_ref)
                elif mode == 'fuse':
                    if fac is None:
                        w = np.ones((H - k, 1))
                    else:
                        d, _, _, csum_U2 = fac
                        v_upd = np.maximum(d[k:] - csum_U2[k - 1, k:], 1e-12)
                        v_ref = d[:H - k]
                        w = (v_upd / (v_upd + v_ref))[:, None]
                    e[k:] = (1 - w) * e_upd + w * e_ref
                elif mode == 'fuse_bg':
                    w = fus.weight(k)[:, None]
                    e[k:] = (1 - w) * e_upd + w * e_ref
                if mode == 'fuse_bg':
                    pend.append((i, k, e_upd, e_ref))
            out[i] = e
    mse, mae = float((out ** 2).mean()), float(np.abs(out).mean())
    if return_window:
        return (out ** 2).mean((1, 2))
    if return_lead:
        return mse, mae, (out ** 2).mean((0, 2))
    return mse, mae


def freshness_prediction(E, L, H, y):
    """Refresh MSE predicted from the frozen lead-error curve m(j) alone (stationarity assumption)."""
    n = E.shape[0]
    m = (E ** 2).mean((0, 2))
    tot, cnt = 0.0, 0
    for i0, b_end, p in tafas_batches(y, n, L):
        for i in range(i0, b_end):
            k = min(i0 + p - i, H)
            if 0 < k < H and i + k < n:
                tot += m[:k].sum() + m[:H - k].sum()
            else:
                tot += m.sum()
            cnt += H
    return tot / cnt, float(m.mean())


def service(E, pred, gt, H, r, Hs, mode='hold', rho=1.0, shrink='lw', min_windows=None, return_served=False,
            return_std=False, gamma=GAMMA):
    """Decision-time serving under a model-call budget (calls at origins 0, r, 2r, ...).
    mode:
      'hold'   : serve the latest unrevised forecast.
      'update' : single-origin innovation update of the latest call (INFUSE-U).
      'fuse'   : cone-coherent revision + excess-precision fusion of the active in-flight calls {k, r + k} (INFUSE).
    return_std: also return the closed-form posterior error standard deviation scale [n, Hs, C] for uncertainty-aware
                downstream provisioning."""
    n, _, C = pred.shape
    assert Hs + r - 1 <= H
    cov = ErrorCov(H, C, rho=rho, shrink=shrink, stride=r)
    min_windows = WARMUP if min_windows is None else min_windows
    min_w = min_windows // max(r, 1) + 3
    calls = list(range(0, n, r))
    nxt_c = 0
    se, cnt = 0.0, 0
    fac = None
    served = np.empty((n, Hs, C), np.float32) if return_served else None
    std_scale = np.ones((n, Hs, C), np.float32) if return_std else None
    for t in range(0, n):
        changed = False
        while nxt_c < len(calls) and calls[nxt_c] + H <= t:
            cov.add(E[calls[nxt_c]]); nxt_c += 1; changed = True
        if changed and (fac is None or t % max(r, 12) == 0):
            fac = None
        ready = cov.ready(min_w)
        if ready and fac is None:
            Sig = cov.sigma()
            Kmax = min(H - Hs, max(2 * r, 24))
            L_mat, U, csum_U2 = chol_factors(Sig, Kmax)
            sc = np.sqrt(cov.ss / max(cov.wc, 1.0)) + 1e-8
            fac = (np.diag(Sig), L_mat, U, csum_U2, sc / (sc.mean() + 1e-8))
        i = r * (t // r)
        k = t - i
        e1 = E[i, k:k + Hs]
        v_post = np.ones(Hs)
        sc_rel = np.ones(C)
        if ready and mode in ('update', 'fuse'):
            d, L_mat, U, csum_U2, sc_rel = fac
            if k > 0:
                nu1 = solve_triangular(L_mat[:k, :k], E[i, :k], lower=True)
                e1 = e1 - U[:k, k:k + Hs].T @ nu1
                v1 = np.maximum(d[k:k + Hs] - csum_U2[k - 1, k:k + Hs], 1e-12)
            else:
                v1 = d[:Hs]
            v_post = v1
            m_anc = max(1, min((H - Hs - k) // max(r, 1), 12 // max(r, 1)))
            i0 = i - m_anc * r
            k0 = t - i0
            if mode == 'fuse' and i0 >= 0 and k0 + Hs <= H and 0 < k0 <= L_mat.shape[0]:
                nu0 = solve_triangular(L_mat[:k0, :k0], E[i0, :k0], lower=True)
                e0 = E[i0, k0:k0 + Hs] - U[:k0, k0:k0 + Hs].T @ nu0
                v0 = np.maximum(d[k0:k0 + Hs] - csum_U2[k0 - 1, k0:k0 + Hs], 1e-12)
                ok = (k == 0) | (v0 < (1.0 + CONE_TAU) * v1)
                v_irr = gamma * np.minimum(v0, v1)
                tv0 = np.maximum(v0 - v_irr, 1e-12)
                tv1 = np.maximum(v1 - v_irr, 1e-12)
                w = np.where(ok, tv0 / (tv0 + tv1), 1.0)[:, None]
                e1 = (1 - w) * e0 + w * e1
                v_post = np.where(ok, v_irr + (tv0 * tv1) / (tv0 + tv1), v1)
            v_post = np.sqrt(v_post / np.maximum(d[:Hs], 1e-8))
        se += float((e1 ** 2).sum()); cnt += e1.size
        if return_served:
            served[t] = e1
        if return_std:
            std_scale[t] = v_post[:, None] * (0.5 + 0.5 * sc_rel[None, :])
    out = (se / cnt, len(calls))
    if return_served:
        out = out + (served,)
    if return_std:
        out = out + (std_scale,)
    return out
