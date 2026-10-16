"""ELF (Lee et al., ICML 2025, "Lightweight Online Adaption for Time Series Foundation Model Forecasts"), re-implemented
from Algorithms 1-5 and Appendix B of the paper, applied to the dumped forecasts of any forecaster.

ELF-Forecaster: linear model in the Fourier domain (context DFT with the highest (1 - alpha) share of frequencies
removed -> complex weights -> lowest alpha*H/2 RFT components of the target -> iRFT), mean-normalised context, ridge
lambda = 20, refit in closed form every M = 200 steps on all fully revealed (context, target) pairs of the deployment
stream (one weight matrix shared across channels; the series are already standardised with training statistics).
ELF-Weighter: fast (last B = 5 updates) and slow exponential weighters (eta = 0.5) on the average MASE of the
forecasts revealed since the last update, merged by a third exponential weighter; ELF is not used during the first
5 update steps after the first fit. Emitted forecasts at origin i only use data observed before i.

Output: dumps/grid/elf/ELF_<base>.npz or dumps/fm/elf/ELF_<base>.npz (pred only; the series comes from the base dump).
Usage: python run_elf.py <glob of base dumps> [...] [--workers N]
"""
import glob
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

import inflight as I

LE, ALPHA, LAM, M, ETA, B, WARM = 512, 0.9, 20.0, 200, 0.5, 5, 5
SEAS = {'ETTh1': 24, 'ETTh2': 24, 'ETTm1': 96, 'ETTm2': 96, 'weather': 144, 'exchange_rate': 1, 'azure_functions': 60,
        'electricity': 24, 'traffic': 24}


def sigmoid(v):
    return float(0.5 * (1.0 + np.tanh(0.5 * v)))


def logit(p):
    p = min(max(p, 1e-12), 1 - 1e-12)
    return float(np.log(p / (1 - p)))


def out_path(path):
    return os.path.join(os.path.dirname(path), 'elf', 'ELF_' + os.path.basename(path))


def run(path):
    t0 = time.time()
    y, pred, gt, E, L, H = I.load(path)
    n, _, C = pred.shape
    ds = next(d for d in I.DATASETS if '_%s_' % d in os.path.basename(path))
    S = np.load('dumps/series_%s.npz' % ds)
    ts = int(S['test_start'])
    x = S['x'][:ts + len(y)].astype(np.float64)                 # x[ts + u] = y[u]
    s = SEAS[ds]
    keep = np.r_[0:int(np.floor(ALPHA * LE / 2)) + 1, LE - int(np.floor(ALPHA * LE / 2)):LE]
    hk = int(np.floor(ALPHA * H / 2))
    d = len(keep)
    XtX = np.zeros((d, d), complex); XtY = np.zeros((d, hk), complex)
    W = None
    ctx_all = np.lib.stride_tricks.sliding_window_view(x[ts + L - LE:ts + L + n - 1], LE, axis=0)   # [n, C, LE]
    tgt = gt                                                                                          # [n, H, C]

    def feats(ctx):                                   # ctx: [m, C, LE] -> (complex features [m*C, d], means [m, C])
        mu = ctx.mean(-1)
        Xf = np.fft.fft(ctx - mu[..., None], axis=-1, norm='ortho')[..., keep]
        return Xf.reshape(-1, d), mu

    def scale(ctx):                                   # MASE denominator per window and channel (floored at 0.01 std:
        if s >= LE:                                   # near-constant channels such as rain would dominate otherwise)
            return np.maximum(np.abs(np.diff(ctx, axis=-1)).mean(-1), 1e-2)
        return np.maximum(np.abs(ctx[..., s:] - ctx[..., :-s]).mean(-1), 1e-2)

    ef = np.empty_like(pred)
    out = np.empty_like(pred)
    w_slow, beta, hist = 0.5, 0.5, []
    w_fast = 0.5
    fitted_at = None
    n_fit = 0
    nxt_fit = 0                                       # next window whose (context, target) enters the fit
    t = 0
    while t < n:
        if t > 0:
            # update step at decision index t: fit on newly revealed windows, update the weighter
            new = np.arange(nxt_fit, max(nxt_fit, t - H + 1))
            new = new[new + H <= t]
            if len(new):
                Xf, mu = feats(ctx_all[new])
                Yc = np.transpose(tgt[new], (0, 2, 1)) - mu[..., None]                 # [m, C, H]
                Yf = np.fft.rfft(Yc, axis=-1, norm='ortho')[..., :hk].reshape(-1, hk)
                XtX += Xf.conj().T @ Xf; XtY += Xf.conj().T @ Yf
                W = np.linalg.solve(XtX + LAM * np.eye(d), XtY)
                nxt_fit = new[-1] + 1
                if fitted_at is None:
                    fitted_at = t
                n_fit += 1
            rev = np.arange(max(0, t - M - H), t - H + 1) if t - H >= 0 else np.arange(0)
            rev = rev[(rev + H <= t) & (rev + H > t - M)]
            if fitted_at is not None and len(rev) and rev[0] >= fitted_at:
                sc = scale(ctx_all[rev])[:, None, :]                                    # [m, 1, C]

                def mase(p):
                    return float((np.abs(tgt[rev] - p) / sc).mean())
                l1, l2 = mase(pred[rev]), mase(ef[rev])
                ls = mase(w_slow * pred[rev] + (1 - w_slow) * ef[rev])
                lf = mase(w_fast * pred[rev] + (1 - w_fast) * ef[rev])
                # exponential-weighting updates written on the logit scale (numerically stable)
                w_slow = sigmoid(logit(w_slow) - ETA * (l1 - l2))
                hist = (hist + [(l1, l2)])[-B:]
                s1, s2 = sum(h[0] for h in hist), sum(h[1] for h in hist)
                w_fast = sigmoid(-ETA * (s1 - s2))
                beta = sigmoid(logit(beta) - ETA * (lf - ls))
        w = beta * w_fast + (1 - beta) * w_slow
        if fitted_at is None or n_fit <= WARM:
            w = 1.0
        blk = np.arange(t, min(t + M, n))
        ctx = ctx_all[blk]
        if W is None:                                 # naive seasonal forecaster before the first fit
            per = min(s, LE) if s > 1 else 1
            idx = LE - per + (np.arange(H) % per)
            e = ctx[..., idx]
        else:
            Xf, mu = feats(ctx)
            Yf = np.zeros((len(blk) * C, H // 2 + 1), complex)
            Yf[:, :hk] = Xf @ W
            e = np.fft.irfft(Yf, n=H, axis=-1, norm='ortho').reshape(len(blk), C, H) + mu[..., None]
        ef[blk] = np.transpose(e, (0, 2, 1))
        out[blk] = w * pred[blk] + (1 - w) * ef[blk]
        t = blk[-1] + 1
    o = out_path(path)
    os.makedirs(os.path.dirname(o), exist_ok=True)
    np.savez(o, pred=out.astype(np.float32))
    mse = lambda p: float(((gt - p) ** 2).mean())
    return os.path.basename(path), mse(pred), mse(ef), mse(out), float(w), time.time() - t0


def main():
    args = sys.argv[1:]
    workers = 6
    if '--workers' in args:
        i = args.index('--workers'); workers = int(args[i + 1]); del args[i:i + 2]
    todo = sorted({p for pat in args for p in glob.glob(pat) if not os.path.exists(out_path(p))}, key=os.path.getsize)
    with Pool(workers) as pool:
        for name, a, b, c, w, sec in pool.imap_unordered(run, todo):
            print('%-45s base %.4f elf-forecaster %.4f ELF %.4f (%+.1f%%) last w_FM %.2f (%.0fs)' % (
                name, a, b, c, 100 * (c - a) / a, w, sec), flush=True)


if __name__ == '__main__':
    main()
