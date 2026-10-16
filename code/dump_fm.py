"""Zero-shot Chronos-Bolt forecasts for every test origin (stride 1); run in the `tsfm` conda env.

Usage: python dump_fm.py <dataset> <model_id> <H> <ctx> <out.npz>
Horizons beyond the native 64 steps are produced autoregressively: the median of each 64-step block is
appended to the context (the same unrolling Chronos-Bolt uses, with the median as the point path).
Output layout matches dump_preds.py: y (test series incl. first lookback), pred (n x H x C, fp16).
"""
import sys
import time

import numpy as np
import torch
from chronos import BaseChronosPipeline

BLOCK = 64


def forecast(pipe, ctx, H):
    out = []
    cur = ctx
    while sum(o.shape[1] for o in out) < H:
        q, _ = pipe.predict_quantiles(cur, prediction_length=BLOCK, quantile_levels=[0.5])
        med = q[..., 0].float()
        out.append(med)
        cur = torch.cat([cur, med], dim=1)[:, -ctx.shape[1]:]
    return torch.cat(out, dim=1)[:, :H]


def main():
    name, model_id, H, ctx_len, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    S = np.load('dumps/series_%s.npz' % name)
    x, ts, L = S['x'], int(S['test_start']), int(S['L'])
    y = x[ts:]
    T, C = y.shape
    origins = np.arange(ts + L, ts + T - H + 1)
    n = len(origins)
    pipe = BaseChronosPipeline.from_pretrained(model_id, device_map='cuda', torch_dtype=torch.bfloat16)
    xt = torch.tensor(x, dtype=torch.float32)
    pred = np.zeros((n, H, C), dtype=np.float16)
    per_call = 4096
    bw = max(1, per_call // C)
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, n, bw):
            oo = origins[s:s + bw]
            ctxs = torch.stack([xt[max(0, o - ctx_len):o] for o in oo])          # [b, ctx, C]
            b = ctxs.shape[0]
            flat = ctxs.permute(0, 2, 1).reshape(b * C, -1)
            f = forecast(pipe, flat, H)                                           # [b*C, H]
            pred[s:s + b] = f.reshape(b, C, H).permute(0, 2, 1).numpy().astype(np.float16)
    gt = np.stack([y[L + i:L + i + H] for i in range(n)])
    err = pred.astype(np.float32) - gt
    print('FM %s %s H=%d ctx=%d n=%d C=%d mse=%.6f mae=%.6f (%.0fs)' % (
        name, model_id, H, ctx_len, n, C, (err ** 2).mean(), np.abs(err).mean(), time.time() - t0), flush=True)
    np.savez(out, y=y, pred=pred, L=L, H=H)


if __name__ == '__main__':
    main()
