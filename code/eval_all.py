"""Evaluate every dump: frozen / refresh / INFUSE-U / INFUSE (+ Bates-Granger fusion) under the TAFAS protocol,
the freshness prediction (Prop. 1), the lead-wise curves, and the serving frontier.
Optionally the update ablations. Results go to a JSON-lines file keyed by dump path (finished dumps are skipped).

Usage: python eval_all.py <out.jsonl> <glob> [--service] [--ablate] [--workers N]
"""
import glob
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

import inflight as I

R_LIST = (1, 2, 4, 8, 12, 24, 48)
HS = 48


def evaluate(args):
    path, service, ablate = args
    t0 = time.time()
    y, pred, gt, E, L, H = I.load(path)
    n, _, C = pred.shape
    B = I.tafas_batches(y, n, L)
    rec = {'path': path.replace('\\', '/'), 'n': n, 'C': C, 'H': H, 'periods': sorted({int(p) for _, _, p in B})}
    for m in ('frozen', 'refresh', 'update', 'fuse', 'fuse_eq', 'fuse_bg'):
        mse, mae, lead = I.tafas(y, pred, gt, E, L, H, mode=m, batches=B, return_lead=True)
        rec[m] = [mse, mae]
        rec[m + '_lead'] = [float(v) for v in lead]
    rec['fresh_pred'] = I.freshness_prediction(E, L, H, y)[0]
    rec['lead_curve'] = [float(v) for v in (E ** 2).mean((0, 2))]
    if ablate:
        ab = {}
        ab['single_u'] = I.tafas(y, pred, gt, E, L, H, mode='single_u', batches=B)[0]
        ab['single_f'] = I.tafas(y, pred, gt, E, L, H, mode='single_f', batches=B)[0]
        ab['no_shrink'] = I.tafas(y, pred, gt, E, L, H, mode='update', batches=B, shrink=0.0, cone=False)[0]
        ab['shrink_0.3'] = I.tafas(y, pred, gt, E, L, H, mode='update', batches=B, shrink=0.3, cone=False)[0]
        ab['forget_0.99'] = I.tafas(y, pred, gt, E, L, H, mode='update', batches=B, rho=0.99, cone=False)[0]
        if H == 96:
            ab['per_channel'] = I.tafas(y, pred, gt, E, L, H, mode='update', batches=B, per_channel=True)[0]
        rec['ablate'] = ab
    if service:
        rec['service'] = {}
        for r in R_LIST:
            if HS + r - 1 > H:
                continue
            h, nc = I.service(E, pred, gt, H, r, HS, mode='hold')
            u, _ = I.service(E, pred, gt, H, r, HS, mode='update')
            f, _ = I.service(E, pred, gt, H, r, HS, mode='fuse')
            rec['service'][str(r)] = [h, u, nc, f]
    rec['seconds'] = time.time() - t0
    return rec


def main():
    out, pat = sys.argv[1], sys.argv[2]
    service, ablate = '--service' in sys.argv, '--ablate' in sys.argv
    workers = int(sys.argv[sys.argv.index('--workers') + 1]) if '--workers' in sys.argv else 6
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)['path'] for l in open(out)}
    todo = [p for p in sorted(glob.glob(pat)) if p.replace('\\', '/') not in done]
    todo.sort(key=lambda p: os.path.getsize(p))
    with Pool(workers) as pool:
        for rec in pool.imap_unordered(evaluate, [(p, service, ablate) for p in todo]):
            with open(out, 'a') as f:
                f.write(json.dumps(rec) + '\n')
            fr = rec['frozen'][0]
            print('%-48s frozen %.4f refresh %+.1f%% upd %+.1f%% fuse %+.1f%% bg %+.1f%% (%.0fs)' % (
                os.path.basename(rec['path']), fr,
                *(100 * (rec[m][0] - fr) / fr for m in ('refresh', 'update', 'fuse', 'fuse_bg')), rec['seconds']),
                flush=True)


if __name__ == '__main__':
    main()
