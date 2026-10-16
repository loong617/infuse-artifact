"""Training-free revision rules under the TAFAS protocol, as baselines for INFUSE-U (no model call):
last (carry the latest revealed error forward), mean (shift by the mean revealed error), ar1 (AR(1) error decay
with the lag-1 coefficient of the pooled error covariance). Finished dumps are skipped.

Usage: python eval_revise.py <out.jsonl> <glob> [<glob> ...] [--workers N]
"""
import glob
import json
import os
import sys
import time
from multiprocessing import Pool

import inflight as I

RULES = ('last', 'mean', 'ar1')


def evaluate(path):
    t0 = time.time()
    y, pred, gt, E, L, H = I.load(path)
    B = I.tafas_batches(y, pred.shape[0], L)
    rec = {'path': path.replace('\\', '/')}
    for m in RULES:
        rec[m] = list(I.tafas(y, pred, gt, E, L, H, mode=m, batches=B))
    rec['seconds'] = time.time() - t0
    return rec


def main():
    args = [a for a in sys.argv[1:]]
    workers = 6
    if '--workers' in args:
        i = args.index('--workers')
        workers = int(args[i + 1])
        del args[i:i + 2]
    out, pats = args[0], args[1:]
    done = {json.loads(l)['path'] for l in open(out)} if os.path.exists(out) else set()
    todo = sorted({p for pat in pats for p in glob.glob(pat) if p.replace('\\', '/') not in done},
                  key=os.path.getsize)
    with Pool(workers) as pool:
        for rec in pool.imap_unordered(evaluate, todo):
            with open(out, 'a') as f:
                f.write(json.dumps(rec) + '\n')
            print('%-48s %s (%.0fs)' % (os.path.basename(rec['path']),
                                       ' '.join('%s %.4f' % (m, rec[m][0]) for m in RULES), rec['seconds']), flush=True)


if __name__ == '__main__':
    main()
