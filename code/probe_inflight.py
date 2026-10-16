import sys
import time

import numpy as np

import inflight as I

for path in sys.argv[1:]:
    y, pred, gt, E, L, H = I.load(path)
    t0 = time.time()
    B = I.tafas_batches(y, pred.shape[0], L)
    res = {m: I.tafas(y, pred, gt, E, L, H, mode=m, batches=B) for m in ('frozen', 'refresh', 'update', 'fuse')}
    fr = res['frozen'][0]
    pr, _ = I.freshness_prediction(E, L, H, y)
    print('== %s (%.0fs)' % (path, time.time() - t0))
    print('   ' + '  '.join('%s %.4f (%+.1f%%)' % (m, v[0], 100 * (v[0] - fr) / fr) for m, v in res.items()))
    print('   refresh predicted from lead curve: %.4f (%+.1f%%)' % (pr, 100 * (pr - fr) / fr))
    for r in (1, 4, 12, 24, 48):
        if 47 + r - 1 > H:
            continue
        h, nc = I.service(E, pred, gt, H, r, 48, mode='hold')
        u, _ = I.service(E, pred, gt, H, r, 48, mode='update')
        print('   service r=%2d calls=%5d  hold %.4f  update %.4f' % (r, nc, h, u))
