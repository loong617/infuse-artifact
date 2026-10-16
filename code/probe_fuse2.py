import sys

import inflight as I

for path in sys.argv[1:]:
    y, pred, gt, E, L, H = I.load(path)
    B = I.tafas_batches(y, pred.shape[0], L)
    fr = float((E ** 2).mean())
    out = []
    for mw in (2 * H, 32):
        for m in ('refresh', 'update', 'fuse_eq', 'fuse_bg', 'fuse'):
            if mw == 2 * H and m in ('fuse_eq', 'fuse_bg'):
                continue
            v = I.tafas(y, pred, gt, E, L, H, mode=m, batches=B, min_windows=mw)[0]
            out.append('%s/%s %+.2f%%' % (m, 'mw2H' if mw == 2 * H else 'mw32', 100 * (v - fr) / fr))
    print(path.split('/')[-1], ' | '.join(out), flush=True)
