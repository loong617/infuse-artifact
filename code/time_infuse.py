"""Per-batch CPU time of INFUSE-U / INFUSE (single-threaded NumPy) under the TAFAS protocol.
Run with OMP/MKL threads = 1. Usage: python time_infuse.py <npz> [...]"""
import sys
import time

import inflight as I

for path in sys.argv[1:]:
    y, pred, gt, E, L, H = I.load(path)
    B = I.tafas_batches(y, pred.shape[0], L)
    t = {}
    for m in ('frozen', 'update', 'fuse'):
        t0 = time.perf_counter()
        I.tafas(y, pred, gt, E, L, H, mode=m, batches=B)
        t[m] = time.perf_counter() - t0
    nb = len(B)
    print('%s batches=%d  update %.2f ms/batch  fuse %.2f ms/batch (loop overhead %.2f ms/batch removed)' % (
        path.split('/')[-1], nb, 1e3 * (t['update'] - t['frozen']) / nb, 1e3 * (t['fuse'] - t['frozen']) / nb,
        1e3 * t['frozen'] / nb))
