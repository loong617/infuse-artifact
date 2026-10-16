"""Per-batch cost under the TTA protocol, measured with the GPU otherwise idle.

Adapters: their own timers (processing time per batch excluding scoring); frozen forward per batch from COSA's
base-prediction timer. INFUSE: CPU time of the update (single-threaded NumPy) on the frozen forecasts.
Writes ../../PETSA/timing.json with rows {setting, method, ms}.
"""
import json
import os
import re
import subprocess
import sys
import time

DATA = '../../data/'
PETSA = '../../code/'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_grid import METHODS  # noqa: E402

SETTINGS = [('iTransformer', 'ETTh1', '96'), ('iTransformer', 'weather', '96'), ('PatchTST', 'ETTm1', '96')]


def run(args):
    p = subprocess.run([sys.executable, 'main.py'] + args, capture_output=True, text=True, encoding='utf-8',
                       errors='ignore')
    return p.stdout


def blocks(out):
    """All JSON objects printed by the adapter, merged."""
    found = {}
    for m in re.finditer(r'"([a-z_]+_ms|total_time_seconds|full_adaptation_count|partial_adaptation_count|'
                         r'full_gt_adaptation_count|partial_gt_adaptation_count|adaptation_count)":\s*([0-9.]+)', out):
        found.setdefault(m.group(1), []).append(float(m.group(2)))
    return found


def main():
    rows = []
    for m, ds, H in SETTINGS:
        tag = '%s %s' % ({'iTransformer': 'iTrans.', 'PatchTST': 'PatchTST'}[m], {'weather': 'Weather'}.get(ds, ds))
        base = ['DATA.BASE_DIR', DATA, 'DATA.NAME', ds, 'DATA.PRED_LEN', H, 'MODEL.NAME', m, 'MODEL.pred_len', H,
                'TRAIN.CHECKPOINT_DIR', './checkpoints/%s/%s_%s/' % (m, ds, H), 'DATA_LOADER.NUM_WORKERS', '0',
                'TRAIN.ENABLE', 'False', 'TEST.ENABLE', 'False', 'TTA.ENABLE', 'True']
        env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        nb = int(subprocess.run([sys.executable, '-c',
                                 "import sys; sys.path.insert(0, r'%s'); import inflight as I; "
                                 "y,p,g,E,L,H=I.load(r'%sdumps/grid/%s_%s_%s.npz'); print(len(I.tafas_batches(y,p.shape[0],L)))"
                                 % (PETSA, PETSA, m, ds, H)], capture_output=True, text=True, env=env).stdout.split()[0])
        for meth in ('COSA', 'TAFAS', 'PETSA'):
            # wall time of the whole adaptation loop (forward, backward, revision, scoring) per batch; GPU work is
            # asynchronous, so component timers without synchronization are not comparable across methods
            out = run(base + METHODS[meth])
            b = blocks(out)
            ms = 1000.0 * b.get('total_time_seconds', [0])[0] / nb
            rows.append({'setting': tag, 'method': meth, 'ms': ms, 'raw': {k: v[:3] for k, v in b.items()}})
            print(tag, meth, '%.2f ms/batch (%d batches)' % (ms, nb), flush=True)
        code = ("import sys,time; sys.path.insert(0, r'%s'); import inflight as I; "
                "y,p,g,E,L,H=I.load(r'%sdumps/grid/%s_%s_%s.npz'); B=I.tafas_batches(y,p.shape[0],L); t={}\n"
                "for md in ('frozen','update','fuse'):\n"
                "    t0=time.perf_counter(); I.tafas(y,p,g,E,L,H,mode=md,batches=B); t[md]=time.perf_counter()-t0\n"
                "print('%%.4f %%.4f' %% ((t['update']-t['frozen'])/len(B)*1e3, (t['fuse']-t['frozen'])/len(B)*1e3))"
                % (PETSA, PETSA, m, ds, H))
        o = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=env).stdout.split()
        rows.append({'setting': tag, 'method': 'INFUSE-U', 'ms': float(o[0])})
        rows.append({'setting': tag, 'method': 'INFUSE', 'ms': float(o[1])})
        print(tag, 'INFUSE-U %s ms, INFUSE %s ms' % (o[0], o[1]), flush=True)
    json.dump({'rows': rows}, open(PETSA + 'timing.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
