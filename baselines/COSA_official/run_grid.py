"""Benchmark grid on the official COSA harness (TAFAS / PETSA / COSA with the authors' script settings).

Usage: python run_grid.py <out.csv> <models,comma> <datasets,comma> <horizons,comma> [seed]
Per setting: train (official recipe, if no checkpoint), dump frozen predictions, run TAFAS, PETSA, COSA.
Rows: model,dataset,H,seed,method,mse,mae,seconds
"""
import os
import re
import subprocess
import sys
import time

DATA = '../../data/'
DUMPS = '../../code/dumps/grid/'
METHODS = {
    'TAFAS': ['RESULT_DIR', './results/TAFAS/', 'TTA.SOLVER.BASE_LR', '0.001', 'TTA.SOLVER.WEIGHT_DECAY', '0.0001',
              'TTA.TAFAS.GATING_INIT', '0.01'],
    'PETSA': ['RESULT_DIR', './results/PETSA/', 'TTA.SOLVER.BASE_LR', '0.001', 'TTA.SOLVER.WEIGHT_DECAY', '0.0001',
              'TTA.PETSA.GATING_INIT', '0.01', 'TTA.PETSA.RANK', '16', 'TTA.PETSA.LOSS_ALPHA', '0.1'],
    'COSA': ['RESULT_DIR', './results/COSA/', 'TTA.SOLVER.BASE_LR', '0.001', 'TTA.SOLVER.WEIGHT_DECAY', '0.0001',
             'TTA.COSA.BATCH_SIZE', '48', 'TTA.COSA.STEPS', '3', 'TTA.COSA.BUFFER_CONTEXT_SIZE', '10',
             'TTA.COSA.ADAPTER_TYPE', 'Linear', 'TTA.COSA.FAST_ADAPTATION', 'True',
             'TTA.COSA.PER_BATCH_LR_RESET', 'True', 'TTA.COSA.ADAPTIVE_LR', 'True', 'TTA.COSA.PAAS', 'True',
             'TTA.COSA.PERIOD_N', '1', 'TTA.COSA.POGT', 'True', 'TTA.COSA.PARTIAL_ADAPT', 'True',
             'TTA.COSA.ADJUST_PRED', 'True', 'TTA.COSA.MAX_LR', '0.005', 'TTA.COSA.MIN_LR', '0.0001'],
}


METHODS_NOADJ = {k + '-noadj': v + ['TTA.%s.ADJUST_PRED' % k, 'False'] for k, v in METHODS.items()}
# delta-Adapter (official Adapter-X+Y procedure, tta/dadapter.py) has no revision step, so no -noadj variant
METHODS['DADAPTER'] = ['RESULT_DIR', './results/DADAPTER/']
# adaptation only: every forecast is emitted before the batch's revealed targets are used (COSA needs an
# extra switch because its official loop otherwise emits the output of the partially adapted adapter)
ENV_NOADJ = {'COSA-noadj': {'COSA_EMIT_ORIGINAL': '1'}}


def py(args, env=None):
    e = dict(os.environ)
    e.update(env or {})
    p = subprocess.run([sys.executable] + args, capture_output=True, text=True, encoding='utf-8',
                       errors='ignore', env=e)
    return p.stdout + '\n' + p.stderr


def grab(out, key):
    m = re.findall(r'"%s":\s*([0-9.eE+-]+)' % key, out)
    return float(m[-1]) if m else None


def main():
    out_csv, models, dsets, hs = sys.argv[1], sys.argv[2].split(','), sys.argv[3].split(','), sys.argv[4].split(',')
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    # modes: '' (train if needed, dump, all methods), 'nodad' (same without delta-Adapter), 'noadj' (adaptation-only
    # variants with dumps), 'rerun' (official adapters again, e.g. for MAE), 'only:A+B' (just these methods)
    mode = sys.argv[6] if len(sys.argv) > 6 else ''
    if mode == 'noadj':
        methods = METHODS_NOADJ
    elif mode in ('rerun', 'nodad'):
        methods = {k: v for k, v in METHODS.items() if k != 'DADAPTER'}
    elif mode.startswith('only:'):
        methods = {k: METHODS[k] for k in mode[5:].split('+')}
    else:
        methods = METHODS
    full = mode in ('', 'nodad')           # train missing checkpoints and dump frozen forecasts
    os.makedirs(DUMPS, exist_ok=True)
    done = set()
    if os.path.exists(out_csv):
        done = {tuple(l.split(',')[:5]) for l in open(out_csv) if len(l.split(',')) >= 6 and l.split(',')[5]}
    for H in hs:
        for ds in dsets:
            for m in models:
                sfx = '' if seed == 0 else '_s%d' % seed
                ck = './checkpoints/%s/%s_%s%s/' % (m, ds, H, sfx)
                base = ['main.py', 'DATA.BASE_DIR', DATA, 'DATA.NAME', ds, 'DATA.PRED_LEN', H, 'MODEL.NAME', m,
                        'MODEL.pred_len', H, 'TRAIN.CHECKPOINT_DIR', ck, 'DATA_LOADER.NUM_WORKERS', '0',
                        'SEED', str(seed)]
                key = (m, ds, H, str(seed))
                t0 = time.time()
                if not full and not os.path.exists(ck + 'checkpoint_best.pth'):
                    continue                       # adaptation-only pass never trains
                if not full and not os.path.exists(DUMPS + '%s_%s_%s%s.npz' % (m, ds, H, sfx)):
                    continue                       # checkpoint may still be training: wait for its dump
                if not os.path.exists(ck + 'checkpoint_best.pth'):
                    o = py(base + ['TRAIN.ENABLE', 'True', 'TTA.ENABLE', 'False', 'TEST.ENABLE', 'False'])
                    if not os.path.exists(ck + 'checkpoint_best.pth'):
                        print('TRAIN FAILED', key, o[-500:], flush=True)
                        continue
                    print('trained', key, '%.0fs' % (time.time() - t0), flush=True)
                rows = []
                if key + ('frozen',) not in done and full:
                    t0 = time.time()
                    npz = DUMPS + '%s_%s_%s%s.npz' % (m, ds, H, sfx)
                    o = py(['dump_preds.py', ck, ds, m, H, npz])
                    mm = re.search(r'FROZEN .* mse=([0-9.]+) mae=([0-9.]+)', o)
                    rows.append(('frozen', mm.group(1) if mm else '', mm.group(2) if mm else '', time.time() - t0))
                    if not mm:
                        print('DUMP FAILED', key, o[-500:], flush=True)
                for meth, args in methods.items():
                    if key + (meth,) in done:
                        continue
                    t0 = time.time()
                    env = dict(ENV_NOADJ.get(meth, {}))
                    if mode == 'noadj' or meth == 'DADAPTER':
                        os.makedirs(DUMPS + 'tta/', exist_ok=True)
                        env['TTA_DUMP_PRED'] = DUMPS + 'tta/%s_%s_%s_%s%s.npz' % (meth, m, ds, H, sfx)
                    o = py(base + ['TRAIN.ENABLE', 'False', 'TEST.ENABLE', 'False', 'TTA.ENABLE', 'True'] + args, env)
                    mse, mae = grab(o, 'test_mse'), grab(o, 'test_mae')
                    if mse is None:
                        mm = re.search(r'Test MSE: ([0-9.]+), Test MAE: ([0-9.]+)', o)
                        if mm:
                            mse, mae = float(mm.group(1)), float(mm.group(2))
                    if mse is None:
                        print('NO RESULT', key, meth, o[-400:].replace('\n', ' '), flush=True)
                    rows.append((meth, '' if mse is None else '%.6f' % mse, '' if mae is None else '%.6f' % mae,
                                 time.time() - t0))
                with open(out_csv, 'a') as f:
                    for meth, mse, mae, dt in rows:
                        f.write('%s,%s,%s,%s,%s,%s,%s,%.0f\n' % (m, ds, H, seed, meth, mse, mae, dt))
                print(key, ' '.join('%s=%s' % (r[0], r[1][:6]) for r in rows), flush=True)


if __name__ == '__main__':
    main()
