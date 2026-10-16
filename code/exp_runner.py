"""Job runner for the WWW rebuild: train/evaluate backbones and TTA methods, append results to a CSV.

Usage: python exp_runner.py <jobfile.json> <out.csv>
A job: {"bb": "CycleNet", "ds": "ETTh2", "H": 96, "seed": 2024, "norm": 1, "method": "frozen"|"TAFAS"|...,
        "env": {...}, "tag": "..."}
"""
import json
import os
import re
import subprocess
import sys
import time

CYCLE = {'ETTh1': 24, 'ETTh2': 24, 'ETTm1': 96, 'ETTm2': 96, 'weather': 144, 'electricity': 168,
         'traffic': 168, 'PEMS03': 288, 'PEMS04': 288, 'PEMS07': 288, 'PEMS08': 288, 'Solar': 144,
         'exchange_rate': 1}
RESULT_DIR = {'TAFAS': 'TAFAS', 'PETSA': 'PETSA', 'COSA': 'COSA', 'DAdapter': 'DAdapter', 'LoRA': 'LoRA',
              'RevIN': 'RevIN', 'Proceed': 'ProceedTTA', 'ReCycle': 'Recycle'}
METHOD_ARGS = {'TAFAS': ['TTA.SOLVER.BASE_LR', '0.001', 'TTA.SOLVER.WEIGHT_DECAY', '0.0001',
                         'TTA.TAFAS.GATING_INIT', '0.01'],
               'PETSA': ['TTA.SOLVER.BASE_LR', '0.005', 'TTA.SOLVER.WEIGHT_DECAY', '0.0001']}
FLOAT = r'([0-9]+\.[0-9]+)'


def ckpt_dir(job):
    suf = '' if job['norm'] else '_nonorm'
    return './checkpoints/%s/%s_%d%s_s%d/' % (job['bb'], job['ds'], job['H'], suf, job['seed'])


def common(job):
    return ['DATA.NAME', job['ds'], 'DATA.PRED_LEN', str(job['H']), 'MODEL.NAME', job['bb'],
            'MODEL.pred_len', str(job['H']), 'MODEL.cycle', str(CYCLE[job['ds']]),
            'TRAIN.CHECKPOINT_DIR', ckpt_dir(job), 'NORM_MODULE.ENABLE', 'False',
            'DATA_LOADER.NUM_WORKERS', '0', 'SEED', str(job['seed'])] + job.get('extra', [])


def run(args, env):
    p = subprocess.run([sys.executable, 'main.py'] + args, env=env, capture_output=True, text=True,
                       encoding='utf-8', errors='ignore')
    return p.stdout + p.stderr


def do_job(job):
    env = dict(os.environ)
    env.pop('NONORM', None)
    if not job['norm']:
        env['NONORM'] = '1'
    env.update({k: str(v) for k, v in job.get('env', {}).items()})
    c = common(job)
    if not os.path.exists(ckpt_dir(job) + 'checkpoint_best.pth'):
        t0 = time.time()
        out = run(c + ['TRAIN.ENABLE', 'True', 'TTA.ENABLE', 'False', 'TEST.ENABLE', 'False'], env)
        if not os.path.exists(ckpt_dir(job) + 'checkpoint_best.pth'):
            return None, None, 'TRAIN_FAILED ' + out[-400:]
        print('  trained %s in %.0fs' % (ckpt_dir(job), time.time() - t0), flush=True)
    m = job['method']
    if m == 'frozen':
        out = run(c + ['TRAIN.ENABLE', 'False', 'TTA.ENABLE', 'False', 'TEST.ENABLE', 'True',
                       'TEST.BATCH_SIZE', '20000', 'TEST.DROP_LAST', 'False'], env)
        mm = re.search(r'test_mse: ' + FLOAT + r', test_mae: ' + FLOAT, out)
    else:
        rd = './results/%s_%s/' % (RESULT_DIR[m], job.get('tag', 'x'))
        out = run(c + ['TRAIN.ENABLE', 'False', 'TTA.ENABLE', 'True', 'TEST.ENABLE', 'False',
                       'RESULT_DIR', rd] + METHOD_ARGS.get(m, []), env)
        mm = re.search(r'Test MSE: ' + FLOAT + r', Test MAE: ' + FLOAT, out)
    if not mm:
        return None, None, 'NO_RESULT ' + out[-600:]
    return float(mm.group(1)), float(mm.group(2)), ''


def main():
    jobs = json.load(open(sys.argv[1]))
    out_csv = sys.argv[2]
    done = set()
    if os.path.exists(out_csv):
        for line in open(out_csv):
            done.add(line.split(',')[0])
    for job in jobs:
        key = '|'.join(str(job.get(k, '')) for k in ('bb', 'ds', 'H', 'seed', 'norm', 'method', 'tag'))
        if key in done:
            continue
        t0 = time.time()
        mse, mae, err = do_job(job)
        dt = time.time() - t0
        with open(out_csv, 'a') as f:
            f.write('%s,%s,%s,%.0f,%s\n' % (key, '' if mse is None else '%.4f' % mse,
                                         '' if mae is None else '%.4f' % mae, dt,
                                         err.replace('\n', ' ').replace(',', ';')[:300]))
        print('%-60s mse=%s mae=%s (%.0fs) %s' % (key, mse, mae, dt, err[:120]), flush=True)


if __name__ == '__main__':
    main()
