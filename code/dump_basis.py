# -*- coding: utf-8 -*-
"""Dump the model-in-the-loop affine basis for ONE adaptation (Fig 3b landscape).

For electricity/seed2024/channel 117, chunk 100: run the three basis forward
passes (cycle = Q0 / zeros / ones) over the 15 estimation windows and reduce to
the six sufficient statistics of the per-channel window objective
    L(a,b) = Srr - 2 a Sgr - 2 b Shr + a^2 Sgg + b^2 Shh + 2 a b Sgh
so the exact quadratic surface can be rebuilt offline (Lemma 1 makes it exact).
"""
import sys

import numpy as np
import torch

sys.path.insert(0, '.')
from utils.parser import parse_args, load_config  # noqa: E402
from datasets.build import update_cfg_from_dataset  # noqa: E402
from datasets.loader import get_test_dataloader  # noqa: E402
from models.build import build_model, load_best_model  # noqa: E402
from models.forecast import forecast  # noqa: E402
from utils.misc import set_seeds, set_devices  # noqa: E402

DS, CYC, SEED, CH, CHUNK = 'electricity', 168, 2024, 117, 100
PERIOD = 24          # FFT cadence for electricity (chunk = period + 1 = 25 windows)
HIST = 1440          # frozen estimation history (15 windows of 96)

sys.argv = ['x', 'DATA.NAME', DS, 'DATA.PRED_LEN', '96', 'MODEL.NAME', 'CycleNet',
            'MODEL.pred_len', '96', 'MODEL.cycle', str(CYC), 'TRAIN.ENABLE', 'False',
            'TRAIN.CHECKPOINT_DIR', './checkpoints/CycleNet/%s_96_nonorm_s%s/' % (DS, SEED),
            'NORM_MODULE.ENABLE', 'False', 'DATA_LOADER.NUM_WORKERS', '0', 'SEED', str(SEED)]
args = parse_args()
cfg = load_config(args)
update_cfg_from_dataset(cfg, cfg.DATA.NAME)
set_devices(cfg.VISIBLE_DEVICES)
set_seeds(cfg.SEED)
model = build_model(cfg)
model = load_best_model(cfg, model)
model.eval()

ds = get_test_dataloader(cfg).dataset
test = torch.tensor(np.asarray(ds.test), dtype=torch.float32).cuda()
stamp = torch.tensor(np.asarray(ds.test_stamp), dtype=torch.float32).cuda()
L, LAB, H = cfg.DATA.SEQ_LEN, cfg.DATA.LABEL_LEN, cfg.DATA.PRED_LEN
Q0 = model.cycleQueue.data.clone()

origin = CHUNK * (PERIOD + 1) + L
ows = [origin - H * (t + 1) for t in range(HIST // H)]
ows = [o for o in ows if o - L >= 0]
print('chunk %d origin %d, %d estimation windows (%d..%d)' %
      (CHUNK, origin, len(ows), ows[-1], ows[0]))


def make_inputs(origins):
    enc = torch.stack([test[o - L:o] for o in origins])
    enc_s = torch.stack([stamp[o - L:o] for o in origins])
    dec = torch.stack([test[o - LAB:o + H] for o in origins])
    dec_s = torch.stack([stamp[o - LAB:o + H] for o in origins])
    return (enc, enc_s, dec, dec_s)


inp = make_inputs(ows)
y = torch.stack([test[o:o + H] for o in ows])          # [n, H, C] ground truth
with torch.no_grad():
    model.cycleQueue.data.copy_(torch.zeros_like(Q0))
    f0 = forecast(cfg, inp, model, None)[0]
    model.cycleQueue.data.copy_(Q0)
    fq = forecast(cfg, inp, model, None)[0]
    model.cycleQueue.data.copy_(torch.ones_like(Q0))
    f1 = forecast(cfg, inp, model, None)[0]
model.cycleQueue.data.copy_(Q0)

g = (fq - f0)[:, :, CH].double()
h = (f1 - f0)[:, :, CH].double()
r = (y - f0)[:, :, CH].double()
S = dict(Sgg=(g * g).sum().item(), Shh=(h * h).sum().item(), Sgh=(g * h).sum().item(),
         Sgr=(g * r).sum().item(), Shr=(h * r).sum().item(), Srr=(r * r).sum().item())
n = g.numel()
Q = np.array([[S['Sgg'], S['Sgh']], [S['Sgh'], S['Shh']]])
c = np.array([S['Sgr'], S['Shr']])
theta = np.linalg.solve(Q, c)
L_id = (S['Srr'] - 2 * S['Sgr'] - 2 * S['Shr'] + S['Sgg'] + S['Shh'] + 2 * S['Sgh']) / n
L_opt = (S['Srr'] - c @ theta) / n
ev = np.linalg.eigvalsh(Q)
print('closed-form optimum: alpha=%.3f beta=%.3f' % (theta[0], theta[1]))
print('window MSE: identity %.4f -> optimum %.4f (%.1f%% reduction)' %
      (L_id, L_opt, 100 * (1 - L_opt / L_id)))
print('Q eigenvalues: %.3e / %.3e (cond %.1f)' % (ev[0], ev[1], ev[1] / ev[0]))

A = np.loadtxt('alpha_traj_electricity.csv', delimiter=',')
print('actual Hedge estimate at this chunk: alpha=%.3f (chunk %d, ch %d)' %
      (A[CHUNK, CH], CHUNK, CH))

np.savez('basis_ch117_elec.npz', chunk=CHUNK, ch=CH, n=n, **S,
         alpha_opt=theta[0], beta_opt=theta[1], L_identity=L_id, L_optimum=L_opt,
         eig=ev, alpha_hat=A[CHUNK, CH])
print('saved basis_ch117_elec.npz')
