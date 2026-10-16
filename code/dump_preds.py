"""Dump frozen-backbone predictions for every test window (stride 1) to an .npz.

Usage: python dump_preds.py <ckpt_dir> <dataset> <model> <H> <out.npz> [norm 1|0] [extra cfg opts...]
Saves y (T_test x C, standardized), pred (n x H x C, float16), origin offset L.
"""
import os
import sys

import numpy as np
import torch

from config import get_cfg_defaults
from datasets.build import update_cfg_from_dataset
from datasets.loader import get_test_dataloader
from models.build import build_model, load_best_model
from models.forecast import forecast

CYCLE = {'ETTh1': 24, 'ETTh2': 24, 'ETTm1': 96, 'ETTm2': 96, 'weather': 144, 'electricity': 168,
         'traffic': 168, 'PEMS03': 288, 'PEMS04': 288, 'PEMS07': 288, 'PEMS08': 288, 'Solar': 144,
         'exchange_rate': 1}


def main():
    ckpt, ds_name, model_name, H, out = sys.argv[1:6]
    norm = int(sys.argv[6]) if len(sys.argv) > 6 else 1
    extra = sys.argv[7:]
    H = int(H)
    if not norm:
        os.environ['NONORM'] = '1'
    else:
        os.environ.pop('NONORM', None)
    cfg = get_cfg_defaults()
    cfg.merge_from_list(['DATA.NAME', ds_name, 'DATA.PRED_LEN', str(H), 'MODEL.NAME', model_name,
                         'MODEL.pred_len', str(H), 'MODEL.cycle', str(CYCLE[ds_name]),
                         'TRAIN.CHECKPOINT_DIR', ckpt, 'DATA_LOADER.NUM_WORKERS', '0'] + extra)
    update_cfg_from_dataset(cfg, ds_name)
    model = build_model(cfg)
    model = load_best_model(cfg, model).cuda().eval()
    ds = get_test_dataloader(cfg).dataset
    test = torch.tensor(np.asarray(ds.test), dtype=torch.float32).cuda()
    stamp = torch.tensor(np.asarray(ds.test_stamp), dtype=torch.float32).cuda()
    L, LL = cfg.DATA.SEQ_LEN, cfg.DATA.LABEL_LEN
    T = test.shape[0]
    origins = list(range(L, T - H + 1))
    preds = []
    bs = max(1, int(2e7 // (H * test.shape[1] + L * test.shape[1])))
    bs = min(bs, 1024)
    with torch.no_grad():
        for s in range(0, len(origins), bs):
            oo = origins[s:s + bs]
            enc = torch.stack([test[o - L:o] for o in oo])
            enc_s = torch.stack([stamp[o - L:o] for o in oo])
            dec = torch.stack([test[o - LL:o + H] for o in oo])
            dec_s = torch.stack([stamp[o - LL:o + H] for o in oo])
            p, _ = forecast(cfg, (enc, enc_s, dec, dec_s), model, None)
            preds.append(p.half().cpu().numpy())
    pred = np.concatenate(preds)
    y = test.cpu().numpy()
    gt = np.stack([y[o:o + H] for o in origins])
    print('%s %s H=%d n=%d C=%d frozen MSE=%.4f MAE=%.4f' % (
        ds_name, model_name, H, len(origins), y.shape[1], ((pred.astype(np.float32) - gt) ** 2).mean(),
        np.abs(pred.astype(np.float32) - gt).mean()))
    np.savez(out, y=y, pred=pred, L=L, H=H)


if __name__ == '__main__':
    main()
