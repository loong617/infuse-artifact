"""Dump frozen-backbone predictions for every test window (stride 1).

Usage: python dump_preds.py <ckpt_dir> <dataset> <model> <H> <out.npz> [extra cfg opts...]
Saves y (test series incl. the first lookback, standardized), pred (n x H x C, float16), L, H.
"""
import sys

import numpy as np
import torch

from config import get_cfg_defaults
from datasets.build import update_cfg_from_dataset
from datasets.loader import get_test_dataloader
from models.build import build_model, load_best_model
from models.forecast import forecast

DATA = '../../data/'


def main():
    ckpt, ds_name, model_name, H, out = sys.argv[1:6]
    extra = sys.argv[6:]
    H = int(H)
    cfg = get_cfg_defaults()
    cfg.merge_from_list(['DATA.BASE_DIR', DATA, 'DATA.NAME', ds_name, 'DATA.PRED_LEN', str(H),
                         'MODEL.NAME', model_name, 'MODEL.pred_len', str(H), 'TRAIN.CHECKPOINT_DIR', ckpt,
                         'DATA_LOADER.NUM_WORKERS', '0'] + extra)
    update_cfg_from_dataset(cfg, ds_name)
    model = load_best_model(cfg, build_model(cfg)).cuda().eval()
    ds = get_test_dataloader(cfg).dataset
    test = torch.tensor(np.asarray(ds.test), dtype=torch.float32).cuda()
    stamp = torch.tensor(np.asarray(ds.test_stamp), dtype=torch.float32).cuda()
    L, LL = cfg.DATA.SEQ_LEN, cfg.DATA.LABEL_LEN
    T, C = test.shape
    origins = list(range(L, T - H + 1))
    bs = int(min(512, max(1, 3e7 // ((H + L) * C))))
    preds = []
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
    err = pred.astype(np.float32) - gt
    print('FROZEN %s %s H=%d n=%d C=%d mse=%.6f mae=%.6f' % (ds_name, model_name, H, len(origins), C,
                                                          (err ** 2).mean(), np.abs(err).mean()))
    np.savez(out, y=y, pred=pred, L=L, H=H)


if __name__ == '__main__':
    main()
