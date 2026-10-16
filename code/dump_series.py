"""Export the standardized series and split indices of a dataset exactly as the TAFAS/PETSA loader builds them.

Usage: python dump_series.py <dataset> [<dataset> ...]  -> dumps/series_<dataset>.npz
Fields: x (T_total x C standardized), test_start (index of the first test lookback step), L, stamp
"""
import sys

import numpy as np

from config import get_cfg_defaults
from datasets.build import update_cfg_from_dataset
from datasets.loader import get_test_dataloader

for name in sys.argv[1:]:
    cfg = get_cfg_defaults()
    cfg.merge_from_list(['DATA.NAME', name, 'DATA_LOADER.NUM_WORKERS', '0'])
    update_cfg_from_dataset(cfg, name)
    ds = get_test_dataloader(cfg).dataset
    L = cfg.DATA.SEQ_LEN
    tr, va, te = np.asarray(ds.train), np.asarray(ds.val), np.asarray(ds.test)
    x = np.concatenate([tr, va[L:], te[L:]]).astype(np.float32)
    test_start = len(tr) + len(va) - L - L      # index in x of the first test lookback step
    assert np.allclose(x[test_start:test_start + len(te)], te, atol=1e-5)
    st = np.concatenate([np.asarray(ds.train_stamp), np.asarray(ds.val_stamp)[L:], np.asarray(ds.test_stamp)[L:]])
    np.savez('dumps/series_%s.npz' % name, x=x, test_start=test_start, L=L, stamp=st,
             n_train=len(tr), n_val=len(va) - L)
    print(name, x.shape, 'test_start', test_start, 'test len', len(te))
