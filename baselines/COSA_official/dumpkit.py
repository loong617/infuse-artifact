"""Optional dump of the predictions an adapter emits (env TTA_DUMP_PRED=<path.npz>); no effect otherwise."""
import os

import numpy as np

_PATH = os.environ.get('TTA_DUMP_PRED')
_PREDS = []


def add(pred):
    if _PATH:
        _PREDS.append(pred.detach().float().cpu().numpy().astype(np.float16))


def save():
    if _PATH and _PREDS:
        np.savez(_PATH, pred=np.concatenate(_PREDS))
