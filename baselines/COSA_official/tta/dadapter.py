"""delta-Adapter (Liang et al., ICLR 2026), additive Adapter-X+Y variant, ported from the official code
(baselines/AdapterOfficial/Adapter-X+Y/experiments/exp_online_xy_add.py: post_train, then test2 with
post_process=True, online=True, as in run_xy_add.py).

1. Offline: an input adapter (L*C -> d_model -> L*C) and an output adapter (H*C -> d_model -> H*C), both
   PostProcessingNet with tanh * delta, are trained on the training split around the frozen forecaster
   (Adam, lr 1e-4, batch 32, at most 10 episodes, lr halved after every episode from the second on, early
   stopping on the validation MSE with patience 3; delta = 0.1, d_model = 512).
2. Online: test windows are processed one at a time in order. After window i is scored, one Adam step
   (lr 1e-6, gradient-norm clipping 1e-4 per adapter) is taken on the most recent fully observed window, the one
   whose last target is the first target of window i (the official Buffer); the step therefore only affects
   later windows.
Interface changes: the official output adapter is constructed with input size L*C, which only runs for H = L;
we use H*C. Past windows are fed with their time stamps (the official code passes None).
"""
import copy
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import dumpkit
from datasets.loader import get_test_dataloader, get_train_dataloader, get_val_dataloader
from models.forecast import forecast
from utils.misc import prepare_inputs

D_MODEL, DELTA = 512, 0.1
POST_LR, EPISODES, PATIENCE, BATCH = 1e-4, 10, 3, 32
ONLINE_LR, CLIP = 1e-6, 1e-4
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


class PostProcessingNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, delta=0.1):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.bn1 = torch.nn.InstanceNorm1d(hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = torch.nn.InstanceNorm1d(hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)
        self.delta = delta

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.fc3(x)
        return torch.tanh(x) * self.delta


class DeltaAdapter:
    def __init__(self, cfg, model, norm_module=None):
        self.cfg, self.model, self.norm_module = cfg, model, norm_module
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        L, H, C = cfg.DATA.SEQ_LEN, cfg.DATA.PRED_LEN, cfg.DATA.N_VAR
        self.L, self.H = L, H
        self.post_x = PostProcessingNet(L * C, D_MODEL, L * C, DELTA).to(DEV)
        self.post_y = PostProcessingNet(H * C, D_MODEL, H * C, DELTA).to(DEV)
        self.train_ds = get_train_dataloader(cfg).dataset
        self.val_ds = get_val_dataloader(cfg).dataset
        self.test_ds = get_test_dataloader(cfg).dataset
        self.mse_all, self.mae_all = None, None
        self.times = {}

    def _predict(self, batch):
        enc, enc_s, dec, dec_s = prepare_inputs(batch)
        b = enc.shape[0]
        enc = enc + self.post_x(enc.reshape(b, -1)).view_as(enc)
        out, gt = forecast(self.cfg, (enc, enc_s, dec, dec_s), self.model, self.norm_module)
        out = out + self.post_y(out.reshape(b, -1)).view_as(out)
        return out, gt

    def _val_loss(self):
        losses = []
        with torch.no_grad():
            for batch in DataLoader(self.val_ds, batch_size=BATCH, shuffle=False, drop_last=True):
                out, gt = self._predict(batch)
                losses.append(F.mse_loss(out, gt).item())
        return float(np.mean(losses))

    def post_train(self):
        opt_x = torch.optim.Adam(self.post_x.parameters(), lr=POST_LR)
        opt_y = torch.optim.Adam(self.post_y.parameters(), lr=POST_LR)
        best, best_state, bad = np.inf, None, 0
        loader = DataLoader(self.train_ds, batch_size=BATCH, shuffle=True, drop_last=True)
        for episode in range(EPISODES):
            for batch in loader:
                opt_x.zero_grad()
                opt_y.zero_grad()
                out, gt = self._predict(batch)
                F.mse_loss(out, gt).backward()
                opt_x.step()
                opt_y.step()
            val = self._val_loss()
            print('post_train episode %d val_mse %.5f (%.0fs)' % (episode, val, time.time() - self._t0), flush=True)
            if val < best:
                best, bad = val, 0
                best_state = (copy.deepcopy(self.post_x.state_dict()), copy.deepcopy(self.post_y.state_dict()))
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
            for opt in (opt_x, opt_y):                  # official adjust_learning_rate_v2, lradj type1
                for g in opt.param_groups:
                    g['lr'] = POST_LR * 0.5 ** episode
        self.post_x.load_state_dict(best_state[0])
        self.post_y.load_state_dict(best_state[1])
        return best

    def adapt(self):
        t0 = self._t0 = time.time()
        best_val = self.post_train()
        self.times['post_train_seconds'] = round(time.time() - t0, 2)
        opt_x = torch.optim.Adam(self.post_x.parameters(), lr=ONLINE_LR)
        opt_y = torch.optim.Adam(self.post_y.parameters(), lr=ONLINE_LR)
        test = torch.tensor(np.asarray(self.test_ds.test), dtype=torch.float32).to(DEV)
        stamp = torch.tensor(np.asarray(self.test_ds.test_stamp), dtype=torch.float32).to(DEV)
        L, LL, H = self.L, self.cfg.DATA.LABEL_LEN, self.H

        def window(i):                                  # same slicing as ForecastingDataset.__getitem__(i)
            o = i + L
            return test[o - L:o][None], stamp[o - L:o][None], test[o - LL:o + H][None], stamp[o - LL:o + H][None]
        n = len(self.test_ds)
        mse, mae = torch.zeros(n, device=DEV), torch.zeros(n, device=DEV)
        t1 = time.time()
        for i in range(n):
            with torch.no_grad():
                out, gt = self._predict(window(i))
                mse[i] = ((out - gt) ** 2).mean()
                mae[i] = (out - gt).abs().mean()
            dumpkit.add(out)
            j = i - H + 1                               # window whose last target is the first target of window i
            if j >= L:                                  # the official buffer needs L + H observed first targets
                opt_x.zero_grad()
                opt_y.zero_grad()
                p_out, p_gt = self._predict(window(j))
                F.mse_loss(p_out, p_gt).backward()
                torch.nn.utils.clip_grad_norm_(self.post_x.parameters(), CLIP)
                torch.nn.utils.clip_grad_norm_(self.post_y.parameters(), CLIP)
                opt_x.step()
                opt_y.step()
        if DEV == 'cuda':
            torch.cuda.synchronize()
        self.times['online_seconds'] = round(time.time() - t1, 2)
        dumpkit.save()
        mse, mae = mse.cpu().numpy(), mae.cpu().numpy()
        self.mse_all, self.mae_all = mse, mae
        print(json.dumps({
            'model': 'DADAPTER',
            'time_statistics': {'overall_stats': {'total_time_seconds': round(time.time() - t0, 2), **self.times}},
            'final_results': {'best_val_mse': best_val, 'test_mse': float(mse.mean()), 'test_mae': float(mae.mean())},
        }, indent=2))

    def count_parameters(self):
        n = sum(p.numel() for p in list(self.post_x.parameters()) + list(self.post_y.parameters()))
        print(json.dumps({'parameters': {'total_params': int(n)}}))


def build_adapter(cfg, model, norm_module=None):
    return DeltaAdapter(cfg, model, norm_module=norm_module)
