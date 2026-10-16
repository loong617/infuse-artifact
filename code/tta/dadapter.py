"""delta-Adapter (ICLR 2026, arXiv 2601.20280) faithful-lite port into the same online protocol.
Bounded input nudging + output residual correction, learned online by Adam on revealed data.
Channel-independent tiny adapters: A_in: Linear(L->L), A_out: Linear(H->H), tanh-squashed,
edits bounded by delta (0.1), following the paper's Ada-X+Y design at reduced parameter count.
"""
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.forecast import forecast
from datasets.loader import get_test_dataloader


class DeltaAdapter(nn.Module):
    def __init__(self, cfg, model, norm_module=None):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.norm_module = norm_module
        for p in self.model.parameters():
            p.requires_grad_(False)
        L, H, C = cfg.DATA.SEQ_LEN, cfg.DATA.PRED_LEN, cfg.MODEL.enc_in
        self.delta = float(os.environ.get('DA_DELTA', '0.1'))
        self.a_in = nn.Linear(L, L)
        self.a_out = nn.Linear(H, H)
        for m in (self.a_in, self.a_out):
            for p in m.parameters():
                p.requires_grad_(True)
        self.opt = torch.optim.Adam(list(self.a_in.parameters()) + list(self.a_out.parameters()),
                                    lr=float(os.environ.get('DA_LR', '0.001')))
        self.steps = int(os.environ.get('DA_STEPS', '1'))
        loader = get_test_dataloader(cfg)
        ds = loader.dataset
        self.ds = ds
        self.test = torch.tensor(np.asarray(ds.test), dtype=torch.float32).cuda()
        self.stamp = torch.tensor(np.asarray(ds.test_stamp), dtype=torch.float32).cuda()
        self.seq_len = cfg.DATA.SEQ_LEN
        self.pred_len = cfg.DATA.PRED_LEN
        self.label_len = cfg.DATA.LABEL_LEN
        self.n = len(ds)
        self.hist = 1440
        self.cuda()

    def _make_inputs(self, origins):
        enc = torch.stack([self.test[o - self.seq_len:o] for o in origins])
        enc_s = torch.stack([self.stamp[o - self.seq_len:o] for o in origins])
        dec = torch.stack([self.test[o - self.label_len:o + self.pred_len] for o in origins])
        dec_s = torch.stack([self.stamp[o - self.label_len:o + self.pred_len] for o in origins])
        return enc, enc_s, dec, dec_s

    def _nudge_in(self, x):
        return x + self.delta * torch.tanh(self.a_in(x.permute(0, 2, 1)).permute(0, 2, 1))

    def _adapt(self, origin):
        n_est = max(self.hist // self.pred_len, 1)
        ows = [origin - self.pred_len * (t + 1) for t in range(n_est)]
        ows = [o for o in ows if o - self.seq_len >= 0]
        if not ows:
            return
        enc, enc_s, dec, dec_s = self._make_inputs(ows)
        y = dec[:, -self.pred_len:, :].cuda()
        enc2 = enc.clone()
        with torch.enable_grad():
            for _ in range(self.steps):
                self.opt.zero_grad()
                enc2 = self._nudge_in(enc)
                pred = forecast(self.cfg, (enc2, enc_s, dec, dec_s), self.model, self.norm_module)[0]
                pred = pred + self.delta * torch.tanh(self.a_out(pred.permute(0, 2, 1)).permute(0, 2, 1))
                loss = F.mse_loss(pred, y)
                loss.backward()
                self.opt.step()

    def adapt(self):
        period = 24
        try:
            # 周期检测只用部署前数据(val尾部4L),不读测试流未揭示步
            if not hasattr(self, '_val_tail'):
                from datasets.loader import get_val_dataloader
                self._val_tail = torch.tensor(np.asarray(get_val_dataloader(self.cfg).dataset.val), dtype=torch.float32).cuda()
            x = self._val_tail[-self.seq_len * 4:]
            fft = torch.fft.rfft(x - x.mean(dim=0), dim=0)
            amp = torch.abs(fft)
            power = amp.pow(2).mean(0)
            period = max(min(x.shape[0] // torch.argmax(amp[:, power.argmax()]).item(), self.pred_len), 1)
        except Exception:
            pass
        chunk = period + 1
        mse_all, mae_all, t_upd, n_adapt = [], [], 0.0, 0
        i = 0
        with torch.no_grad():
            while i < self.n:
                b_end = min(i + chunk, self.n)
                origin = i + self.seq_len
                if origin >= self.seq_len + self.pred_len:
                    t0 = time.perf_counter()
                    with torch.enable_grad():
                        self._adapt(origin)
                    torch.cuda.synchronize()
                    t_upd += time.perf_counter() - t0
                    n_adapt += 1
                origins = list(range(i + self.seq_len, b_end + self.seq_len))
                enc, enc_s, dec, dec_s = self._make_inputs(origins)
                with torch.enable_grad():
                    enc2 = self._nudge_in(enc)
                pred = forecast(self.cfg, (enc2, enc_s, dec, dec_s), self.model, self.norm_module)[0]
                pred = pred + self.delta * torch.tanh(self.a_out(pred.permute(0, 2, 1)).permute(0, 2, 1))
                gt = dec[:, -self.pred_len:, :].cuda()
                mse_all.append(F.mse_loss(pred, gt, reduction='none').mean(dim=(-2, -1)).detach().cpu().numpy())
                mae_all.append(F.l1_loss(pred, gt, reduction='none').mean(dim=(-2, -1)).detach().cpu().numpy())
                i = b_end
        mse_all = np.concatenate(mse_all)
        mae_all = np.concatenate(mae_all)
        print('After TSF-TTA of delta-Adapter (ported)')
        print(f'Number of adaptations: {n_adapt}, avg {t_upd / max(n_adapt, 1) * 1000:.2f} ms')
        print(f'Test MSE: {mse_all.mean():.4f}, Test MAE: {mae_all.mean():.4f}')


def build_dadapter(cfg, model, norm_module=None):
    return DeltaAdapter(cfg, model, norm_module=norm_module)
