"""δ-Adapter (ICLR 2026) — 官方代码忠实移植
来源: baselines/AdapterOfficial/Adapter-X+Y/experiments/exp_online_xy_add.py
核心: PostProcessingNet(3层MLP+InstanceNorm+tanh×δ) × 2(输入X+输出Y)
参数: δ=0.1, lr=1e-5, hidden=256(论文默认)
"""
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.forecast import forecast
from datasets.loader import get_test_dataloader


class PostProcessingNet(nn.Module):
    """官方 PostProcessingNet — 逐字移植"""
    def __init__(self, state_dim, hidden_dim, action_dim, delta=0.1):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.bn1 = nn.InstanceNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.InstanceNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.delta = delta

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.fc3(x)
        return torch.tanh(x) * self.delta


class DeltaAdapterOfficial:
    """δ-Adapter 官方 Adapter-X+Y 模式(在线适应)"""
    def __init__(self, cfg, model, norm_module=None):
        self.cfg = cfg
        self.model = model
        self.norm_module = norm_module
        loader = get_test_dataloader(cfg)
        ds = loader.dataset
        self.test = torch.tensor(np.asarray(ds.test), dtype=torch.float32).cuda()
        self.stamp = torch.tensor(np.asarray(ds.test_stamp), dtype=torch.float32).cuda()
        self.seq_len = cfg.DATA.SEQ_LEN
        self.pred_len = cfg.DATA.PRED_LEN
        self.label_len = cfg.DATA.LABEL_LEN
        self.n = len(ds)
        C = self.test.shape[1]
        hidden = 256  # 论文默认
        delta = 0.1   # 论文默认

        # 输入适配器(X): [L,C] → [L,C]
        self.post_x = PostProcessingNet(self.seq_len, hidden, self.seq_len, delta).cuda()
        # 输出适配器(Y): [H,C] → [H,C]
        self.post_y = PostProcessingNet(self.pred_len, hidden, self.pred_len, delta).cuda()

        # 官方 lr=1e-5
        self.opt = torch.optim.Adam(
            list(self.post_x.parameters()) + list(self.post_y.parameters()),
            lr=1e-5, foreach=False)

    def _make_inputs(self, origins):
        enc = torch.stack([self.test[o - self.seq_len:o] for o in origins])
        enc_s = torch.stack([self.stamp[o - self.seq_len:o] for o in origins])
        dec = torch.stack([self.test[o - self.label_len:o + self.pred_len] for o in origins])
        dec_s = torch.stack([self.stamp[o - self.label_len:o + self.pred_len] for o in origins])
        return enc, enc_s, dec, dec_s

    def _fft_period(self):
        # 周期检测只用部署前数据(val尾部4L),不读测试流未揭示步
        if not hasattr(self, '_val_tail'):
            from datasets.loader import get_val_dataloader
            self._val_tail = torch.tensor(np.asarray(get_val_dataloader(self.cfg).dataset.val), dtype=torch.float32).cuda()
        x = self._val_tail[-self.seq_len * 4:]
        try:
            fft = torch.fft.rfft(x - x.mean(dim=0), dim=0)
            amp = torch.abs(fft)
            power = amp.pow(2).mean(0)
            return max(min(x.shape[0] // torch.argmax(amp[:, power.argmax()]).item(), self.pred_len), 1)
        except Exception:
            return 24

    def _forward_with_adapters(self, inp):
        enc, enc_s, dec, dec_s = inp
        # 输入适配(官方: batch_x → post_x → modified_x)
        # PostProcessingNet 操作最后一维,输入 [B,L,C] → permute → [B,C,L] → adapter → [B,C,L] → permute
        enc_adj = self.post_x(enc.permute(0, 2, 1)).permute(0, 2, 1)
        base = forecast(self.cfg, (enc_adj, enc_s, dec, dec_s), self.model, self.norm_module)[0]
        # 输出适配(官方: output → post_y → final)
        out_adj = self.post_y(base.permute(0, 2, 1)).permute(0, 2, 1)
        return out_adj

    def adapt(self):
        period = self._fft_period()
        chunk = period + 1
        mse_all, mae_all, t_upd, n_adapt = [], [], 0.0, 0
        i = 0
        while i < self.n:
            b_end = min(i + chunk, self.n)
            origin = i + self.seq_len
            origins = list(range(i + self.seq_len, b_end + self.seq_len))
            inp = self._make_inputs(origins)
            enc, enc_s, dec, dec_s = inp
            gt = dec[:, -self.pred_len:, :].cuda()
            t0 = time.perf_counter()
            # 在线训练(用已揭示窗口)
            if origin >= self.seq_len + self.pred_len:
                n_est = 15
                ows = [origin - self.pred_len * (t + 1) for t in range(n_est)]
                ows = [o for o in ows if o - self.seq_len >= 0]
                if ows:
                    einp = self._make_inputs(ows)
                    egt = einp[2][:, -self.pred_len:, :].cuda()
                    with torch.enable_grad():
                        self.opt.zero_grad()
                        pred = self._forward_with_adapters(einp)
                        loss = F.mse_loss(pred, egt)
                        loss.backward()
                        self.opt.step()
            torch.cuda.synchronize()
            t_upd += time.perf_counter() - t0
            n_adapt += 1
            # 预测
            with torch.no_grad():
                pred = self._forward_with_adapters(inp)
            mse_all.append(F.mse_loss(pred, gt, reduction='none').mean(dim=(-2, -1)).reshape(-1).cpu().numpy())
            mae_all.append(F.l1_loss(pred, gt, reduction='none').mean(dim=(-2, -1)).reshape(-1).cpu().numpy())
            i = b_end
        mse_all = np.concatenate(mse_all)
        mae_all = np.concatenate(mae_all)
        print(f'After TTA of δ-Adapter (ICLR 2026, official)')
        print(f'Test MSE: {mse_all.mean():.4f}, Test MAE: {mae_all.mean():.4f}')


def build_dadapter_official(cfg, model, norm_module=None):
    return DeltaAdapterOfficial(cfg, model, norm_module=norm_module)
