"""SAN-TTA (NeurIPS'23) + FAN (NeurIPS'24) 两个归一化系基线
SAN: 切片级自适应归一化——将窗口分片,逐片估计统计量并归一化/反归一化
FAN: 频域自适应归一化——FFT分解窗口,移除Top-K频率分量,预测后加回
"""
import time

import numpy as np
import torch
import torch.nn.functional as F

from models.forecast import forecast
from datasets.loader import get_test_dataloader


class NormBase:
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

    def adapt(self):
        period = self._fft_period()
        chunk = period + 1
        mse_all, mae_all, t_upd, n_adapt = [], [], 0.0, 0
        i = 0
        with torch.no_grad():
            while i < self.n:
                b_end = min(i + chunk, self.n)
                origin = i + self.seq_len
                origins = list(range(i + self.seq_len, b_end + self.seq_len))
                inp = self._make_inputs(origins)
                enc, enc_s, dec, dec_s = inp
                gt = dec[:, -self.pred_len:, :].cuda()
                t0 = time.perf_counter()
                pred = self._predict(inp)
                torch.cuda.synchronize()
                t_upd += time.perf_counter() - t0
                n_adapt += 1
                mse_all.append(F.mse_loss(pred, gt, reduction='none').mean(dim=(-2, -1)).reshape(-1).cpu().numpy())
                mae_all.append(F.l1_loss(pred, gt, reduction='none').mean(dim=(-2, -1)).reshape(-1).cpu().numpy())
                i = b_end
        mse_all = np.concatenate(mse_all)
        mae_all = np.concatenate(mae_all)
        print(f'After TTA of {self.tag}')
        print(f'Test MSE: {mse_all.mean():.4f}, Test MAE: {mae_all.mean():.4f}')


class SANAdapter(NormBase):
    """SAN (NeurIPS 2023): 切片级自适应归一化
    将 lookback 窗口按周期切片,逐片计算均值/方差并归一化;
    预测后用外推的切片统计量反归一化。
    """
    def __init__(self, cfg, model, norm_module=None):
        super().__init__(cfg, model, norm_module)
        self.tag = 'SAN-TTA (NeurIPS 2023)'
        self.period = self._fft_period()

    def _predict(self, inp):
        enc, enc_s, dec, dec_s = inp
        B, L, C = enc.shape
        H = self.pred_len
        P = self.period
        # 切片归一化(lookback)
        n_slices = max(L // P, 2)
        slice_means, slice_stds = [], []
        enc_norm = enc.clone()
        for s in range(n_slices):
            lo, hi = s * L // n_slices, (s + 1) * L // n_slices
            mu = enc[:, lo:hi, :].mean(dim=1, keepdim=True)
            sd = torch.sqrt(enc[:, lo:hi, :].var(dim=1, keepdim=True) + 1e-5)
            enc_norm[:, lo:hi, :] = (enc[:, lo:hi, :] - mu) / sd
            slice_means.append(mu)
            slice_stds.append(sd)
        # 预测
        pred = forecast(self.cfg, (enc_norm, enc_s, dec, dec_s), self.model, self.norm_module)[0]
        # 反归一化:用最后一片的统计量外推(线性外推均值)
        last_mu = slice_means[-1]
        prev_mu = slice_means[-2] if len(slice_means) > 1 else last_mu
        trend = last_mu - prev_mu
        future_mu = last_mu + trend * (H / max(L // n_slices, 1))
        future_sd = slice_stds[-1]
        return pred * future_sd + future_mu


class FANAdapter(NormBase):
    """FAN (NeurIPS 2024): 频域自适应归一化
    FFT 分解 lookback 窗口,移除 Top-K 频率分量(非平稳成分);
    预测残差后,用输入窗口的频率分量重建未来。
    """
    def __init__(self, cfg, model, norm_module=None):
        super().__init__(cfg, model, norm_module)
        self.tag = 'FAN (NeurIPS 2024)'
        self.K = 4  # Top-K 频率

    def _predict(self, inp):
        enc, enc_s, dec, dec_s = inp
        B, L, C = enc.shape
        H = self.pred_len
        # FFT 分解 lookback
        fft = torch.fft.rfft(enc, dim=1)
        amp = torch.abs(fft)
        power = amp.pow(2).mean(-1)  # [B, F]
        # 找 Top-K 频率
        topk_freq = torch.topk(power, min(self.K, power.shape[1]), dim=1).indices  # [B, K]
        # 构建掩码
        mask = torch.zeros_like(fft)
        for b in range(B):
            mask[b, topk_freq[b]] = fft[b, topk_freq[b]]
        # 非平稳成分(时域)
        nonstationary = torch.fft.irfft(mask, n=L, dim=1)
        # 平稳成分
        stationary = enc - nonstationary
        # 用平稳成分预测
        pred_stat = forecast(self.cfg, (stationary, enc_s, dec, dec_s), self.model, self.norm_module)[0]
        # 重建非平稳成分(用最后周期的模式外推)
        period = self._fft_period()
        ns_forecast = torch.zeros(B, H, C, device=enc.device)
        for h in range(H):
            idx = (L - period + h % period)
            ns_forecast[:, h, :] = nonstationary[:, idx, :]
        return pred_stat + ns_forecast


def build_san_adapter(cfg, model, norm_module=None):
    return SANAdapter(cfg, model, norm_module=norm_module)


def build_fan_adapter(cfg, model, norm_module=None):
    return FANAdapter(cfg, model, norm_module=norm_module)
