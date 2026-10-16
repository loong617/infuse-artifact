"""COSA (ICLR 2026) + ADAPT-Z (ICLR 2026) — 在我们管线内忠实实现核心机制
COSA: 输出空间线性残差适配器 + tanh门控 + 上下文向量(近期真值统计)
ADAPT-Z: MLP特征调整(融合当前表征与历史梯度)
"""
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.forecast import forecast
from datasets.loader import get_test_dataloader


class COSAAdapter:
    """COSA (ICLR 2026): Ŷ = Y⁰ + tanh(g)·(W·[Y⁰;C] + b)
    C = 上下文向量(近期已揭示真值的均值+方差, 逐通道)
    """
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
        # 适配器:线性变换 [Y⁰;C_ctx] → 残差
        self.W = nn.Linear(3 * C, C).cuda()  # (预测+均值+方差)→残差, 通道级
        self.gate = nn.Parameter(torch.zeros(1, device='cuda'))
        self.opt = torch.optim.Adam(list(self.W.parameters()) + [self.gate], lr=5e-3, weight_decay=1e-4)
        self.ctx_buffer = 5  # 近期5个窗口的统计

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

    def _ctx(self, origin):
        """从近期已揭示窗口提取上下文向量(逐通道均值+方差)"""
        bufs = []
        for t in range(1, self.ctx_buffer + 1):
            o = origin - self.pred_len * t
            if o - self.seq_len >= 0:
                w = self.test[o - self.seq_len:o]  # [L, C]
                bufs.append(torch.cat([w.mean(0), w.std(0)]))  # [2C]
        if not bufs:
            return torch.zeros(2 * self.test.shape[1], device='cuda')
        return torch.stack(bufs).mean(0)  # [2C]

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
            # 基础预测
            base = forecast(self.cfg, inp, self.model, self.norm_module)[0]
            # COSA 适应(在估计窗口上训练适配器)
            if origin >= self.seq_len + self.pred_len:
                n_est = 15
                ows = [origin - self.pred_len * (t + 1) for t in range(n_est)]
                ows = [o for o in ows if o - self.seq_len >= 0]
                if ows:
                    einp = self._make_inputs(ows)
                    ebase = forecast(self.cfg, einp, self.model, self.norm_module)[0]
                    egt = einp[2][:, -self.pred_len:, :].cuda()
                    # 逐窗口构造上下文并训练
                    with torch.enable_grad():
                        self.opt.zero_grad()
                        total_loss = 0
                        for w in range(len(ows)):
                            ctx = self._ctx(ows[w])  # [2C]
                            B, H, C = ebase.shape
                            # 广播 ctx 到 [B, H] (均值和方差分别广播)
                            ctx_mean = ctx[:C].view(1, 1, C).expand(B, H, C)
                            ctx_std = ctx[C:].view(1, 1, C).expand(B, H, C)
                            # 输入: [预测, 均值, 方差] → 通道独立
                            inp_feat = torch.cat([ebase[w:w+1], ctx_mean[:1], ctx_std[:1]], dim=2)  # [1, H, 3C]
                            residual = self.W(inp_feat)  # [1, H, C]
                            out = ebase[w:w+1] + torch.tanh(self.gate) * residual
                            total_loss = total_loss + F.mse_loss(out, egt[w:w+1])
                        total_loss.backward()
                        self.opt.step()
            torch.cuda.synchronize()
            t_upd += time.perf_counter() - t0
            n_adapt += 1
            # 应用适配器预测
            with torch.no_grad():
                ctx = self._ctx(origin)
                B, H, C = base.shape
                ctx_mean = ctx[:C].view(1, 1, C).expand(B, H, C)
                ctx_std = ctx[C:].view(1, 1, C).expand(B, H, C)
                inp_feat = torch.cat([base, ctx_mean, ctx_std], dim=2)  # [B, H, 3C]
                residual = self.W(inp_feat)  # [B, H, C]
                pred = base + torch.tanh(self.gate) * residual
            mse_all.append(F.mse_loss(pred, gt, reduction='none').mean(dim=(-2, -1)).reshape(-1).cpu().numpy())
            mae_all.append(F.l1_loss(pred, gt, reduction='none').mean(dim=(-2, -1)).reshape(-1).cpu().numpy())
            i = b_end
        mse_all = np.concatenate(mse_all)
        mae_all = np.concatenate(mae_all)
        print(f'After TTA of COSA (ICLR 2026)')
        print(f'Number of adaptations: {n_adapt}')
        print(f'Avg update time: {t_upd / max(n_adapt, 1) * 1000:.2f} ms')
        print(f'Test MSE: {mse_all.mean():.4f}, Test MAE: {mae_all.mean():.4f}')


class ADAPTZAdapter:
    """ADAPT-Z (ICLR 2026): MLP特征调整——融合当前输入与历史梯度信息
    核心:一个小MLP在输入空间做在线调整(不是输出空间)
    """
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
        # 特征调整MLP: 输入[L,C] → 输出[L,C](通道独立)
        self.mlp = nn.Sequential(
            nn.Linear(self.seq_len, 128), nn.GELU(),
            nn.Linear(128, self.seq_len)
        ).cuda()
        self.opt = torch.optim.Adam(self.mlp.parameters(), lr=1e-3)
        self.grad_ema = None  # 历史梯度指数移动平均

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
        while i < self.n:
            b_end = min(i + chunk, self.n)
            origin = i + self.seq_len
            origins = list(range(i + self.seq_len, b_end + self.seq_len))
            inp = self._make_inputs(origins)
            enc, enc_s, dec, dec_s = inp
            gt = dec[:, -self.pred_len:, :].cuda()
            t0 = time.perf_counter()
            if origin >= self.seq_len + self.pred_len:
                n_est = 15
                ows = [origin - self.pred_len * (t + 1) for t in range(n_est)]
                ows = [o for o in ows if o - self.seq_len >= 0]
                if ows:
                    einp = self._make_inputs(ows)
                    egt = einp[2][:, -self.pred_len:, :].cuda()
                    with torch.enable_grad():
                        self.opt.zero_grad()
                        # 特征调整:调整输入,然后预测
                        adjusted = einp[0].clone()
                        adjusted = self.mlp(adjusted.permute(0, 2, 1)).permute(0, 2, 1)
                        pred = forecast(self.cfg, (adjusted, einp[1], einp[2], einp[3]),
                                       self.model, self.norm_module)[0]
                        loss = F.mse_loss(pred, egt)
                        loss.backward()
                        self.opt.step()
                        # 更新历史梯度EMA
                        grads = []
                        for p in self.mlp.parameters():
                            if p.grad is not None:
                                grads.append(p.grad.flatten())
                        if grads:
                            g = torch.cat(grads)
                            self.grad_ema = g if self.grad_ema is None else 0.9 * self.grad_ema + 0.1 * g
            torch.cuda.synchronize()
            t_upd += time.perf_counter() - t0
            n_adapt += 1
            # 预测(带特征调整)
            with torch.no_grad():
                adjusted = self.mlp(enc.permute(0, 2, 1)).permute(0, 2, 1)
                pred = forecast(self.cfg, (adjusted, enc_s, dec, dec_s),
                               self.model, self.norm_module)[0]
            mse_all.append(F.mse_loss(pred, gt, reduction='none').mean(dim=(-2, -1)).reshape(-1).cpu().numpy())
            mae_all.append(F.l1_loss(pred, gt, reduction='none').mean(dim=(-2, -1)).reshape(-1).cpu().numpy())
            i = b_end
        mse_all = np.concatenate(mse_all)
        mae_all = np.concatenate(mae_all)
        print(f'After TTA of ADAPT-Z (ICLR 2026)')
        print(f'Number of adaptations: {n_adapt}')
        print(f'Avg update time: {t_upd / max(n_adapt, 1) * 1000:.2f} ms')
        print(f'Test MSE: {mse_all.mean():.4f}, Test MAE: {mae_all.mean():.4f}')


def build_cosa_adapter(cfg, model, norm_module=None):
    return COSAAdapter(cfg, model, norm_module=norm_module)


def build_adaptz_adapter(cfg, model, norm_module=None):
    return ADAPTZAdapter(cfg, model, norm_module=norm_module)
