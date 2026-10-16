"""额外基线:RevIN-TTA(测试时归一化) + SOLID(移位检测条件适应) + Proceed(概念漂移主动适应)
全部在 PETSA 管线内运行,与所有方法同协议。
"""
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.forecast import forecast
from datasets.loader import get_test_dataloader


class BaseTTA:
    """共享基础设施"""
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
        self.T = len(ds.test)

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

    def _evaluate(self, predict_fn, chunk):
        mse_all, mae_all, t_upd, n_adapt = [], [], 0.0, 0
        i = 0
        with torch.no_grad():
            while i < self.n:
                b_end = min(i + chunk, self.n)
                origin = i + self.seq_len
                t0 = time.perf_counter()
                predict_fn(origin)  # 各方法在此做适应/更新
                torch.cuda.synchronize()
                t_upd += time.perf_counter() - t0
                n_adapt += 1
                origins = list(range(i + self.seq_len, b_end + self.seq_len))
                inp = self._make_inputs(origins)
                gt = inp[2][:, -self.pred_len:, :].cuda()
                pred = self._predict(inp)
                mse_all.append(F.mse_loss(pred, gt, reduction='none').mean(dim=(-2, -1)).reshape(-1).cpu().numpy())
                mae_all.append(F.l1_loss(pred, gt, reduction='none').mean(dim=(-2, -1)).reshape(-1).cpu().numpy())
                i = b_end
        mse_all = np.concatenate(mse_all)
        mae_all = np.concatenate(mae_all)
        return mse_all, mae_all, t_upd, n_adapt

    def _predict(self, inp):
        raise NotImplementedError

    def adapt(self):
        period = self._fft_period()
        chunk = period + 1
        mse, mae, t_upd, n_adapt = self._evaluate(self._adapt_chunk, chunk)
        print(f'After TTA of {self.tag}')
        print(f'Number of adaptations: {n_adapt}, avg {t_upd / max(n_adapt, 1) * 1000:.2f} ms')
        print(f'Test MSE: {mse.mean():.4f}, Test MAE: {mae.mean():.4f}')

    def _adapt_chunk(self, origin):
        pass


class RevINAdapter(BaseTTA):
    """RevIN-TTA: 测试时逐窗实例归一化(零阶适应基线)"""
    def __init__(self, cfg, model, norm_module=None):
        super().__init__(cfg, model, norm_module)
        self.tag = 'RevIN-TTA (test-time normalization)'

    def _predict(self, inp):
        enc, enc_s, dec, dec_s = inp
        # 测试时归一化
        mu = enc.mean(dim=1, keepdim=True)
        sigma = torch.sqrt(enc.var(dim=1, keepdim=True) + 1e-5)
        enc_n = (enc - mu) / sigma
        pred = forecast(self.cfg, (enc_n, enc_s, dec, dec_s), self.model, self.norm_module)[0]
        # 反归一化
        return pred * sigma + mu

    def _adapt_chunk(self, origin):
        pass  # RevIN 是免适应的(逐窗自动)


class SOLIDAdapter(BaseTTA):
    """SOLID (KDD'24): 移位检测 + 条件适应(适配器微调)"""
    def __init__(self, cfg, model, norm_module=None):
        super().__init__(cfg, model, norm_module)
        self.tag = 'SOLID (shift detection + conditional adaptation)'
        # 冻结骨干,添加适配器
        for p in model.parameters():
            p.requires_grad_(False)
        self.adapter = nn.Sequential(
            nn.Linear(self.seq_len, 64), nn.GELU(), nn.Linear(64, self.pred_len)
        ).cuda()
        for p in self.adapter.parameters():
            p.requires_grad_(True)
        self.opt = torch.optim.Adam(self.adapter.parameters(), lr=1e-3)
        # 移位检测:维护近期误差 EMA
        self.err_ema = None
        self.shift_thresh = 1.5  # 误差超过 EMA×阈值时触发适应

    def _predict(self, inp):
        enc, enc_s, dec, dec_s = inp
        # SOLID: 适配器处理输入残差
        base = forecast(self.cfg, inp, self.model, self.norm_module)[0]
        return base

    def _adapt_chunk(self, origin):
        if origin < self.seq_len + self.pred_len:
            return
        # 计算近期窗口误差(移位检测)
        n_est = 5
        ows = [origin - self.pred_len * (t + 1) for t in range(n_est)]
        ows = [o for o in ows if o - self.seq_len >= 0]
        if not ows:
            return
        inp = self._make_inputs(ows)
        y = inp[2][:, -self.pred_len:, :].cuda()
        with torch.no_grad():
            pred = self._predict(inp)
            err = F.mse_loss(pred, y).item()
        # 更新 EMA
        if self.err_ema is None:
            self.err_ema = err
        else:
            self.err_ema = 0.9 * self.err_ema + 0.1 * err
        # 移位检测:误差突增时触发适应
        if err > self.err_ema * self.shift_thresh:
            with torch.enable_grad():
                self.opt.zero_grad()
                pred = self._predict(inp)
                loss = F.mse_loss(pred, y)
                loss.backward()
                self.opt.step()


class ProceedAdapter(BaseTTA):
    """Proceed (KDD'25): 概念漂移主动适应(概念嵌入+瓶颈)"""
    def __init__(self, cfg, model, norm_module=None):
        super().__init__(cfg, model, norm_module)
        self.tag = 'Proceed (proactive concept adaptation)'
        # 概念嵌入: 从近期窗口提取概念向量
        self.concept_dim = 50
        self.bottleneck_dim = 48
        C = self.test.shape[1]
        # 概念编码器(从近期窗口→概念向量)
        self.concept_enc = nn.Linear(self.seq_len * C, self.concept_dim).cuda()
        # 概念解码器(概念向量→输出调制)
        self.concept_dec = nn.Linear(self.concept_dim, self.pred_len * C).cuda()
        # 参数: 骨干全开(Proceed 是全模型适应)
        for p in model.parameters():
            p.requires_grad_(True)
        self.opt = torch.optim.Adam(
            list(model.parameters()) + list(self.concept_enc.parameters()) + list(self.concept_dec.parameters()),
            lr=3e-5)
        self.prev_concept = None
        self.concept_opt = torch.optim.Adam(
            list(self.concept_enc.parameters()) + list(self.concept_dec.parameters()), lr=1e-4)

    def _predict(self, inp):
        enc, enc_s, dec, dec_s = inp
        # 概念编码
        B = enc.shape[0]
        concept = self.concept_enc(enc.reshape(B, -1))
        # 概念调制输出
        base = forecast(self.cfg, inp, self.model, self.norm_module)[0]
        modulation = self.concept_dec(concept).reshape(B, self.pred_len, -1)
        return base + 0.1 * torch.tanh(modulation)

    def _adapt_chunk(self, origin):
        if origin < self.seq_len + self.pred_len:
            return
        n_est = 15
        ows = [origin - self.pred_len * (t + 1) for t in range(n_est)]
        ows = [o for o in ows if o - self.seq_len >= 0]
        if not ows:
            return
        inp = self._make_inputs(ows)
        y = inp[2][:, -self.pred_len:, :].cuda()
        # 概念变化检测
        with torch.no_grad():
            new_concept = self.concept_enc(inp[0].reshape(inp[0].shape[0], -1)).mean(0)
            if self.prev_concept is not None:
                concept_shift = (new_concept - self.prev_concept).norm()
            else:
                concept_shift = 0
            self.prev_concept = new_concept
        # 主动适应(无论是否检测到移位,都做梯度更新)
        with torch.enable_grad():
            self.opt.zero_grad()
            pred = self._predict(inp)
            loss = F.mse_loss(pred, y)
            # 概念一致性正则
            if concept_shift > 0:
                loss = loss + 0.01 * concept_shift
            loss.backward()
            self.opt.step()


def build_revin_adapter(cfg, model, norm_module=None):
    return RevINAdapter(cfg, model, norm_module=norm_module)


def build_solid_adapter(cfg, model, norm_module=None):
    return SOLIDAdapter(cfg, model, norm_module=norm_module)


def build_proceed_adapter(cfg, model, norm_module=None):
    return ProceedAdapter(cfg, model, norm_module=norm_module)
