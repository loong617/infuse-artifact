"""统一协议下的额外基线:Online(纯梯度) + LoRA-TTA(低秩适配)
在 PETSA 管线内运行,与 TAFAS/PETSA/ReCycle 完全同协议同骨干。
"""
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.forecast import forecast
from datasets.loader import get_test_dataloader


class OnlineAdapter:
    """Online: 纯梯度在线学习(continual fine-tuning)—— OnlineTSF 框架的 Naive 基线"""
    def __init__(self, cfg, model, norm_module=None, mode='online'):
        self.cfg = cfg
        self.model = model
        self.norm_module = norm_module
        self.mode = mode  # 'online' | 'lora'
        loader = get_test_dataloader(cfg)
        ds = loader.dataset
        self.test = torch.tensor(np.asarray(ds.test), dtype=torch.float32).cuda()
        self.stamp = torch.tensor(np.asarray(ds.test_stamp), dtype=torch.float32).cuda()
        self.seq_len = cfg.DATA.SEQ_LEN
        self.pred_len = cfg.DATA.PRED_LEN
        self.label_len = cfg.DATA.LABEL_LEN
        self.n = len(ds)
        self.T = len(ds.test)

        # 选择可训练参数
        if mode == 'lora':
            # LoRA: 冻结骨干,只训低秩适配器(挂在 cycleQueue 所在骨干的每个 Linear 后)
            for p in model.parameters():
                p.requires_grad_(False)
            self.lora_layers = []
            for name, mod in model.named_modules():
                if isinstance(mod, nn.Linear):
                    lora = LoRALayer(mod.in_features, mod.out_features, rank=8).cuda()
                    self.lora_layers.append((name, mod, lora))
                    for p in lora.parameters():
                        p.requires_grad_(True)
            params = [p for l in self.lora_layers for p in l[2].parameters()]
        else:
            for p in model.parameters():
                p.requires_grad_(True)
            params = list(model.parameters())
        self.optimizer = torch.optim.Adam(params, lr=1e-3)

    def _make_inputs(self, origins):
        enc = torch.stack([self.test[o - self.seq_len:o] for o in origins])
        enc_s = torch.stack([self.stamp[o - self.seq_len:o] for o in origins])
        dec = torch.stack([self.test[o - self.label_len:o + self.pred_len] for o in origins])
        dec_s = torch.stack([self.stamp[o - self.label_len:o + self.pred_len] for o in origins])
        return enc, enc_s, dec, dec_s

    def _forward(self, inp):
        enc, enc_s, dec, dec_s = inp
        if self.mode == 'lora':
            # 注入 LoRA
            originals = {}
            for name, mod, lora in self.lora_layers:
                originals[name] = mod.forward
                mod.forward = lora(mod)
            pred = forecast(self.cfg, inp, self.model, self.norm_module)[0]
            for name, mod in [(n, m) for n, m, _ in self.lora_layers]:
                mod.forward = originals[name]
            return pred
        return forecast(self.cfg, inp, self.model, self.norm_module)[0]

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
        mse_all, mae_all = [], []
        t_upd, n_adapt = 0.0, 0
        i = 0
        with torch.no_grad():
            while i < self.n:
                b_end = min(i + chunk, self.n)
                origin = i + self.seq_len
                if origin >= self.seq_len + self.pred_len:
                    t0 = time.perf_counter()
                    # 适应:用最近的已揭示窗口做梯度步
                    n_est = 15
                    ows = [origin - self.pred_len * (t + 1) for t in range(n_est)]
                    ows = [o for o in ows if o - self.seq_len >= 0]
                    if ows:
                        inp = self._make_inputs(ows)
                        y = inp[2][:, -self.pred_len:, :].cuda()
                        with torch.enable_grad():
                            self.optimizer.zero_grad()
                            pred = self._forward(inp)
                            loss = F.mse_loss(pred, y)
                            loss.backward()
                            self.optimizer.step()
                    torch.cuda.synchronize()
                    t_upd += time.perf_counter() - t0
                    n_adapt += 1
                # 预测当前块
                origins = list(range(i + self.seq_len, b_end + self.seq_len))
                inp = self._make_inputs(origins)
                pred = self._forward(inp)
                gt = inp[2][:, -self.pred_len:, :].cuda()
                mse_all.append(F.mse_loss(pred, gt, reduction='none').mean(dim=(-2, -1)).reshape(-1).cpu().numpy())
                mae_all.append(F.l1_loss(pred, gt, reduction='none').mean(dim=(-2, -1)).reshape(-1).cpu().numpy())
                i = b_end
        mse_all = np.concatenate(mse_all)
        mae_all = np.concatenate(mae_all)
        tag = 'Online (gradient)' if self.mode == 'online' else 'LoRA-TTA'
        print(f'After TTA of {tag}')
        print(f'Number of adaptations: {n_adapt}, avg {t_upd / max(n_adapt, 1) * 1000:.2f} ms')
        print(f'Test MSE: {mse_all.mean():.4f}, Test MAE: {mae_all.mean():.4f}')


class LoRALayer(nn.Module):
    def __init__(self, in_f, out_f, rank=8):
        super().__init__()
        self.lora_A = nn.Parameter(torch.zeros(rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.scale = 0.1

    def forward(self, layer):
        original_forward = layer.forward
        def wrapped(x):
            return original_forward(x) + self.scale * (x @ self.lora_A.T @ self.lora_B.T)
        return wrapped


def build_online_adapter(cfg, model, norm_module=None):
    return OnlineAdapter(cfg, model, norm_module=norm_module, mode='online')


def build_lora_adapter(cfg, model, norm_module=None):
    return OnlineAdapter(cfg, model, norm_module=norm_module, mode='lora')
