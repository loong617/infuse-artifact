"""DLinear + RCF(CycleNet 的周期分量,官方 Table 4 同款组合)作为第二骨干,
用于骨干泛化实验:证明 ReCycle 不依赖 CycleNet 骨干。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.cycle = configs.cycle
        # 官方 CycleNet Table 4 的 DLinear+RCF:周期分量 + 趋势/季节双线性
        self.cycleQueue = nn.Parameter(torch.zeros(self.cycle, self.enc_in), requires_grad=True)
        self.linear_seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.linear_trend = nn.Linear(self.seq_len, self.pred_len)
        self.linear_seasonal.weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        self.linear_trend.weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        wd = x_mark_enc[:, -1, 2].long()
        hr = x_mark_enc[:, -1, 3].long()
        c0 = ((wd * 24 + hr) + 1) % self.cycle
        dev = self.cycleQueue.device
        ar_in = torch.arange(self.seq_len, device=dev)
        q_in = self.cycleQueue[(c0[:, None] + ar_in[None, :]) % self.cycle]
        x = x_enc - q_in
        # DLinear: 滑动均值分解(官方 kernel 25)
        trend = F.avg_pool1d(x.permute(0, 2, 1), kernel_size=25, stride=1, padding=12)[:, :, :self.seq_len]
        trend = trend.permute(0, 2, 1)
        seasonal = x - trend
        y = (self.linear_seasonal(seasonal.permute(0, 2, 1)) +
             self.linear_trend(trend.permute(0, 2, 1))).permute(0, 2, 1)
        ar_out = torch.arange(self.pred_len, device=dev) + self.seq_len
        q_out = self.cycleQueue[(c0[:, None] + ar_out[None, :]) % self.cycle]
        return y + q_out
