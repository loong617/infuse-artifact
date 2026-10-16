"""CycleNet backbone (Linear + learnable recurrent cycle) for the PETSA/TAFAS pipeline.
Interface matches other backbones: forward(x_enc, x_mark_enc, x_dec, x_mark_dec).
Cycle position is derived from the calendar marks (timeenc=0: [month, day, weekday, hour]).
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.cycle = configs.cycle
        self.cycleQueue = nn.Parameter(torch.zeros(self.cycle, self.enc_in), requires_grad=True)
        self.model = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # 第一个预测步的周期位置(最后一格编码窗口 + 1);学习到的 Q 会吸收任何常数偏移
        wd = x_mark_enc[:, -1, 2].long()
        hr = x_mark_enc[:, -1, 3].long()
        c0 = ((wd * 24 + hr) + 1) % self.cycle                    # [B]
        dev = self.cycleQueue.device
        ar_in = torch.arange(self.seq_len, device=dev)
        q_in = self.cycleQueue[(c0[:, None] + ar_in[None, :]) % self.cycle]     # [B,L,C]
        x = x_enc - q_in
        y = self.model(x.permute(0, 2, 1)).permute(0, 2, 1)
        ar_out = torch.arange(self.pred_len, device=dev) + self.seq_len
        q_out = self.cycleQueue[(c0[:, None] + ar_out[None, :]) % self.cycle]   # [B,H,C]
        return y + q_out
