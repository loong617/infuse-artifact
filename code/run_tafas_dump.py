"""D6 诊断:运行 TAFAS 时逐步快照模型参数,解剖梯度到底学了什么。
用法: NONORM=1 python run_tafas_dump.py ETTh1 24
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, '.')
from utils.parser import parse_args, load_config  # noqa: E402
from datasets.build import update_cfg_from_dataset  # noqa: E402
from models.build import build_model, load_best_model, build_norm_module  # noqa: E402
from utils.misc import set_seeds, set_devices  # noqa: E402
from tta.tafas import build_adapter  # noqa: E402
import tta.tafas as tafas_mod  # noqa: E402
from config import get_norm_module_cfg  # noqa: E402

DS, CYC = sys.argv[1], sys.argv[2]
snap = {'n': 0, 'cyc': [], 'w': []}
_orig_forecast = tafas_mod.forecast


def forecast_probe(cfg, inputs, model, norm_module=None):
    if snap['n'] % 20 == 0:
        snap['cyc'].append(model.cycleQueue.data.detach().cpu().clone())
        snap['w'].append(model.model.weight.detach().cpu().clone())
    snap['n'] += 1
    return _orig_forecast(cfg, inputs, model, norm_module)


tafas_mod.forecast = forecast_probe

sys.argv = ['x', 'DATA.NAME', DS, 'DATA.PRED_LEN', '96', 'MODEL.NAME', 'CycleNet',
            'MODEL.pred_len', '96', 'MODEL.cycle', CYC, 'TRAIN.ENABLE', 'False',
            'TRAIN.CHECKPOINT_DIR', f'./checkpoints/CycleNet/{DS}_96_nonorm/',
            'TTA.ENABLE', 'True', 'TEST.ENABLE', 'False', 'RESULT_DIR', './results/TAFAS_dump/',
            'TTA.SOLVER.BASE_LR', '0.001', 'TTA.SOLVER.WEIGHT_DECAY', '0.0001',
            'TTA.TAFAS.GATING_INIT', '0.01', 'NORM_MODULE.ENABLE', 'False',
            'DATA_LOADER.NUM_WORKERS', '0']

args = parse_args()
cfg = load_config(args)
update_cfg_from_dataset(cfg, cfg.DATA.NAME)
cfg.RESULT_DIR = './results/TAFAS_dump/'
os.makedirs(cfg.RESULT_DIR, exist_ok=True)
set_devices(cfg.VISIBLE_DEVICES)
set_seeds(cfg.SEED)
model = build_model(cfg)
model = load_best_model(cfg, model)

adapter = build_adapter(cfg, model)
adapter.adapt()

C = torch.stack(snap['cyc'])   # [T, W, C]
W = torch.stack(snap['w'])     # [T, H, L]
Q0, W0 = C[0], W[0]
print(f'\n===== D6 解剖 ({DS}): {len(C)} 个快照 =====')

# 1) cycleQueue 的最终变化能否被仿射(α,β)解释?
Qf = C[-1]
qd = (Qf - Q0).flatten()
alpha_fit = (Q0.flatten() * qd).sum() / (Q0.flatten() ** 2).sum()
beta_fit = qd.mean()
resid = qd - (alpha_fit * Q0.flatten() + beta_fit)
r2 = 1 - resid.norm() ** 2 / (qd.norm() ** 2 + 1e-9)
print(f'Δcycle: 范数 {qd.norm():.4f} | 最佳仿射解释 R² {r2:.3f} (α-1={alpha_fit:.4f}, β={beta_fit:.4f})')

# 2) ΔW 按输入位置(时间新旧)的分布:梯度把权重往哪里搬?
dW = (W[-1] - W0)                     # [H, L]
col = dW.abs().mean(0)                # [L], 每个输入位置的权重变化
old_half = col[:L // 2].mean() if (L := col.shape[0]) else 0
new_half = col[L // 2:].mean()
print(f'ΔW 幅度: 输入旧半段 {old_half:.5f} vs 新半段 {new_half:.5f} (比值 {new_half / (old_half + 1e-12):.2f})')
last8 = col[-8:].mean(); first8 = col[:8].mean()
print(f'ΔW: 最后8步 {last8:.5f} vs 最初8步 {first8:.5f} (比值 {last8 / (first8 + 1e-12):.2f})')

# 3) ΔW 按输出 horizon 位置的分布
row = dW.abs().mean(1)
print(f'ΔW by horizon: 近(前1/3) {row[:row.shape[0] // 3].mean():.5f} 中 {row[row.shape[0] // 3: 2 * row.shape[0] // 3].mean():.5f} 远(后1/3) {row[2 * row.shape[0] // 3:].mean():.5f}')

# 4) 行为测试:常数平移敏感度 d pred / d c(适应前后)
x = torch.randn(1, 96, 7).cuda() * 0.5 + 1.0
xm = torch.zeros(1, 96, 4).cuda(); xm[..., 2] = 1; xm[..., 3] = 10
xd = torch.zeros(1, 96 + 96, 7).cuda()
with torch.no_grad():
    model.cycleQueue.data.copy_(Q0.cuda()); model.model.weight.data.copy_(W0.cuda())
    p0 = model(x, xm, xd, xm)
    p0s = model(x + 0.1, xm, xd, xm)
    g0 = ((p0s - p0) / 0.1).mean().item()
    model.cycleQueue.data.copy_(C[-1].cuda()); model.model.weight.data.copy_(W[-1].cuda())
    p1 = model(x, xm, xd, xm)
    p1s = model(x + 0.1, xm, xd, xm)
    g1 = ((p1s - p1) / 0.1).mean().item()
print(f'常数平移增益: 适应前 {g0:.3f} → 适应后 {g1:.3f}')

# 5) ΔW 的低维性:奇异值谱
sv = torch.linalg.svdvals(dW)
print(f'ΔW 奇异值谱 top5: {[f"{v:.4f}" for v in sv[:5]]} | 谱熵贡献: top1 {sv[0] ** 2 / (sv ** 2).sum():.2f}, top3 {(sv[:3] ** 2).sum() / (sv ** 2).sum():.2f}')
np.save('d6_dW.npy', dW.numpy())
np.save('d6_dQ.npy', (Qf - Q0).numpy())
print('saved d6_dW.npy / d6_dQ.npy')
