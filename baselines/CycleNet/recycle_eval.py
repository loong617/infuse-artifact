"""ReCycle stage-1 evaluation: test-time closed-form recalibration of the learnable cycle.
Loads a trained CycleNet checkpoint, walks the test set chronologically in chunks,
recalibrates cycleQueue with ridge-closed-form (alpha, beta) from the recent true history
(strictly before each chunk's first forecast origin), and reports MSE/MAE.
Vanilla mode (--mode vanilla) reproduces the checkpoint's standard test numbers.
"""
import argparse
import sys
import time

import numpy as np
import torch

sys.path.insert(0, '.')
from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Solar  # noqa: E402
from models import CycleNet as CycleNetModel  # noqa: E402


def build_dataset(a):
    kw = dict(root_path=a.root_path, flag='test', size=[a.seq_len, 96, a.pred_len],
              features='M', data_path=a.data_path, scale=True, timeenc=0, freq='h', cycle=a.cycle)
    if a.data in ('ETTh1', 'ETTh2'):
        return Dataset_ETT_hour(**kw)
    if a.data in ('ETTm1', 'ETTm2'):
        return Dataset_ETT_minute(**kw)
    if a.data == 'Solar':
        return Dataset_Solar(**kw)
    return Dataset_Custom(**kw)


class Cfg:
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root_path', default='./dataset/')
    ap.add_argument('--data_path', required=True)
    ap.add_argument('--data', required=True)          # ETTh1 / ETTh2 / custom
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--seq_len', type=int, default=96)
    ap.add_argument('--pred_len', type=int, default=96)
    ap.add_argument('--enc_in', type=int, required=True)
    ap.add_argument('--cycle', type=int, required=True)
    ap.add_argument('--model_type', default='linear')
    ap.add_argument('--d_model', type=int, default=512)
    ap.add_argument('--use_revin', type=int, default=1)
    ap.add_argument('--mode', choices=['vanilla', 'recycle', 'grad'], default='vanilla')
    ap.add_argument('--steps', type=int, default=20, help='gradient TTA steps per chunk')
    ap.add_argument('--glr', type=float, default=0.01, help='gradient TTA lr')
    ap.add_argument('--hist', type=int, default=720, help='history length for estimation')
    ap.add_argument('--shrink', type=float, default=1.0, help='ridge prior weight toward alpha=1')
    ap.add_argument('--decay', type=float, default=1.0, help='EMA decay over estimation windows')
    ap.add_argument('--select', type=int, default=1, help='per-channel shrink selection on last window')
    ap.add_argument('--gate', type=float, default=0.0, help='do-no-harm gate threshold tau')
    ap.add_argument('--chunk', type=int, default=0, help='update every N steps (0 = pred_len)')
    ap.add_argument('--tag', default='')
    a = ap.parse_args()

    cfg = Cfg()
    for k in ['seq_len', 'pred_len', 'enc_in', 'cycle', 'model_type', 'd_model', 'use_revin']:
        setattr(cfg, k, getattr(a, k))
    ds = build_dataset(a)
    X = torch.tensor(ds.data_x, dtype=torch.float32)     # [T, C] scaled
    CI = ds.cycle_index                                   # [T] absolute-cycle positions
    n = len(ds)
    W = a.cycle
    chunk = a.chunk or a.pred_len

    model = CycleNetModel.Model(cfg)
    model.load_state_dict(torch.load(a.ckpt, map_location='cpu'))
    model.eval()
    Q0 = model.cycleQueue.data.clone()                    # [W, C] frozen learnable cycle
    # 模型约定:真值位于周期位置 p 的数据,其内部 Q 索引为 (p + seq_len) % W
    OFF = a.seq_len % W

    se, ae, cnt = 0.0, 0.0, 0
    t_upd, n_upd = 0.0, 0
    with torch.no_grad():
        k = 0
        while True:
            b = a.seq_len + k * chunk            # chunk 首个预测起点(绝对位置)
            i0 = k * chunk                       # 对应样本下标(origin - seq_len)
            if i0 >= n:
                break
            i1 = min(i0 + chunk, n)
            if a.mode != 'vanilla' and b >= a.seq_len + a.pred_len + 2 * W:
                n_est = max(a.hist // a.pred_len, 1)
                ows = [b - a.pred_len * (t + 1) for t in range(n_est)]
                ows = [o for o in ows if o - a.seq_len >= 0]
                exb = torch.stack([X[o - a.seq_len:o] for o in ows])
                ecb = torch.tensor([CI[o] for o in ows])
                eyb = torch.stack([X[o:o + a.pred_len] for o in ows])
                t0 = time.perf_counter()
                if a.mode == 'grad':
                    # 梯度式 TTA:同样只更新 cycleQueue,同一批估计窗口,Adam 优化
                    with torch.enable_grad():
                        with torch.no_grad():
                            model.cycleQueue.data.copy_(Q0)
                        model.cycleQueue.data.requires_grad_(True)
                        opt = torch.optim.Adam([model.cycleQueue.data], lr=a.glr)
                        for _ in range(a.steps):
                            opt.zero_grad()
                            loss = ((model(exb, ecb) - eyb) ** 2).mean()
                            loss.backward()
                            opt.step()
                        model.cycleQueue.data.requires_grad_(False)
                else:
                    # 模型在环闭式重校准:pred(alpha,beta) = f0 + alpha*(fq-f0) + beta*(f1-f0)
                    model.cycleQueue.data.copy_(torch.zeros_like(Q0)); f0 = model(exb, ecb)
                    model.cycleQueue.data.copy_(Q0);                    fq = model(exb, ecb)
                    model.cycleQueue.data.copy_(torch.ones_like(Q0));  f1 = model(exb, ecb)
                    g, h, r = fq - f0, f1 - f0, eyb - f0        # [n_est, H, C], ows[0] 最新
                    wts = (a.decay ** torch.arange(len(ows), dtype=torch.float32))[:, None, None]

                    def fit(shrink, idx):
                        gg, hh, gh_, gr, hr = (g * g * wts)[idx].sum((0, 1)) + 1e-8, \
                            (h * h * wts)[idx].sum((0, 1)), (g * h * wts)[idx].sum((0, 1)), \
                            (g * r * wts)[idx].sum((0, 1)), (h * r * wts)[idx].sum((0, 1))
                        lam = shrink * gg
                        det = (gg + lam) * (hh + lam) - gh_ * gh_ + 1e-8
                        al = ((gr + lam) * (hh + lam) - gh_ * hr) / det
                        be = ((gg + lam) * hr - gh_ * (gr + lam)) / det
                        return torch.clamp(al, 0.5, 2.0), torch.clamp(be, -2.0, 2.0)

                    if len(ows) >= 3 and a.select:
                        fit_idx = torch.arange(1, len(ows))     # 用旧窗口拟合
                        best_err, best_ab = None, None
                        for s in (0.0, 0.2, 1.0, 5.0):
                            al, be = fit(s, fit_idx)
                            err = (((al * g[0] + be * h[0]) - r[0]) ** 2).mean(0)  # [C]
                            if best_err is None:
                                best_err, best_ab = err, (al, be)
                            else:
                                m = err < best_err
                                best_ab = (torch.where(m, al, best_ab[0]), torch.where(m, be, best_ab[1]))
                                best_err = torch.where(m, err, best_err)
                        alpha, beta = best_ab
                        # 不伤害门控:改善不足 τ 的通道回退为恒等(α=1, β=0)
                        err_id = ((g[0] - r[0]) ** 2).mean(0)
                        gmask = best_err < err_id * (1.0 - a.gate)
                        alpha = torch.where(gmask, alpha, torch.ones_like(alpha))
                        beta = torch.where(gmask, beta, torch.zeros_like(beta))
                    else:
                        alpha, beta = fit(a.shrink, torch.arange(len(ows)))
                    model.cycleQueue.data.copy_(alpha[None, :] * Q0 + beta[None, :])
                t_upd += time.perf_counter() - t0
                n_upd += 1
            else:
                model.cycleQueue.data.copy_(Q0)
            for j in range(i0, i1, 64):
                js = min(j + 64, i1)
                xb = torch.stack([X[t:t + a.seq_len] for t in range(j, js)])
                cb = torch.tensor([CI[t + a.seq_len] for t in range(j, js)])
                true = torch.stack([X[t + a.seq_len:t + a.seq_len + a.pred_len] for t in range(j, js)])
                pred = model(xb, cb)
                se += ((pred - true) ** 2).sum().item()
                ae += (pred - true).abs().sum().item()
                cnt += true.numel()
            k += 1

    mse, mae = se / cnt, ae / cnt
    upd_ms = (t_upd / n_upd * 1000) if n_upd else 0.0
    print(f"[{a.tag or a.mode}] {a.data_path} H={a.pred_len} mode={a.mode} hist={a.hist} shrink={a.shrink} "
          f"chunk={chunk} mse:{mse:.6f} mae:{mae:.6f} update_ms/chunk:{upd_ms:.1f} (n={n_upd})")


if __name__ == '__main__':
    main()
