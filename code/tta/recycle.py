"""ReCycle adapter inside the PETSA/TAFAS online protocol.
Chunked online evaluation mirroring petsa.Adapter's stepping and metric accounting,
but adaptation is a closed-form recalibration of cycleQueue using ONLY strictly-past
revealed ground truth (no partial-future use, i.e. an information-conservative variant).
"""
import os
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F

from models.forecast import forecast
from datasets.loader import get_test_dataloader, get_val_dataloader


class RecycleAdapter:
    def __init__(self, cfg, model, norm_module=None):
        self.cfg = cfg
        self.model = model
        self.norm_module = norm_module
        self.test_loader = get_test_dataloader(cfg)
        ds = self.test_loader.dataset
        self.ds = ds
        self.test = torch.tensor(np.asarray(ds.test), dtype=torch.float32).cuda()
        self.val = torch.tensor(np.asarray(get_val_dataloader(cfg).dataset.val), dtype=torch.float32).cuda()
        self.stamp = torch.tensor(np.asarray(ds.test_stamp), dtype=torch.float32).cuda()
        self.seq_len = cfg.DATA.SEQ_LEN
        self.pred_len = cfg.DATA.PRED_LEN
        self.label_len = cfg.DATA.LABEL_LEN
        self.cycle = cfg.MODEL.cycle
        self.Q0 = model.cycleQueue.data.clone()
        self.n = len(ds)                      # 测试窗口数
        self.T = len(ds.test)                 # 测试序列长度
        self.hist = int(os.environ.get('RECYCLE_HIST', '1440'))
        self.gate = 0.1
        # 混合模式:梯度持续适应骨干(TAFAS式)+ 闭式重校准周期(ReCycle)
        self.hybrid = os.environ.get('RECYCLE_HYBRID') == '1'
        self.blr = float(os.environ.get('RECYCLE_BLR', '0.001'))
        self.gsteps = int(os.environ.get('RECYCLE_GSTEPS', '1'))
        if self.hybrid:
            for p in self.model.model.parameters():
                p.requires_grad_(True)
            self.bopt = torch.optim.Adam(self.model.model.parameters(), lr=self.blr)
            self.W0 = deepcopy(self.model.model.state_dict())   # v12: CoTTA式随机恢复用

    # ---------- 工具 ----------
    def _make_inputs(self, origins):
        """给定预测起点列表(第一个预测步的下标), 构造 forecast() 需要的四元组"""
        enc = torch.stack([self.test[o - self.seq_len:o] for o in origins])
        enc_s = torch.stack([self.stamp[o - self.seq_len:o] for o in origins])
        dec = torch.stack([self.test[o - self.label_len:o + self.pred_len] for o in origins])
        dec_s = torch.stack([self.stamp[o - self.label_len:o + self.pred_len] for o in origins])
        return enc, enc_s, dec, dec_s

    def _fft_period(self):
        # chunk周期由部署前数据(val尾部4L步)的FFT确定:全程因果,不读测试流任何未揭示步
        # (周期值与原开发版从测试流前缀读到的一致,已逐数据集核验)
        x = self.val[-self.seq_len * 4:]
        fft = torch.fft.rfft(x - x.mean(dim=0), dim=0)
        amp = torch.abs(fft)
        power = amp.pow(2).mean(0)
        try:
            period = x.shape[0] // torch.argmax(amp[:, power.argmax()]).item()
        except Exception:
            period = 24
        return max(min(period, self.pred_len), 1)

    def _recalibrate(self, origin):
        """闭式重校准:模型在环三基向量,只用 origin 之前的真值"""
        if os.environ.get('RECYCLE_OUTAFF'):
            return  # OUTAFF 模式 = 纯输出空间仿射(MOS式)基线,Q 保持冻结
        n_est = max(self.hist // self.pred_len, 1)
        ows = [origin - self.pred_len * (t + 1) for t in range(n_est)]
        ows = [o for o in ows if o - self.seq_len >= 0]
        if len(ows) < 3:
            self.model.cycleQueue.data.copy_(self.Q0)
            return
        enc, enc_s, dec, dec_s = self._make_inputs(ows)
        inputs = (enc, enc_s, dec, dec_s)
        m = self.model
        omode = os.environ.get('RECYCLE_ORACLE')
        if omode == '2' and not hasattr(self, '_static_ab'):
            # D8:静态 oracle——用全部测试窗口拟一次全局 (alpha,beta),再统一应用
            stats = None
            i0 = 0
            while i0 + self.seq_len + self.pred_len <= self.T:
                o = i0 + self.seq_len
                enc_c, enc_s_c, dec_c, dec_s_c = self._make_inputs([o])
                yc = dec_c[:, -self.pred_len:, :].cuda()
                inp = (enc_c, enc_s_c, dec_c, dec_s_c)
                m.cycleQueue.data.copy_(torch.zeros_like(self.Q0)); fc0 = forecast(self.cfg, inp, m, self.norm_module)[0]
                m.cycleQueue.data.copy_(self.Q0);                    fcq = forecast(self.cfg, inp, m, self.norm_module)[0]
                m.cycleQueue.data.copy_(torch.ones_like(self.Q0));   fc1 = forecast(self.cfg, inp, m, self.norm_module)[0]
                gc, hc, rc = fcq - fc0, fc1 - fc0, yc - fc0
                s = (gc * gc).sum((0, 1)), (hc * hc).sum((0, 1)), (gc * hc).sum((0, 1)), \
                    (gc * rc).sum((0, 1)), (hc * rc).sum((0, 1))
                stats = [x.clone() if torch.is_tensor(x) else x for x in s] if stats is None else [a + b for a, b in zip(stats, s)]
                i0 += self.pred_len
            gg, hh, gh_, gr, hr = stats
            gg = gg + 1e-8
            det = gg * hh - gh_ * gh_ + 1e-8
            self._static_ab = (torch.clamp((gr * hh - gh_ * hr) / det, 0.5, 2.0),
                               torch.clamp((gg * hr - gh_ * gr) / det, -2.0, 2.0))
        if omode in ('1', '2'):
            if omode == '1':
                # 诊断用:直接在当前 chunk 上拟合(含泄漏),给出仿射周期子空间的上界
                enc_c, enc_s_c, dec_c, dec_s_c = self._make_inputs([origin])
                yc = dec_c[:, -self.pred_len:, :].cuda()
                inp = (enc_c, enc_s_c, dec_c, dec_s_c)
                m.cycleQueue.data.copy_(torch.zeros_like(self.Q0)); fc0 = forecast(self.cfg, inp, m, self.norm_module)[0]
                m.cycleQueue.data.copy_(self.Q0);                    fcq = forecast(self.cfg, inp, m, self.norm_module)[0]
                m.cycleQueue.data.copy_(torch.ones_like(self.Q0));   fc1 = forecast(self.cfg, inp, m, self.norm_module)[0]
                gc, hc, rc = fcq - fc0, fc1 - fc0, yc - fc0
                gg = (gc * gc).sum((0, 1)) + 1e-8; hh = (hc * hc).sum((0, 1))
                gh_ = (gc * hc).sum((0, 1)); gr = (gc * rc).sum((0, 1)); hr = (hc * rc).sum((0, 1))
                det = gg * hh - gh_ * gh_ + 1e-8
                al = (gr * hh - gh_ * hr) / det
                be = (gg * hr - gh_ * gr) / det
                al, be = torch.clamp(al, 0.5, 2.0), torch.clamp(be, -2.0, 2.0)
                if os.environ.get('RECYCLE_LOGAB') == '1':
                    with open(f'ab_oracle_{self.cfg.DATA.NAME}.csv', 'a') as f:
                        f.write(','.join([f'{v:.4f}' for v in al.cpu().numpy()] + [f'{v:.4f}' for v in be.cpu().numpy()]) + '\n')
            else:
                al, be = self._static_ab
            m.cycleQueue.data.copy_(al[None, :] * self.Q0 + be[None, :])
            return
        m.cycleQueue.data.copy_(torch.zeros_like(self.Q0)); f0 = forecast(self.cfg, inputs, m, self.norm_module)[0]
        m.cycleQueue.data.copy_(self.Q0);                    fq = forecast(self.cfg, inputs, m, self.norm_module)[0]
        m.cycleQueue.data.copy_(torch.ones_like(self.Q0));   f1 = forecast(self.cfg, inputs, m, self.norm_module)[0]
        y = dec[:, -self.pred_len:, :].cuda()
        g, h, r = fq - f0, f1 - f0, y - f0

        # 数据驱动Q:从估计窗口直接计算周期模式(不依赖骨干质量)
        if os.environ.get('RECYCLE_DATAQ') == '1' and len(ows) >= 3:
            C_ch = self.Q0.shape[1]
            W_cy = self.Q0.shape[0]
            # 收集所有估计窗口的数据,按周期位置平均
            all_data = []
            all_pos = []
            for o in ows:
                for t in range(self.seq_len):
                    abs_t = o - self.seq_len + t
                    all_data.append(self.test[abs_t])
                    all_pos.append(abs_t % W_cy)
                # 也加入预测目标的已揭示数据
                for t in range(min(self.pred_len, self.test.shape[0] - o)):
                    abs_t = o + t
                    if abs_t < self.test.shape[0]:
                        all_data.append(self.test[abs_t])
                        all_pos.append(abs_t % W_cy)
            if len(all_data) > W_cy:
                data_tensor = torch.stack(all_data)  # [N, C]
                pos_tensor = torch.tensor(all_pos, device='cuda')
                Q_data = torch.zeros(W_cy, C_ch, device='cuda')
                for p_idx in range(W_cy):
                    mask = pos_tensor == p_idx
                    if mask.sum() > 0:
                        Q_data[p_idx] = data_tensor[mask].mean(dim=0)
                Q_data = Q_data - Q_data.mean(dim=0, keepdim=True)
                # 用 Q_data 做基向量前向
                m.cycleQueue.data.copy_(Q_data)
                fqd = forecast(self.cfg, inputs, m, self.norm_module)[0]
                gd = fqd - f0  # 数据驱动Q的基向量
                # 将 Q_data 重校准作为额外候选
                # 简单1参数拟合: alpha_d = <gd, r> / <gd, gd>
                Sdd = (gd * gd).sum((0, 1)) + 1e-8
                Sdr = (gd * r).sum((0, 1))
                alpha_d = torch.clamp(Sdr / Sdd, 0.5, 2.0)
                # 添加为候选(在标准候选之后)
                gd_cands = [(0, gd, s_) for s_ in (0.0, 0.2, 1.0)]
                for _, gdv, ss in gd_cands:
                    lam = ss * Sdd
                    ad = (Sdr + lam) / (Sdd + lam)
                    ad = torch.clamp(ad, 0.5, 2.0)
                    # 输出: 重建的Q_data重校准预测
                    Q_recal = ad[None, :] * Q_data
                    m.cycleQueue.data.copy_(Q_recal)
                    fqd_r = forecast(self.cfg, inputs, m, self.norm_module)[0]
                    err_d = ((fqd_r[0] - y[0]) ** 2).mean(0)
                    # 存储Q_data候选的重建Q和误差
                    if not hasattr(self, '_dataq_cands'):
                        self._dataq_cands = []
                    self._dataq_cands.append((err_d, Q_recal, ad))
            m.cycleQueue.data.copy_(self.Q0)  # 恢复


        # v7:相位候选(RECYCLE_PHASE=1):对若干相位偏移 d,用 roll(Q0,d) 的基向量,
        # 每通道在验证窗上选最优 (d, alpha, beta)
        ph_on = os.environ.get('RECYCLE_PHASE') == '1'
        oracle = os.environ.get('RECYCLE_ORACLE') == '1'
        ph_cands = {}
        if ph_on:
            for d in (-4, -2, -1, 1, 2, 4):
                Qr = torch.roll(self.Q0, shifts=d, dims=0)
                m.cycleQueue.data.copy_(Qr)
                fqr = forecast(self.cfg, inputs, m, self.norm_module)[0]
                ph_cands[d] = fqr - f0
            m.cycleQueue.data.copy_(self.Q0)

        def fit(shrink, idx):
            gg = (g * g)[idx].sum((0, 1)) + 1e-8
            hh = (h * h)[idx].sum((0, 1))
            gh = (g * h)[idx].sum((0, 1))
            gr = (g * r)[idx].sum((0, 1))
            hr = (h * r)[idx].sum((0, 1))
            lam = shrink * gg
            det = (gg + lam) * (hh + lam) - gh * gh + 1e-8
            al = ((gr + lam) * (hh + lam) - gh * hr) / det
            be = ((gg + lam) * hr - gh * (gr + lam)) / det
            return torch.clamp(al, 0.5, 2.0), torch.clamp(be, -2.0, 2.0)

        # v14:分段Q基(变系数季节调整)——周期按位置分G组,逐通道逐组α_g+共享β,闭式LSQ
        slot = os.environ.get('RECYCLE_SLOT')
        if slot and len(ows) >= 3:
            G = max(int(slot), 2)
            if self.Q0.shape[0] % G != 0:
                G = 1
            if G > 1:
                Ds = []
                dev0 = self.Q0.device
                base_idx = torch.arange(self.Q0.shape[0], device=dev0)
                for gi in range(G):
                    mg = (base_idx // (self.Q0.shape[0] // G) == gi).float()[:, None]
                    m.cycleQueue.data.copy_(self.Q0 * mg)
                    Ds.append(forecast(self.cfg, inputs, m, self.norm_module)[0] - f0)
                m.cycleQueue.data.copy_(self.Q0)
                Dm = torch.stack(Ds + [h])                      # [G+1, n, H, C]
                Df = Dm.reshape(G + 1, -1, Csz := self.Q0.shape[1])
                rf = r.reshape(-1, Csz)
                Mm = torch.einsum('ktc,jtc->kjc', Df, Df)       # [G+1,G+1,C]
                vv = torch.einsum('ktc,tc->kc', Df, rf)         # [G+1,C]
                # 用旧窗拟,新窗验
                n_w = Df.shape[1] // self.pred_len if Df.shape[1] >= self.pred_len else 1
                cut = Df.shape[1] - self.pred_len
                Mm2 = torch.einsum('ktc,jtc->kjc', Df[:, cut:], Df[:, cut:])
                vv2 = torch.einsum('ktc,tc->kc', Df[:, cut:], rf[cut:])
                ths, e2s = [], []
                for sh2 in (0.2, 1.0, 5.0):
                    lam2 = sh2 * Mm2.diagonal(dim1=0, dim2=1).mean(-1)          # [C]
                    eye = torch.eye(G + 1, device=dev0)[..., None].expand(G + 1, G + 1, Csz)
                    A_ = (Mm2 + lam2[None, None, :] * eye).permute(2, 0, 1)     # [C,G+1,G+1]
                    b_ = vv2.T                                                   # [C,G+1]
                    th = torch.linalg.solve(A_, b_).T                            # [G+1,C]
                    e2 = ((f0[0][None] + torch.einsum('kc,khc->hc', th, Dm[:, 0]) - r[0]) ** 2).mean(0)
                    ths.append(torch.clamp(th, -2.0, 2.5)); e2s.append(e2)
                EE = torch.stack(e2s)                                            # [S,C]
                if EE.dim() == 3:
                    EE = EE.mean(1)
                TH = torch.stack(ths)                                            # [S,G+1,C]
                sel = EE.argmin(0)                                               # [C]
                best_th = TH.gather(0, sel[None, None].expand(1, G + 1, Csz)).squeeze(0)
                best_e = EE.min(0).values
                e_id2 = ((g[0] - r[0]) ** 2).mean(0)
                gm3 = best_e < e_id2 * (1.0 - self.gate)
                Qt = self.Q0.clone()
                aGs = torch.where(gm3[None, :], th[:G], torch.ones_like(th[:G]))
                bS = torch.where(gm3, th[G], torch.zeros_like(th[G]))
                for gi in range(G):
                    mg = (base_idx // (self.Q0.shape[0] // G) == gi).float()[:, None]
                    Qt = Qt + (aGs[gi][None, :] * (self.Q0 * mg) + bS[None, :] * mg - self.Q0 * mg)
                m.cycleQueue.data.copy_(Qt)
                return
        C = self.Q0.shape[1]
        dev = self.Q0.device
        best_err = None
        best_alpha = torch.ones(C, device=dev); best_beta = torch.zeros(C, device=dev)
        best_d = torch.zeros(C, dtype=torch.long, device=dev)
        cand_list = [(0, g, s) for s in (0.0, 0.2, 1.0, 5.0)]
        cand_list += [(d, gv, s) for d, gv in ph_cands.items() for s in (0.2, 1.0)]

        def fit_on(gv, shrink, idx):
            gg = (gv * gv)[idx].sum((0, 1)) + 1e-8
            hh = (h * h)[idx].sum((0, 1))
            gh_ = (gv * h)[idx].sum((0, 1))
            gr = (gv * r)[idx].sum((0, 1))
            hr = (h * r)[idx].sum((0, 1))
            lam = shrink * gg
            det = (gg + lam) * (hh + lam) - gh_ * gh_ + 1e-8
            al = ((gr + lam) * (hh + lam) - gh_ * hr) / det
            be = ((gg + lam) * hr - gh_ * (gr + lam)) / det
            return torch.clamp(al, 0.5, 2.0), torch.clamp(be, -2.0, 2.0)

        nv = min(int(os.environ.get('RECYCLE_NVAL', '1')), 2 if len(ows) >= 14 else 1)
        use_comb = os.environ.get('RECYCLE_COMB') == '1'
        errs, als, bes, ds = [], [], [], []
        fit_idx = torch.arange(nv, len(ows))
        vw = [0.7 ** t for t in range(nv)]
        for d_, gv, s_ in cand_list:
            al, be = fit_on(gv, s_, fit_idx)
            err = sum(w * (((al * gv[t] + be * h[t]) - r[t]) ** 2).mean(0) for t, w in enumerate(vw)) / sum(vw)
            errs.append(err); als.append(al); bes.append(be); ds.append(d_)
            if not use_comb:
                if best_err is None:
                    best_err = err.clone()
                    best_alpha, best_beta, best_d = al, be, torch.full_like(best_d, d_)
                else:
                    msk = err < best_err
                    best_alpha = torch.where(msk, al, best_alpha)
                    best_beta = torch.where(msk, be, best_beta)
                    best_d = torch.where(msk, torch.full_like(best_d, d_), best_d)
                    best_err = torch.where(msk, err, best_err)
        if not use_comb:
            pass  # non-comb path handled above
        # 如果有数据驱动Q的候选,加入候选池
        if hasattr(self, '_dataq_cands') and self._dataq_cands:
            for err_d, Q_recal, ad in self._dataq_cands[-3:]:  # 最多3个Q_data候选
                errs.append(err_d)
                als.append(ad)  # alpha from Q_data fit
                bes.append(torch.zeros_like(err_d))  # no beta for Q_data
                ds.append(0)  # same phase
            self._dataq_cands = []  # 清空(下次chunk重新计算)
        # 季节朴素候选:直接复制上周数据(不依赖骨干,强周期数据的天然强基线)
        if os.environ.get('RECYCLE_SEASONAL') == '1' and len(ows) >= 3:
            W_cy = self.Q0.shape[0]
            # 估计窗口上的朴素预测: data[t-W] 作为 t 的预测
            # 用最新估计窗口验证
            ow0 = ows[0]  # 最新窗口起点
            naive_pred = self.test[ow0 - W_cy:ow0 - W_cy + self.pred_len]  # [H, C]
            naive_gt = self.test[ow0:ow0 + self.pred_len]  # [H, C] 已揭示
            err_naive = ((naive_pred - naive_gt) ** 2).mean(0)  # [C]
            # 加入候选: 用alpha=0,beta=0(恒等)+朴素修正
            # 实际做法:直接把朴素预测作为额外候选
            if not hasattr(self, '_naive_preds'):
                self._naive_preds = []
            # 计算朴素预测的"alpha"使其适配我们的框架
            # 简化:加入一个 shrink=0 的Q重校准,其效果等价于朴素
            # 实际上最简单的做法是把朴素预测的误差加入候选池
            # 用alpha=1,beta=0但预测来自朴素
            errs.append(err_naive)
            als.append(torch.ones_like(err_naive))
            bes.append(torch.zeros_like(err_naive))
            ds.append(0)
            # 存储朴素预测(在最终应用时用)
            origin_naive = self._current_origin if hasattr(self, '_current_origin') else ow0
            self._naive_preds.append((err_naive, naive_pred))

        err_id = ((g[0] - r[0]) ** 2).mean(0)
        if use_comb:
            # v10/v11:在线组合替代硬选择(Bates-Granger/Hedge),含恒等候选;
            # v11:跨通道部分池化候选(James-Stein 式,利用跨通道漂移相关性)
            errs.append(err_id); als.append(torch.ones_like(err_id)); bes.append(torch.zeros_like(err_id)); ds.append(0)
            if os.environ.get('RECYCLE_POOL') == '1':
                for k in range(len(als) - 1):
                    a_pool = als[k].median() if os.environ.get('RECYCLE_RPOOL') == '1' else als[k].mean()
                    b_pool = bes[k].median() if os.environ.get('RECYCLE_RPOOL') == '1' else bes[k].mean()
                    for kap in (0.25, 0.5, 0.75, 0.9):
                        al2 = kap * als[k] + (1 - kap) * a_pool
                        be2 = kap * bes[k] + (1 - kap) * b_pool
                        e2 = (((al2 * (g if ds[k] == 0 else ph_cands[ds[k]])[0] + be2 * h[0]) - r[0]) ** 2).mean(0)
                        errs.append(e2); als.append(al2); bes.append(be2); ds.append(ds[k])
            E = torch.stack(errs)                                    # [K, C]
            if not hasattr(self, '_cw'):
                self._cw = torch.zeros_like(E)
            rho = float(os.environ.get('RECYCLE_COMBRHO', '0.9'))
            eta = float(os.environ.get('RECYCLE_COMBETA', '20.0'))
            self._cw = rho * self._cw + E
            w = torch.softmax(-eta * self._cw / self._cw.mean(dim=0, keepdim=True).clamp_min(1e-8), dim=0)  # [K,C] 逐通道证据归一
            if os.environ.get('RECYCLE_WSTAT') == '1':
                if not hasattr(self, '_wid_sum'):
                    self._wid_sum, self._wid_n = 0.0, 0
                self._wid_sum += w[-1].mean().item()
                self._wid_n += 1
            A = torch.stack(als); B = torch.stack(bes); D = torch.tensor(ds, device=dev)
            best_alpha = (w * A).sum(0); best_beta = (w * B).sum(0)
            wd = torch.zeros_like(best_alpha)
            for kk in torch.unique(D):
                mskk = D == kk
                wd += w[mskk].sum(0) * float(kk)
            best_d = torch.round(wd).long()                          # 加权相位(四舍五入到主导相位)
            gmask = (E.argmin(0) == len(errs) - 1)                   # 仅当恒等为历史最优时保持恒等
            best_alpha = torch.where(gmask, torch.ones_like(best_alpha), best_alpha)
            best_beta = torch.where(gmask, torch.zeros_like(best_beta), best_beta)
            best_d = torch.where(gmask, torch.zeros_like(best_d), best_d)
        else:
            gmask = best_err < err_id * (1.0 - self.gate)
            best_alpha = torch.where(gmask, best_alpha, torch.ones_like(best_alpha))
            best_beta = torch.where(gmask, best_beta, torch.zeros_like(best_beta))
            best_d = torch.where(gmask, best_d, torch.zeros_like(best_d))
        if os.environ.get('RECYCLE_DUMP') == '1':
            if not hasattr(self, '_ab_stat'):
                self._ab_stat, self._ab_n, self._ab_rows = 0.0, 0, []
            self._ab_stat += (best_alpha - 1).abs().mean().item()
            self._ab_n += 1
            self._ab_rows.append(','.join([f'{v:.4f}' for v in best_alpha.cpu().numpy()]))
        # 逐相位组应用
        Qt = self.Q0.clone()
        for d_ in torch.unique(best_d):
            msk = (best_d == d_)
            if int(d_) != 0:
                Qr = torch.roll(self.Q0, int(d_), 0)
                Qt[:, msk] = best_alpha[msk][None, :] * Qr[:, msk] + best_beta[msk][None, :]
            else:
                Qt[:, msk] = best_alpha[msk][None, :] * self.Q0[:, msk] + best_beta[msk][None, :]
        m.cycleQueue.data.copy_(Qt)

    def _grad_backbone(self, origin):
        """混合模式:骨干权重上的 TAFAS 式持续梯度适应(周期分量保持 Q0)。
        RECYCLE_GGATE=1 时逐块验证门控:步后估计窗损失变差则回滚本步。"""
        n_est = max(self.hist // self.pred_len, 1)
        ows = [origin - self.pred_len * (t + 1) for t in range(n_est)]
        ows = [o for o in ows if o - self.seq_len >= 0]
        if not ows:
            return
        enc, enc_s, dec, dec_s = self._make_inputs(ows)
        y = dec[:, -self.pred_len:, :].cuda()
        self.model.cycleQueue.data.copy_(self.Q0)
        gate = os.environ.get('RECYCLE_GGATE') == '1'
        select_lr = os.environ.get('RECYCLE_GSELECT') == '1'
        heldout = os.environ.get('RECYCLE_GHELD') == '1'
        if (heldout or select_lr) and len(ows) >= 3:
            # 梯度只用旧窗 ows[1:],验证/选择在最新窗 ows[0] 上
            enc_t, enc_s_t, dec_t, dec_s_t = self._make_inputs([ows[0]])
            y_t = dec_t[:, -self.pred_len:, :].cuda()
            genc, genc_s, gdec, gdec_s = self._make_inputs(ows[1:])
            gy = gdec[:, -self.pred_len:, :].cuda()
        else:
            heldout = False
            select_lr = False
            genc, genc_s, gdec, gdec_s, gy = enc, enc_s, dec, dec_s, y
        if select_lr:
            base = deepcopy(self.model.model.state_dict())
            best = None
            for lr in (0.0, 0.001, 0.0001):
                self.model.model.load_state_dict(base)
                if lr > 0:
                    for g in self.bopt.param_groups:
                        g['lr'] = lr
                    with torch.enable_grad():
                        for _ in range(self.gsteps):
                            self.bopt.zero_grad()
                            pred = forecast(self.cfg, (genc, genc_s, gdec, gdec_s), self.model, self.norm_module)[0]
                            F.mse_loss(pred, gy).backward()
                            self.bopt.step()
                with torch.no_grad():
                    e = F.mse_loss(forecast(self.cfg, (enc_t, enc_s_t, dec_t, dec_s_t), self.model, self.norm_module)[0], y_t).item()
                if best is None or e < best[0]:
                    best = (e, deepcopy(self.model.model.state_dict()))
            self.model.model.load_state_dict(best[1])
            return
        if gate:
            sd = deepcopy(self.model.model.state_dict())
            vsrc = y_t if heldout else y
            vinput = (enc_t, enc_s_t, dec_t, dec_s_t) if heldout else (enc, enc_s, dec, dec_s)
            with torch.no_grad():
                pre = F.mse_loss(forecast(self.cfg, vinput, self.model, self.norm_module)[0], vsrc).item()
        with torch.enable_grad():
            for step_i in range(self.gsteps):
                if step_i > 0:
                    # 第二步:仅更新cycleQueue(梯度版闭式精修,温和)
                    lr2 = float(os.environ.get('RECYCLE_BLR2', '0.003'))
                    cq_params = [v for k, v in self.model.named_parameters() if k.startswith('cycleQueue')]
                    opt2 = torch.optim.Adam(cq_params, lr=lr2)
                    self.model.cycleQueue.data.copy_(self.Q0)  # 从源周期出发
                    opt2.zero_grad()
                    pred2 = forecast(self.cfg, (genc, genc_s, gdec, gdec_s), self.model, self.norm_module)[0]
                    F.mse_loss(pred2, gy).backward()
                    opt2.step()
                    break  # 第二步不需要继续循环
                self.bopt.zero_grad()
                pred = forecast(self.cfg, (genc, genc_s, gdec, gdec_s), self.model, self.norm_module)[0]
                loss = F.mse_loss(pred, gy)
                loss.backward()
                self.bopt.step()
        if gate:
            with torch.no_grad():
                post = F.mse_loss(forecast(self.cfg, vinput, self.model, self.norm_module)[0], vsrc).item()
            if post > pre:
                self.model.model.load_state_dict(sd)
        # v12:CoTTA式随机恢复——每块以概率 p 把部分权重拉回源,防误差累积
        if os.environ.get('RECYCLE_SRESTORE'):
            p_r = float(os.environ['RECYCLE_SRESTORE'])
            with torch.no_grad():
                cur = self.model.state_dict()
                for k in self.W0:
                    msk = torch.rand_like(cur[k]) < p_r
                    cur[k][msk] = self.W0[k][msk]

    # ---------- 主循环 ----------
    def adapt(self):
        period = self._fft_period()
        chunk = period + 1
        mse_all, mae_all = [], []
        if os.environ.get('RECYCLE_OUTAFF') == 'static':
            # 静态输出仿射:全测试集拟一次全局 (a,b)
            self.model.cycleQueue.data.copy_(self.Q0)
            P = Y = None
            i = 0
            with torch.no_grad():
                while i < self.n:
                    b_end = min(i + chunk, self.n)
                    origins = list(range(i + self.seq_len, b_end + self.seq_len))
                    enc, enc_s, dec, dec_s = self._make_inputs(origins)
                    p = forecast(self.cfg, (enc, enc_s, dec, dec_s), self.model, self.norm_module)[0]
                    g = dec[:, -self.pred_len:, :].cuda()
                    P = p.reshape(-1, p.shape[-1]).T if P is None else torch.cat([P, p.reshape(-1, p.shape[-1]).T], dim=1)
                    Y = g.reshape(-1, g.shape[-1]).T if Y is None else torch.cat([Y, g.reshape(-1, g.shape[-1]).T], dim=1)
                    i = b_end
            pc = P - P.mean(1, keepdim=True); yc = Y - Y.mean(1, keepdim=True)
            a = (pc * yc).sum(1) / (pc * pc).sum(1).clamp_min(1e-8)
            b = Y.mean(1) - a * P.mean(1)
            self._outab = (torch.clamp(a, 0.2, 3.0), torch.clamp(b, -3.0, 3.0))
        t_upd, n_adapt = 0.0, 0
        dual = os.environ.get('RECYCLE_DUAL') == '1'
        triple = dual and os.environ.get('RECYCLE_TRIPLE') == '1'
        if dual:
            def _bb_sd(m):
                return {k: v for k, v in m.state_dict().items() if not k.startswith('cycleQueue')}
            self._B_pure = _bb_sd(self.model)
            self._B_grad = _bb_sd(self.model)
            self._macc = None
            self._last = None
            _bb_params = [v for k, v in self.model.named_parameters() if not k.startswith('cycleQueue')]
            self.bopt = torch.optim.Adam(_bb_params, lr=float(os.environ.get('RECYCLE_BLR', '0.001')))
            self.W0 = _bb_sd(self.model)
            self._bb_load = lambda m, sd: m.load_state_dict(sd, strict=False)
        if triple:
            Cw = self.Q0.shape[1]
            self._io = torch.nn.ModuleDict({'in': torch.nn.Linear(self.seq_len, self.seq_len),
                                            'out': torch.nn.Linear(self.pred_len, self.pred_len)}).cuda()
            self._io_delta = float(os.environ.get('RECYCLE_IOD', '0.1'))
            self._io_opt = torch.optim.Adam(self._io.parameters(), lr=float(os.environ.get('RECYCLE_IOLR', '0.001')))
        i = 0
        with torch.no_grad():
            while i < self.n:
                b_end = min(i + chunk, self.n)
                origin = i + self.seq_len
                if dual:
                    # v13 三路径:纯闭式 / 梯度+闭式 / 纯梯度
                    if self._last is not None:
                        pps, gt_last = self._last
                        E2 = torch.stack([((p - gt_last) ** 2).mean(0) for p in pps])
                        self._macc = 0.9 * self._macc + E2 if self._macc is not None else E2
                        w = torch.softmax(-float(os.environ.get('RECYCLE_METAETA', '20.0')) * self._macc
                                          / self._macc.mean().clamp_min(1e-8), 0)
                    else:
                        w = None
                    origins = list(range(i + self.seq_len, b_end + self.seq_len))
                    enc, enc_s, dec, dec_s = self._make_inputs(origins)
                    inputs = (enc, enc_s, dec, dec_s)
                    t0 = time.perf_counter()
                    if origin >= self.seq_len + self.pred_len + 2 * self.cycle:
                        # 路径B:梯度步 + 随机恢复(不闭式重校准)
                        self._bb_load(self.model, self._B_grad)
                        self._grad_backbone(origin)
                        if os.environ.get('RECYCLE_SRESTORE'):
                            p_r = float(os.environ['RECYCLE_SRESTORE'])
                            with torch.no_grad():
                                cur2 = self.model.state_dict()
                                for k in self.W0:
                                    msk = torch.rand_like(cur2[k]) < p_r
                                    cur2[k][msk] = self.W0[k][msk]
                        predC = forecast(self.cfg, inputs, self.model, self.norm_module)[0]  # 纯梯度
                        # 路径B+:梯度 + 闭式重校准
                        self._recalibrate(origin)
                        predB = forecast(self.cfg, inputs, self.model, self.norm_module)[0]
                        self._B_grad = deepcopy({k: v for k, v in self.model.state_dict().items() if not k.startswith('cycleQueue')})
                        # 路径A:纯闭式(恢复纯骨干)
                        self._bb_load(self.model, self._B_pure)
                        self._recalibrate(origin)
                        predA = forecast(self.cfg, inputs, self.model, self.norm_module)[0]
                        t_upd += time.perf_counter() - t0
                        n_adapt += 1
                        if self._macc is None:
                            o0 = origin - self.pred_len
                            enc0, enc_s0, dec0, dec_s0 = self._make_inputs([o0])
                            y0 = dec0[:, -self.pred_len:, :].cuda()
                            self._bb_load(self.model, self._B_pure)
                            pA0 = forecast(self.cfg, (enc0, enc_s0, dec0, dec_s0), self.model, self.norm_module)[0]
                            self._macc = 3.0 * torch.stack([
                                ((pA0 - y0) ** 2).mean((0, 1)),
                                ((predB[0] - y0[0]) ** 2).mean(0)])
                        self._bb_load(self.model, self._B_grad)  # 恢复梯度骨干状态
                    else:
                        self.model.cycleQueue.data.copy_(self.Q0)
                        predA = predB = predC = forecast(self.cfg, inputs, self.model, self.norm_module)[0]
                    gt = dec[:, -self.pred_len:, :].cuda()
                    # 自适应触发器:前10块评估闭式改善,不足3%则启用梯度
                    if not hasattr(self, '_use_grad'):
                        self._use_grad = None
                        self._van_err_sum = 0.0
                        self._c_err_sum = 0.0
                        self._trigger_chunks = 0
                    if self._use_grad is None:
                        # 热身期:用C+vanilla对比
                        with torch.no_grad():
                            self.model.cycleQueue.data.copy_(self.Q0)
                            self._bb_load(self.model, self._B_pure)
                            van_pred = forecast(self.cfg, inputs, self.model, self.norm_module)[0]
                            van_err = F.mse_loss(van_pred, dec[:, -self.pred_len:, :].cuda()).item()
                            c_err = F.mse_loss(predA, dec[:, -self.pred_len:, :].cuda()).item()
                        self._van_err_sum += van_err
                        self._c_err_sum += c_err
                        self._trigger_chunks += 1
                        if self._trigger_chunks >= 20:
                            improvement = 1 - self._c_err_sum / max(self._van_err_sum, 1e-8)
                            self._use_grad = improvement < 0.00  # 无任何改善→梯度
                            print(f'[trigger] C improvement={improvement*100:.1f}% → gradient={"ON" if self._use_grad else "OFF"}')
                    # 根据触发器决定输出
                    if self._use_grad:
                        pred = predB  # 梯度+闭式
                    else:
                        pred = predA  # 纯闭式
                    self._last = ((predA[0], predB[0]), gt[0])
                    mse = F.mse_loss(pred, gt, reduction='none').mean(dim=(-2, -1)).reshape(-1).detach().cpu().numpy()
                    mae = F.l1_loss(pred, gt, reduction='none').mean(dim=(-2, -1)).reshape(-1).detach().cpu().numpy()
                    mse_all.append(mse)
                    mae_all.append(mae)
                    i = b_end
                    continue

                    if w is None:
                        pred = (predA + predB + predC_grad) / 3
                    else:
                        pred = (w[0][None, None, :] * predA + w[1][None, None, :] * predB
                                + w[2][None, None, :] * predC_grad)
                    self._last = ((predA[0], predB[0], predC_grad[0]) + ((predC[0],) if predC is not None else ()), gt[0])
                    mse = F.mse_loss(pred, gt, reduction='none').mean(dim=(-2, -1)).reshape(-1).detach().cpu().numpy()
                    mae = F.l1_loss(pred, gt, reduction='none').mean(dim=(-2, -1)).reshape(-1).detach().cpu().numpy()
                    mse_all.append(mse)
                    mae_all.append(mae)
                    i = b_end
                    continue
                if origin >= self.seq_len + self.pred_len + 2 * self.cycle:
                    t0 = time.perf_counter()
                    if self.hybrid:
                        self._grad_backbone(origin)
                    self._recalibrate(origin)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t_upd += time.perf_counter() - t0
                    n_adapt += 1
                else:
                    self.model.cycleQueue.data.copy_(self.Q0)
                origins = list(range(i + self.seq_len, b_end + self.seq_len))
                enc, enc_s, dec, dec_s = self._make_inputs(origins)
                if os.environ.get('RECYCLE_FROZENREF') == '1':
                    # 冻结参照:同块在 Q=Q0 下的预测与误差(chunk级漂移检测器图用)
                    q_keep = self.model.cycleQueue.data.clone()
                    self.model.cycleQueue.data.copy_(self.Q0)
                    f_pred = forecast(self.cfg, (enc, enc_s, dec, dec_s), self.model, self.norm_module)[0]
                    self.model.cycleQueue.data.copy_(q_keep)
                    f_gt = dec[:, -self.pred_len:, :].cuda()
                    fm = F.mse_loss(f_pred, f_gt, reduction='none').mean(dim=(0, 1)).detach().cpu().numpy()
                    self._frozen_mse = getattr(self, '_frozen_mse', []) + [fm]
                    self._frozen_preds = getattr(self, '_frozen_preds', []) + [f_pred[0].detach().cpu().numpy()]
                pred = forecast(self.cfg, (enc, enc_s, dec, dec_s), self.model, self.norm_module)[0]
                oa = os.environ.get('RECYCLE_OUTAFF')
                if oa:  # D10:输出仿射 pred -> a*pred + b(逐通道)
                    gt0 = dec[:, -self.pred_len:, :].cuda()
                    if oa == 'dyn':          # oracle:当前块拟合(泄漏上界)
                        a = ((pred - pred.mean(1, keepdim=True)) * (gt0 - gt0.mean(1, keepdim=True))).mean(1) / \
                            ((pred - pred.mean(1, keepdim=True)) ** 2).mean(1).clamp_min(1e-8)
                        b = (gt0 - a[:, None, :] * pred).mean(1)
                        pred = a[:, None, :] * pred + b[:, None, :]
                    elif oa == 'causal' and origin >= self.seq_len + self.pred_len + self.cycle:
                        # v8:因果输出仿射——估计窗(严格过去)LS + 收缩候选 + 不伤害门控
                        n_est = max(self.hist // self.pred_len, 1)
                        ows = [origin - self.pred_len * (t + 1) for t in range(n_est)]
                        ows = [o for o in ows if o - self.seq_len >= 0]
                        enc_e, enc_s_e, dec_e, dec_s_e = self._make_inputs(ows)
                        pe = forecast(self.cfg, (enc_e, enc_s_e, dec_e, dec_s_e), self.model, self.norm_module)[0]
                        ge = dec_e[:, -self.pred_len:, :].cuda()
                        pe = pe.reshape(-1, pe.shape[-1]); ge = ge.reshape(-1, ge.shape[-1])
                        pm, gm = pe.mean(0), ge.mean(0)
                        pc = pe - pm; gc2 = ge - gm
                        Spp = (pc * pc).sum(0) + 1e-8
                        Spg = (pc * gc2).sum(0)
                        best_e, av, bv = None, None, None
                        for sh in (0.0, 0.2, 1.0, 5.0):
                            lam = sh * Spp
                            a2 = (Spg + lam) / (Spp + lam)
                            b2 = gm - a2 * pm
                            err = ((a2[None, :] * pe[: self.pred_len] + b2[None, :] - ge[: self.pred_len]) ** 2).mean(0)
                            if best_e is None:
                                best_e, av, bv = err.clone(), a2.clone(), b2.clone()
                            else:
                                m2 = err < best_e
                                av = torch.where(m2, a2, av); bv = torch.where(m2, b2, bv)
                                best_e = torch.where(m2, err, best_e)
                        av = torch.clamp(av, 0.3, 2.5); bv = torch.clamp(bv, -2.0, 2.0)
                        err_id_o = ((pe[: self.pred_len] - ge[: self.pred_len]) ** 2).mean(0)
                        gm2 = best_e < err_id_o * (1.0 - self.gate)
                        av = torch.where(gm2, av, torch.ones_like(av))
                        bv = torch.where(gm2, bv, torch.zeros_like(bv))
                        pred = av[None, None, :] * pred + bv[None, None, :]
                    elif oa == 'static' and getattr(self, '_outab', None) is not None:
                        a, b = self._outab
                        pred = a[None, None, :] * pred + b[None, None, :]
                if os.environ.get('RECYCLE_BIAS') == '1' and origin >= self.seq_len + self.pred_len + self.cycle:
                    # 快通道:闭式积分校正 pred += gamma * 近期残差均值(逐通道)
                    enc_p, enc_s_p, dec_p, dec_s_p = self._make_inputs([origin - self.pred_len])
                    with torch.no_grad():
                        pp = forecast(self.cfg, (enc_p, enc_s_p, dec_p, dec_s_p), self.model, self.norm_module)[0]
                    bias = (pp - dec_p[:, -self.pred_len:, :].cuda()).mean(dim=1)        # [C]
                    pred = pred - float(os.environ.get('RECYCLE_BGAMMA', '1.0')) * bias[None, None, :]
                gt = dec[:, -self.pred_len:, :].cuda()
                # R19: horizon 分层偏置追踪(闭式)——吸收窗口间持续的同 horizon 偏置
                if os.environ.get('RECYCLE_AR') == '1':
                    if not hasattr(self, '_ewb'):
                        self._ewb = None          # [H, C] 指数加权偏置
                        self._gcnt = torch.zeros(1).cuda()  # 门控通过计数(平滑)
                        self._pend = None
                    if self._pend is not None and self._pend[0] <= origin:
                        o_prev, p_prev = self._pend
                        g_prev = self.test[o_prev:o_prev + self.pred_len]     # [H, C] 已揭示真值
                        b_prev = (p_prev - g_prev)
                        # 门控:EWB 在该窗口上是否降误差(逐通道)
                        if self._ewb is not None:
                            e_wo = (p_prev - g_prev).pow(2).mean(0)            # [C]
                            e_w = (p_prev + self._ewb - g_prev).pow(2).mean(0)
                            gm = (e_w < e_wo).float()
                            self._gcnt = 0.8 * self._gcnt + 0.2 * gm.mean()
                        else:
                            gm = torch.ones_like(b_prev[0])
                        self._ewb = b_prev if self._ewb is None else 0.5 * self._ewb + 0.5 * b_prev
                        self._pend = None
                    if origin >= self.seq_len + self.pred_len + 2 * self.cycle:
                        self._pend = (origin, pred[0].detach())
                    if self._ewb is not None and self._gcnt.item() > 0.3:
                        pred = pred + self._ewb[None, :, :] * float(os.environ.get('RECYCLE_ARGAMMA', '0.8'))
                # 残差EMA修正:在Q重校准的残差上做(比D9更稳定因为残差已去结构漂移)
                if os.environ.get('RECYCLE_RESIDUAL') == '1':
                    if not hasattr(self, '_res_ema'):
                        self._res_ema = None
                    # 在估计窗上更新EMA
                    if origin >= self.seq_len + self.pred_len + 2 * self.cycle:
                        n_est = max(self.hist // self.pred_len, 1)
                        ows_r = [origin - self.pred_len * (t + 1) for t in range(n_est)]
                        ows_r = [o for o in ows_r if o - self.seq_len >= 0]
                        if ows_r:
                            rin = self._make_inputs(ows_r)
                            with torch.no_grad():
                                self._bb_load(self.model, {k: v for k, v in self.model.state_dict().items()})
                                rp = forecast(self.cfg, rin, self.model, self.norm_module)[0]
                            rgt = rin[2][:, -self.pred_len:, :].cuda()
                            res = (rp - rgt).mean(dim=(0, 1))  # [C]
                            if self._res_ema is None:
                                self._res_ema = res
                            else:
                                self._res_ema = 0.8 * self._res_ema + 0.2 * res
                    # 应用修正
                    if self._res_ema is not None:
                        pred = pred - self._res_ema[None, None, :]
                
                # 上下文感知输出修正:用当前lookback统计量预测应有水平,闭式OLS
                if os.environ.get('RECYCLE_CTX') == '1' and origin >= self.seq_len + self.pred_len + 2 * self.cycle:
                    if not hasattr(self, '_ctx_w'):
                        self._ctx_w = None
                        self._ctx_b = None
                        self._ctx_ready = False
                    # 1. 在估计窗上拟: gt_mean ≈ w·lookback_mean + b (逐通道闭式OLS)
                    n_est = max(self.hist // self.pred_len, 1)
                    ows_c = [origin - self.pred_len * (t + 1) for t in range(n_est)]
                    ows_c = [o for o in ows_c if o - self.seq_len >= 0]
                    if len(ows_c) >= 5:
                        einp = self._make_inputs(ows_c)
                        e_enc = einp[0]  # [n_est, L, C]
                        e_gt = einp[2][:, -self.pred_len:, :].cuda()  # [n_est, H, C]
                        # lookback均值和gt均值(逐窗口逐通道)
                        lm = e_enc.mean(dim=1)  # [n_est, C]
                        gm = e_gt.mean(dim=1)   # [n_est, C]
                        # 闭式OLS: gt_mean ≈ w·lookback_mean + b
                        lm_bar = lm.mean(dim=0)  # [C]
                        gm_bar = gm.mean(dim=0)  # [C]
                        lm_c = lm - lm_bar
                        gm_c = gm - gm_bar
                        self._ctx_w = (lm_c * gm_c).sum(dim=0) / ((lm_c * lm_c).sum(dim=0) + 1e-8)
                        self._ctx_b = gm_bar - self._ctx_w * lm_bar
                        self._ctx_ready = True
                    # 2. 用当前lookback统计量预测应有水平,修正预测
                    if self._ctx_ready:
                        current_ctx = enc.mean(dim=(0, 1))  # [C] 当前lookback均值
                        target_level = self._ctx_w * current_ctx + self._ctx_b  # [C]
                        pred_level = pred.mean(dim=1, keepdim=True)  # [B, 1, C]
                        # 平滑修正(不猛拉,用50%的修正量避免过冲)
                        correction = 0.5 * (target_level[None, None, :] - pred_level)
                        pred = pred + correction
                mse = F.mse_loss(pred, gt, reduction='none').mean(dim=(-2, -1)).reshape(-1).detach().cpu().numpy()
                mae = F.l1_loss(pred, gt, reduction='none').mean(dim=(-2, -1)).reshape(-1).detach().cpu().numpy()
                mse_all.append(mse)
                mae_all.append(mae)
                i = b_end
        mse_all = np.concatenate(mse_all)
        mae_all = np.concatenate(mae_all)
        assert len(mse_all) == self.n, (len(mse_all), self.n)
        if os.environ.get('RECYCLE_WSTAT') == '1' and getattr(self, '_wid_n', 0):
            print(f'identity-weight(按兵不动占比): {self._wid_sum / self._wid_n:.3f}')
        if os.environ.get('RECYCLE_DUMP') == '1':
            print(f'mean|alpha-1|: {self._ab_stat / max(self._ab_n, 1):.4f}')
            with open(f'alpha_traj_{self.cfg.DATA.NAME}.csv', 'w') as f:
                f.write('\n'.join(self._ab_rows))
            np.save(f'Qt_final_{self.cfg.DATA.NAME}.npy', self.model.cycleQueue.data.cpu().numpy())
            np.save(f'Q0_{self.cfg.DATA.NAME}.npy', self.Q0.cpu().numpy())
        print('After TSF-TTA of ReCycle (closed-form)')
        print(f'Number of adaptations: {n_adapt}')
        print(f'Avg update time: {t_upd / max(n_adapt, 1) * 1000:.2f} ms')
        print(f'Test MSE: {mse_all.mean():.4f}, Test MAE: {mae_all.mean():.4f}')
        if os.environ.get('RECYCLE_FROZENREF') == '1' and hasattr(self, '_frozen_mse'):
            np.save(f'chunk_mse_{self.cfg.DATA.NAME}.npy', np.stack(mse_all))
            np.save(f'frozen_mse_{self.cfg.DATA.NAME}.npy', np.array(self._frozen_mse))
            np.save(f'frozen_pred_{self.cfg.DATA.NAME}.npy', np.stack(self._frozen_preds))
            print(f'saved chunk_mse/frozen_mse npy ({len(mse_all)} chunks)')


def build_recycle_adapter(cfg, model, norm_module=None):
    return RecycleAdapter(cfg, model, norm_module=norm_module)
