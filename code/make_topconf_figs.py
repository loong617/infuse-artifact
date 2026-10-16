# -*- coding: utf-8 -*-
"""Generate top-conference-style Motivation Figure (figs/fig_teaser.pdf) and compile the TikZ Framework Figure
(figs/framework.pdf)."""
import json
import os
import subprocess
from collections import defaultdict

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

import inflight as I
import make_paper_assets as A

PAPER = '../paper/'
plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage[T1]{fontenc}\usepackage{libertine}\usepackage[libertine]{newtxmath}',
    'font.family': 'serif', 'font.size': 7.5, 'axes.labelsize': 7.5, 'axes.titlesize': 7.8,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 6.3,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.3, 'ytick.major.size': 2.3, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'axes.grid.axis': 'y', 'grid.linewidth': 0.35, 'grid.color': '#E2E2E2',
    'axes.axisbelow': True, 'figure.dpi': 300, 'savefig.dpi': 300, 'pdf.fonttype': 42,
})

C_OURS = '#D55E00'      # vermilion (INFUSE)
C_STACK = '#8C2D04'     # deep crimson-brown (Adapter + INFUSE)
C_REF = '#E69F00'       # amber (Refresh)
C_BLUE = '#0072B2'      # ocean blue (revealed / covariance)
C_GREY = '#6E6E6E'      # slate grey (frozen)
C_TTA = {'TAFAS': '#56B4E9', 'PETSA': '#0072B2', 'COSA': '#009E73', 'DADAPTER': '#CC79A7'}


def make_teaser(M):
    """Three-panel motivation figure:
    (a) Clean, uncluttered paradigm comparison on a real in-flight trajectory with exact vertical arrows;
    (b) Disentangling parameter adaptation vs. in-flight revision across all 72 settings;
    (c) Accuracy gain vs. per-batch runtime (GPU adapters vs. 1-CPU-thread INFUSE)."""
    rows = list(M.values())

    def red(c):
        v = [-A.rel(r[c][0], r['frozen'][0]) for r in rows if c in r]
        return float(np.mean(v)) if v else np.nan

    fig = plt.figure(figsize=(3.42, 3.35))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.10, 1.05], width_ratios=[1.12, 0.96],
                  hspace=0.46, wspace=0.34, left=0.10, right=0.985, top=0.915, bottom=0.105)

    # ================================================================ Panel (a): In-flight trajectory
    ax = fig.add_subplot(gs[0, :])
    path = 'dumps/fm/bolt-small_azure_functions_96.npz'
    i, ch, k, x0 = 3010, 32, 24, -28
    y, pred, gt, E, L, H = I.load(path)
    C = pred.shape[2]
    cov = I.ErrorCov(H, C)
    cov.add_many(E[:i - H + 1])
    Sig = cov.sigma()
    L_mat, U, csum_U2 = I.chol_factors(Sig, k)
    from scipy.linalg import solve_triangular
    nu = solve_triangular(L_mat, E[i, :k, ch], lower=True)
    upd = pred[i, k:, ch] + U[:, k:].T @ nu
    sc = float(np.sqrt(cov.ss[ch] / max(cov.wc, 1.0)))
    std_p = sc * np.sqrt(np.maximum(np.diag(Sig)[k:] - csum_U2[k - 1, k:], 1e-12))
    ref = pred[i + k, :H - k, ch]
    obs = y[i + L + x0:i + L + H, ch]

    # Clean background zones: Lookback [-28, 0], Revealed [0, 24], Pending [24, 95]
    ax.axvspan(x0, 0, color='#F5F6F8', lw=0, zorder=0)
    ax.axvspan(0, k, color='#E4F0F8', lw=0, zorder=0)
    ax.axvspan(k, H - 1, color='#FFF9F5', lw=0, zorder=0)

    # Shaded gap between observed y and issued forecast on [0, k], and posterior 1-sigma band on [k, H-1]
    t_rev = np.arange(0, k + 1)
    ax.fill_between(t_rev, pred[i, :k + 1, ch], y[i + L:i + L + k + 1, ch],
                    color=C_BLUE, alpha=0.17, lw=0, zorder=1)
    ax.fill_between(np.arange(k, H), upd - 0.5 * std_p, upd + 0.5 * std_p,
                    color=C_OURS, alpha=0.15, lw=0, zorder=2)

    l_obs, = ax.plot(np.arange(x0, H), obs, color='#1A1A1A', lw=0.95, zorder=3)
    l_iss, = ax.plot(np.arange(H), pred[i, :, ch], color=C_GREY, lw=1.1, ls=(0, (3, 1.5)), zorder=4)
    l_ref, = ax.plot(np.arange(k, H), ref, color=C_REF, lw=1.15, ls=(0, (4, 1.2)), zorder=4)
    l_upd, = ax.plot(np.arange(k, H), upd, color=C_OURS, lw=1.55, zorder=5)

    ax.axvline(0, color='#4D4D4D', lw=0.7, ls='--')
    ax.axvline(k, color=C_OURS, lw=0.8, ls='--')

    lo_, hi_ = -0.85, 6.85
    ax.set_ylim(lo_, hi_)
    ax.set_xlim(x0, H - 1)

    # Top zone pills
    y_ban = 6.22
    ax.text(0.5 * x0, y_ban, 'Lookback', ha='center', va='center', fontsize=5.7, color='#555555',
            bbox=dict(boxstyle='round,pad=0.16', fc='white', ec='#CCCCCC', lw=0.4))
    ax.text(0.5 * k, y_ban, r'Revealed ($1{:}k$)', ha='center', va='center', fontsize=5.7, color=C_BLUE,
            bbox=dict(boxstyle='round,pad=0.16', fc='white', ec=C_BLUE, lw=0.5))
    ax.text(0.5 * (k + H - 1), y_ban, r'Pending horizon ($k{+}1{:}H$ in flight)', ha='center', va='center',
            fontsize=5.7, color=C_OURS, bbox=dict(boxstyle='round,pad=0.16', fc='white', ec=C_OURS, lw=0.5))

    # 1. Exact vertical double-arrow inside the revealed zone at x=8 (from issued curve to observed curve)
    xr = 8
    y_iss_r = float(pred[i, xr, ch])
    y_obs_r = float(obs[xr - x0])
    ax.annotate('', xy=(xr, y_obs_r - 0.08), xytext=(xr, y_iss_r + 0.08),
                arrowprops=dict(arrowstyle='<->', color=C_BLUE, lw=0.9, shrinkA=0, shrinkB=0), zorder=6)
    ax.text(xr + 1.5, 0.5 * (y_iss_r + y_obs_r) - 0.35, r'$e_{\mathrm r}$', fontsize=6.3, color=C_BLUE,
            ha='left', va='center', bbox=dict(boxstyle='round,pad=0.12', fc='white', ec=C_BLUE, lw=0.4, alpha=0.92))

    # 2. Exact vertical upward arrow in the pending zone at x=61 (from issued dashed curve to INFUSE-U curve)
    xp = 61
    y_iss_p = float(pred[i, xp, ch])
    y_upd_p = float(upd[xp - k])
    ax.annotate('', xy=(xp, y_upd_p - 0.06), xytext=(xp, y_iss_p + 0.06),
                arrowprops=dict(arrowstyle='->', color=C_OURS, lw=1.1, shrinkA=0, shrinkB=0), zorder=6)
    ax.text(xp + 2.0, 0.5 * (y_iss_p + y_upd_p) - 0.15, r'$+\hat e_{\mathrm p}=U_{1:k}^\top\nu_o$',
            fontsize=5.8, color=C_OURS, ha='left', va='center',
            bbox=dict(boxstyle='round,pad=0.14', fc='white', ec=C_OURS, lw=0.45, alpha=0.95))

    ax.set_xticks([-24, 0, 24, 48, 72, 95])
    ax.set_xticklabels([r'$-24$', r'$0$ (issued)', r'$24$ (now $t$)', r'$48$', r'$72$', r'$96$'])
    ax.set_xlabel('Time steps relative to forecast origin $o$ (minutes)', labelpad=1.5)
    ax.set_ylabel('Load (std.)', labelpad=1.5)
    ax.set_title(r'\textbf{(a)} Revising an in-flight cloud workload forecast as observations arrive',
                 loc='left', pad=13, fontsize=7.5)
    ax.legend([l_obs, l_iss, l_ref, l_upd],
              ['Observed $y$', r'Issued $\hat y_o$', r'Refresh $\hat y_t$ (1 call)',
               r'\textsc{Infuse-U} (0 call)'],
              loc='lower left', bbox_to_anchor=(-0.01, 1.005), ncol=4, fontsize=5.7,
              handlelength=1.35, columnspacing=0.65, handletextpad=0.35, borderaxespad=0.1, frameon=False)

    # ================================================================ Panel (b): Adaptation vs. Revision
    ax = fig.add_subplot(gs[1, 0])
    x, w_bar = 0.0, 0.36
    ticks, labels = [], []
    for t in A.TTA:
        a, b = red(t), red(t + '-noadj')
        ax.bar(x - w_bar / 2, a, w_bar, color=C_TTA[t], edgecolor=C_TTA[t], lw=0.4)
        ax.bar(x + w_bar / 2, b, w_bar, color='white', edgecolor=C_TTA[t], hatch='/////', lw=0.6)
        ticks.append(x); labels.append(t)
        x += 0.95
    if not np.isnan(red('DADAPTER')):
        v = red('DADAPTER')
        ax.bar(x, v, 0.48, color=C_TTA['DADAPTER'], lw=0)
        ax.text(x, v + 0.3, '%.1f' % v, ha='center', va='bottom', fontsize=5.2, color='#444444')
        ticks.append(x); labels.append(r'$\delta$-Ad.')
        x += 0.95
    x += 0.15
    ax.axvline(x - 0.55, color='#CCCCCC', lw=0.6, ls='--')
    for c, lab, col, ht in (('refresh', 'Refresh', C_REF, False),
                            ('update', r'\textsc{Infuse-U}', C_OURS, True),
                            ('fuse', r'\textsc{Infuse}', C_OURS, False),
                            ('dadapter_fuse', r'$\delta$+\textsc{Inf.}', C_STACK, False)):
        v = red(c)
        if np.isnan(v):
            continue
        ax.bar(x, v, 0.54, color='white' if ht else col, edgecolor=col, hatch='/////' if ht else None, lw=0.7)
        ax.text(x, v + 0.3, '%.1f' % v, ha='center', va='bottom', fontsize=5.3,
                fontweight='bold' if c in ('fuse', 'dadapter_fuse') else 'normal',
                color=C_OURS if c == 'fuse' else (C_STACK if c == 'dadapter_fuse' else '#333333'))
        ticks.append(x); labels.append(lab)
        x += 0.92

    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=32, ha='right', fontsize=5.6, rotation_mode='anchor')
    ax.tick_params(axis='x', length=0, pad=1)
    ax.set_ylabel(r'MSE reduction (\%)', labelpad=1.5)
    top_y = max(red('dadapter_fuse'), red('fuse'), 12) * 1.24
    ax.set_ylim(0, top_y)
    ax.legend(handles=[Patch(facecolor='#7F7F7F', lw=0, label='Official'),
                       Patch(facecolor='white', edgecolor='#7F7F7F', hatch='/////', lw=0.6, label='No revision / 0-call')],
              loc='upper left', fontsize=5.4, handlelength=1.1, borderaxespad=0.2, handletextpad=0.35, frameon=False)
    ax.set_title(r'\textbf{(b)} Adaptation vs.\ in-flight revision', loc='left', fontsize=7.3, pad=3)

    # ================================================================ Panel (c): Accuracy vs. Runtime
    ax = fig.add_subplot(gs[1, 1])
    T = json.load(open('timing.json'))['rows'] if os.path.exists('timing.json') else []
    TH = json.load(open('timing_h.json')) if os.path.exists('timing_h.json') else {}
    smap = {'iTrans. ETTh1': 'iTransformer_ETTh1_96', 'iTrans. Weather': 'iTransformer_weather_96',
            'PatchTST ETTm1': 'PatchTST_ETTm1_96'}
    tm = defaultdict(list)
    for r_ in T:
        t_h = TH.get(smap.get(r_['setting'], ''))
        if t_h and r_['method'] in ('INFUSE-U', 'INFUSE'):
            tm[r_['method']].append(t_h['update_ms'] if r_['method'] == 'INFUSE-U' else t_h['fuse_ms'])
        else:
            tm[r_['method']].append(r_['ms'])

    ax.axvspan(15, 135, ymin=0.0, ymax=0.35, color='#F0F3F7', lw=0, zorder=0)
    ax.text(132, 6.4, 'Gradient TTA (GPU)', fontsize=4.8, color='#555555', ha='right', va='bottom')

    pts = [('TAFAS', 'TAFAS', C_TTA['TAFAS'], 'o', (-5, 2), 'right'),
           ('PETSA', 'PETSA', C_TTA['PETSA'], 's', (5, 3), 'left'),
           ('COSA', 'COSA', C_TTA['COSA'], '^', (5, -3), 'left'),
           ('update', 'INFUSE-U', C_OURS, 'D', (0, -8), 'center'),
           ('fuse', 'INFUSE', C_OURS, '*', (5, 1), 'left')]
    coords = {}
    for c, key, col, mk, off, ha in pts:
        if key not in tm or np.isnan(red(c)):
            continue
        xx, yy = float(np.exp(np.mean(np.log(tm[key])))), red(c)
        coords[c] = (xx, yy)
        ax.scatter(xx, yy, s=55 if mk == '*' else 24, marker=mk, color=col,
                   edgecolor='white' if mk != '*' else col, lw=0.4, zorder=4)
        lab = {'update': r'\textsc{Infuse-U}', 'fuse': r'\textbf{\textsc{Infuse}}'}.get(c, c)
        ax.annotate(lab, (xx, yy), xytext=off, textcoords='offset points', fontsize=5.6,
                    color=C_OURS if c in ('update', 'fuse') else '#333333', va='center', ha=ha)

    if 'TAFAS' in coords and 'fuse' in coords:
        x1, y1 = coords['TAFAS']
        x2, y2 = coords['fuse']
        ax.annotate('', xy=(x2 * 1.15, y2 - 0.15), xytext=(x1 * 0.78, y1 + 0.15),
                    arrowprops=dict(arrowstyle='->', color=C_OURS, lw=0.75, ls=(0, (2.5, 1.2)),
                                    connectionstyle='arc3,rad=-0.16'), zorder=3)
        ax.text(28, 10.8, (r'\textbf{%.0f$\times$ faster}' % (x1 / max(x2, 1e-6))) + '\n' +
                (r'\textbf{$-$%.1f\%% MSE}' % (y2 - y1)),
                fontsize=5.3, color=C_OURS, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.15', fc='#FFF5EE', ec=C_OURS, lw=0.4))

    ax.set_xscale('log')
    ax.set_xlim(0.7, 140)
    ax.set_ylim(0, top_y)
    ax.set_xlabel('Per-batch runtime (ms, log)', labelpad=1.5)
    ax.set_ylabel(r'MSE reduction (\%)', labelpad=1.5)
    ax.grid(axis='x', which='major')
    ax.set_title(r'\textbf{(c)} Accuracy vs.\ runtime', loc='left', fontsize=7.3, pad=3)

    fig.savefig(PAPER + 'figs/fig_teaser.pdf')
    plt.close(fig)
    print('saved figs/fig_teaser.pdf')


def _draw_framework(fig, usetex):
    """Draw the 4-stage visual pipeline onto `fig`. When usetex=False, uses Unicode/mathtext strings so SVG exports
    preserve every label as a native editable <text> element."""
    from matplotlib.patches import FancyArrowPatch
    from scipy.linalg import solve_triangular

    path = 'dumps/grid/iTransformer_ETTh1_96.npz'
    y, pred, gt, E, L, H = I.load(path)
    n, _, C = pred.shape
    cov = I.ErrorCov(H, C)
    cov.add_many(E[:1500])
    Sig = cov.sigma()
    d = np.diag(Sig)
    k, k0 = 12, 24
    L_mat, U, csum_U2 = I.chol_factors(Sig, k0)

    # Clear, noise-free diurnal trajectory illustrating each stage of the mathematical formulation
    t_all = np.arange(H, dtype=np.float64)
    obs_y = 0.85 * np.sin(2 * np.pi * (t_all - 3) / 24.0) + 0.25 * np.cos(2 * np.pi * t_all / 12.0) + 0.55
    iss_y = 0.58 * np.sin(2 * np.pi * (t_all - 6) / 24.0) + 0.18 * np.cos(2 * np.pi * t_all / 12.0) - 0.35
    decay_k = np.exp(-np.arange(H - k) / 26.0)
    traj_upd = iss_y[k:] + (obs_y[k:] - iss_y[k:]) * (0.42 + 0.38 * decay_k)
    cone_upd = obs_y[k:] - 0.22 * (obs_y[k:] - iss_y[k:]) * (1.0 - 0.55 * decay_k)
    ref_y = obs_y[k:] + 0.26 * np.sin(2 * np.pi * np.arange(H - k) / 24.0 + 0.7) + 0.16
    v_cone = np.maximum(d[k:] - csum_U2[k0 - 1, k:], 1e-12)
    v_new = d[:H - k]
    v_irr_f = I.GAMMA * np.minimum(v_cone, v_new)
    tvu, tvn = np.maximum(v_cone - v_irr_f, 1e-12), np.maximum(v_new - v_irr_f, 1e-12)
    wf = tvu / (tvu + tvn)
    fused_y = (1 - wf) * cone_upd + wf * ref_y
    v_post = v_irr_f + (tvu * tvn) / (tvu + tvn)
    std_cone = 0.42 * np.sqrt(v_cone)
    std_post = 0.42 * np.sqrt(v_post)

    gs = GridSpec(2, 4, figure=fig, width_ratios=[1.14, 1.00, 1.18, 1.18], height_ratios=[1.0, 1.0],
                  wspace=0.27, hspace=0.34, left=0.015, right=0.99, top=0.73, bottom=0.16)

    if usetex:
        stage_titles = [
            (0.015, 0.245, '#2E3440', r'\textbf{In-Flight Cone at Time $t$}',
             r'Staggered origins $\{t{-}h_b,\,t{-}k_0,\,t{-}k\}$'),
            (0.280, 0.485, C_BLUE, r'\textbf{(1) Progressive Cov.\ \& Cholesky}',
             r'$\tilde\Sigma_{1:K}=LL^\top,\ \ U=L^{-1}\tilde\Sigma$'),
            (0.520, 0.745, C_OURS, r'\textbf{(2) Cone Revision (\textsc{Infuse-U})}',
             r'$\hat e_{\mathrm p}=U_{1:k}^\top\nu_o,\ \ \Delta V(k_0\!\to\! k,\ell)>0$'),
            (0.775, 0.990, C_STACK, r'\textbf{(3) Excess-Precision Fusion (\textsc{Infuse})}',
             r'$w_m \propto (V_m - v_{\mathrm{irr}})^{-1},\ \ v_{\mathrm{irr}}=\gamma\min V$'),
        ]
    else:
        stage_titles = [
            (0.015, 0.245, '#2E3440', 'In-Flight Cone at Time t',
             'Staggered origins {t - h_b, t - k_0, t - k}'),
            (0.280, 0.485, C_BLUE, '(1) Progressive Cov. & Cholesky',
             'Sigma_1:K = L L^T,   U = L^-1 Sigma'),
            (0.520, 0.745, C_OURS, '(2) Cone Revision (INFUSE-U)',
             'e_p = U_1:k^T nu_o,   Delta V(k_0 -> k, l) > 0'),
            (0.775, 0.990, C_STACK, '(3) Excess-Precision Fusion (INFUSE)',
             'w_m ~ (V_m - v_irr)^-1,   v_irr = gamma min V'),
        ]
    for x0_s, x1_s, col, title, sub in stage_titles:
        xc = 0.5 * (x0_s + x1_s)
        fig.text(xc, 0.94, title, ha='center', va='center', fontsize=7.4,
                 fontweight='bold' if not usetex else 'normal', color=col)
        fig.text(xc, 0.855, sub, ha='center', va='center', fontsize=6.0, color='#444444')
        fig.lines.append(plt.Line2D([x0_s, x1_s], [0.81, 0.81], transform=fig.transFigure, color=col, lw=0.9))

    for xa in (0.256, 0.496, 0.753):
        fig.patches.append(FancyArrowPatch((xa - 0.008, 0.45), (xa + 0.014, 0.45),
                                           transform=fig.transFigure, arrowstyle='->,head_width=3.2,head_length=3.8',
                                           color='#555555', lw=1.0))

    # ================================================================ Stage 1: Staggered In-Flight Cone
    ax_s1 = fig.add_subplot(gs[:, 0])
    ax_s1.set_xlim(-40, 54)
    ax_s1.set_ylim(-0.35, 3.55)
    ax_s1.axis('off')

    ax_s1.axvline(0, color=C_OURS, lw=0.9, ls='--')
    ax_s1.text(0, 3.38, r'\textbf{now $t$}' if usetex else 'now t', ha='center', va='bottom',
               fontsize=6.2, fontweight='bold' if not usetex else 'normal', color=C_OURS)
    ax_s1.axvspan(-40, 0, color='#F2F6FA', lw=0, zorder=0)
    ax_s1.axvspan(0, 54, color='#FFF8F3', lw=0, zorder=0)
    ax_s1.text(-20, 3.12, 'Revealed ($<t$)' if usetex else 'Revealed (< t)', ha='center', va='center', fontsize=5.5, color=C_BLUE)
    ax_s1.text(27, 3.12, r'Pending ($\ge t$)' if usetex else 'Pending (>= t)', ha='center', va='center', fontsize=5.5, color=C_OURS)

    t_wave = np.linspace(-37, 0, 50)
    ax_s1.plot(t_wave, 2.42 + 0.16 * np.sin(t_wave * 0.25) + 0.06 * np.cos(t_wave * 0.55), color='#1A1A1A', lw=1.0)
    ax_s1.text(-37, 2.75, r'Stream $y_{<t}$' if usetex else 'Stream y_<t', fontsize=5.5, color='#1A1A1A', ha='left')

    ax_s1.plot([-36, -2], [1.82, 1.82], color=C_BLUE, lw=4.2, solid_capstyle='butt')
    ax_s1.text(-19, 2.08, r'Prefix $e_{o^{\prime}}(1{:}h_b)$' if usetex else 'Prefix e_o\'(1:h_b)',
               ha='center', fontsize=5.5, color=C_BLUE)

    ax_s1.plot([-24, 0], [1.04, 1.04], color=C_BLUE, lw=4.2, solid_capstyle='butt')
    ax_s1.plot([0, 36], [1.04, 1.04], color='#F3B68B', lw=4.2, solid_capstyle='butt')
    ax_s1.text(6, 1.30, r'Anchor $\hat y_{t-k_0}$ ($k_0$ rev.)' if usetex else 'Anchor y_t-k0 (k0 rev.)',
               ha='center', fontsize=5.5, color='#222222')

    ax_s1.plot([-9, 0], [0.26, 0.26], color=C_BLUE, lw=4.2, solid_capstyle='butt')
    ax_s1.plot([0, 48], [0.26, 0.26], color=C_OURS, lw=4.2, solid_capstyle='butt')
    ax_s1.text(-5, 0.52, r'$e_{\mathrm r}$' if usetex else 'e_r', ha='center', fontsize=5.6, color=C_BLUE, fontweight='bold')
    ax_s1.text(24, 0.52, r'Active $\hat y_{t-k}$ ($\hat y_{\mathrm p}$)' if usetex else 'Active y_t-k (pending)',
               ha='center', fontsize=5.5, color=C_OURS)
    ax_s1.text(7, -0.22, r'Co-target cone for $y_{t+\ell-1}$' if usetex else 'Co-target cone for y_t+l-1',
               ha='center', fontsize=5.4, color='#444444')

    # ================================================================ Stage 2: Progressive Cov & Cholesky Factor
    ax_cov = fig.add_subplot(gs[0, 1])
    sd_diag = np.sqrt(d)
    Corr = Sig / sd_diag[:, None] / sd_diag[None, :]
    ax_cov.imshow(Corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto', interpolation='nearest')
    ax_cov.set_xticks([0, 23, 95]); ax_cov.set_xticklabels(['1', '24', '$H$' if usetex else 'H'], fontsize=5.2)
    ax_cov.set_yticks([0, 95]); ax_cov.set_yticklabels(['1', '$H$' if usetex else 'H'], fontsize=5.2)
    ax_cov.tick_params(length=1.5, pad=1)
    for sp in ax_cov.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.4)
    ax_cov.grid(False)
    ax_cov.set_title(r'Pooled $\tilde\Sigma^{(h_b)}$ (24h bands)' if usetex else 'Pooled Sigma^(h_b) (24h bands)',
                     fontsize=5.6, pad=2, loc='center')

    ax_u = fig.add_subplot(gs[1, 1])
    leads_all = np.arange(1, H + 1)
    ax_u.plot(leads_all, U[0], color=C_BLUE, lw=0.95, label='$u_1$ (level)' if usetex else 'u_1 (level)')
    ax_u.plot(leads_all, U[min(11, k0 - 1)], color=C_OURS, lw=0.85, alpha=0.85,
              label='$u_{12}$ (diurnal)' if usetex else 'u_12 (diurnal)')
    ax_u.axhline(0, color='#888888', lw=0.4)
    ax_u.set_xlim(1, H)
    ax_u.set_xticks([1, 24, 48, 96]); ax_u.set_yticks([])
    ax_u.tick_params(labelsize=5.2, length=1.5, pad=1)
    ax_u.set_xlabel('Lead $j$' if usetex else 'Lead j', fontsize=5.8, labelpad=1)
    ax_u.legend(loc='upper right', fontsize=4.8, handlelength=1.0, borderaxespad=0.1, frameon=False)
    ax_u.set_title(r'Basis $U=L^{-1}\tilde\Sigma$' if usetex else 'Basis U = L^-1 Sigma',
                   fontsize=5.6, pad=1.5, loc='center')

    # ================================================================ Stage 3: Cone Revision (INFUSE-U)
    ax_s3 = fig.add_subplot(gs[:, 2])
    t_grid = np.arange(1, H + 1)
    ax_s3.axvspan(1, k, color='#EBF3FA', lw=0, zorder=0)
    ax_s3.plot(t_grid, obs_y, color='#1A1A1A', lw=0.85, alpha=0.88, label='Observed $y$' if usetex else 'Observed y')
    ax_s3.plot(t_grid, iss_y, color=C_GREY, lw=1.0, ls=(0, (3, 1.5)), label=r'Issued $\hat y_{t-k}$' if usetex else 'Issued y_t-k')
    ax_s3.plot(t_grid[k:], traj_upd, color=C_BLUE, lw=1.05, ls=':', label=r'Single-origin $\hat y^{\mathrm{traj}}$' if usetex else 'Single-origin y^traj')
    ax_s3.plot(t_grid[k:], cone_upd, color=C_OURS, lw=1.5, label=r'Cone $\tilde y_o$ (\textsc{Infuse-U})' if usetex else 'Cone y_o (INFUSE-U)')
    ax_s3.fill_between(t_grid[k:], cone_upd - 0.35 * std_cone, cone_upd + 0.35 * std_cone,
                       color=C_OURS, alpha=0.16, lw=0)
    ax_s3.axvline(k, color=C_OURS, lw=0.65, ls='--')
    ax_s3.set_xlim(1, H)
    ax_s3.set_xticks([1, k, 48, 96]); ax_s3.set_xticklabels(['1', '$k$' if usetex else 'k', '48', '$H$' if usetex else 'H'], fontsize=5.5)
    ax_s3.set_yticks([])
    ax_s3.set_xlabel('Lead $j$ (0 model calls)' if usetex else 'Lead j (0 model calls)', fontsize=6.0, labelpad=1.5)
    lo3, hi3 = ax_s3.get_ylim()
    ax_s3.set_ylim(lo3, hi3 + 0.28 * (hi3 - lo3))
    ax_s3.legend(loc='upper right', ncol=2, fontsize=4.9, handlelength=1.2, columnspacing=0.5,
                 handletextpad=0.3, borderaxespad=0.15, frameon=False)

    # ================================================================ Stage 4: Excess-Precision Fusion & Serving
    ax_s4 = fig.add_subplot(gs[:, 3])
    t_pend = np.arange(1, H - k + 1)
    ax_s4.plot(t_pend, obs_y[k:], color='#1A1A1A', lw=0.85, alpha=0.88, label='Observed $y$' if usetex else 'Observed y')
    ax_s4.plot(t_pend, ref_y, color=C_REF, lw=1.0, ls=(0, (3.5, 1.2)), label=r'Fresh $\hat y_t$ ($r$-step call)' if usetex else 'Fresh y_t (r-step call)')
    ax_s4.plot(t_pend, cone_upd, color=C_OURS, lw=1.0, ls=':', label=r'Cone $\tilde y_o$' if usetex else 'Cone y_o')
    ax_s4.plot(t_pend, fused_y, color=C_STACK, lw=1.55, label=r'Served $\bar y_o$ (\textsc{Infuse})' if usetex else 'Served y_o (INFUSE)')
    ax_s4.fill_between(t_pend, fused_y - 0.35 * std_post, fused_y + 0.35 * std_post,
                       color=C_STACK, alpha=0.16, lw=0, label=r'Margin $\pm s_{\mathrm{post}}$' if usetex else 'Margin +- s_post')
    ax_s4.set_xlim(1, H - k)
    ax_s4.set_xticks([1, 24, 48, H - k]); ax_s4.set_xticklabels(['1', '24', '48', '$H{-}k$' if usetex else 'H-k'], fontsize=5.5)
    ax_s4.set_yticks([])
    ax_s4.set_xlabel(r'Pending step $\ell$ $\to$ Web Autoscaler' if usetex else 'Pending step l -> Web Autoscaler',
                     fontsize=6.0, labelpad=1.5)
    lo4, hi4 = ax_s4.get_ylim()
    ax_s4.set_ylim(lo4, hi4 + 0.28 * (hi4 - lo4))
    ax_s4.legend(loc='upper right', ncol=2, fontsize=4.9, handlelength=1.2, columnspacing=0.5,
                 handletextpad=0.3, borderaxespad=0.15, frameon=False)


def make_framework():
    """Generate both paper/figs/framework.pdf (LaTeX typeset) and a genuinely editable paper/figs/framework.svg
    (with native <text> elements, not glyph outlines)."""
    plt.rcParams['text.usetex'] = True
    fig = plt.figure(figsize=(7.15, 1.95))
    _draw_framework(fig, usetex=True)
    fig.savefig(PAPER + 'figs/framework.pdf')
    plt.close(fig)

    plt.rcParams['text.usetex'] = False
    plt.rcParams['svg.fonttype'] = 'none'
    fig_svg = plt.figure(figsize=(7.15, 1.95))
    _draw_framework(fig_svg, usetex=False)
    fig_svg.savefig(PAPER + 'figs/framework.svg')
    plt.close(fig_svg)
    plt.rcParams['text.usetex'] = True
    print('saved figs/framework.pdf and editable figs/framework.svg')


if __name__ == '__main__':
    make_teaser(A.merged())
    make_framework()
