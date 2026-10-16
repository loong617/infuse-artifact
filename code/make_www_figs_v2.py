# -*- coding: utf-8 -*-
"""WWW 2027 figures, v2.

One figure = one PDF (panels drawn together, shared legends), rendered with LaTeX in
Linux Libertine so that figure text matches the acmart body font. Each figure is drawn
at its final printed width (ACM sigconf: column 3.33in, text 7.0in), so LaTeX includes
it at 100% and font sizes are true point sizes.

Data sources are unchanged from make_www_figs.py (same files, same computations).
"""
import datetime
import json

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

OUT = '../paper/figs'
COL = 3.33
TXT = 7.0

plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage[T1]{fontenc}\usepackage{libertine}'
                           r'\usepackage[libertine]{newtxmath}',
    'font.family': 'serif',
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'axes.linewidth': 0.6, 'lines.linewidth': 1.2,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'xtick.minor.size': 1.5, 'ytick.minor.size': 1.5,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'axes.grid.axis': 'y', 'grid.linewidth': 0.4,
    'grid.color': '#D9D9D9', 'grid.linestyle': '-', 'axes.axisbelow': True,
    'axes.titlelocation': 'left', 'axes.titlepad': 4,
    'legend.frameon': False, 'legend.handlelength': 1.6, 'legend.handletextpad': 0.4,
    'legend.columnspacing': 1.0, 'legend.borderaxespad': 0.2,
    'figure.dpi': 300, 'savefig.dpi': 300, 'pdf.fonttype': 42,
    'figure.constrained_layout.h_pad': 0.02, 'figure.constrained_layout.w_pad': 0.02,
    'figure.constrained_layout.hspace': 0.04, 'figure.constrained_layout.wspace': 0.04,
})

# Okabe-Ito, colour-blind safe; one meaning per colour across the paper
OURS = '#D55E00'
FROZEN = '#7F7F7F'
BLUE = '#0072B2'
PINK = '#CC79A7'
GREEN = '#009E73'
DS6 = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#D55E00', '#56B4E9']
MK6 = ['o', 's', '^', 'D', 'v', 'P']
DSLAB = ['ETTh2', 'ETTm2', 'Electricity', 'Traffic', 'PEMS03', 'PEMS04']
DSKEY = ['ETTh2', 'ETTm2', 'electricity', 'traffic', 'PEMS03', 'PEMS04']


def title(ax, tag, text):
    ax.set_title(r'\textbf{(%s)} %s' % (tag, text))


def save(fig, name):
    fig.savefig('%s/%s.pdf' % (OUT, name), facecolor='white')
    plt.close(fig)
    print('saved', name)


# ======================= Figure 1: one deployment (text width) =======================
CH, W = 119, 168
rows = open('../data/electricity/electricity.csv').read().splitlines()[1:]
d0 = datetime.datetime.strptime(rows[0].split(',')[0], '%Y-%m-%d %H:%M:%S')
ho0 = d0.weekday() * 24 + d0.hour
Xv = np.genfromtxt('../data/electricity/electricity.csv', delimiter=',', skip_header=1)[:, 1:]
n_tot = len(Xv)
ho = (np.arange(n_tot) + ho0) % W
tr_idx = np.arange(0, int(n_tot * 0.7))
te_idx = np.arange(int(n_tot * 0.8), n_tot)
ec_tr = np.array([Xv[tr_idx, CH][ho[tr_idx] == p].mean() for p in range(W)])
ec_te = np.array([Xv[te_idx, CH][ho[te_idx] == p].mean() for p in range(W)])
print('fig1a: ch%d amplitude %.0f -> %.0f (%.0f%% drop)' % (
    CH, np.ptp(ec_tr), np.ptp(ec_te), 100 * (1 - np.ptp(ec_te) / np.ptp(ec_tr))))

fm = np.load('frozen_mse_electricity.npy').mean(axis=1)
rm = np.load('chunk_mse_electricity.npy')
nb = min(len(fm), len(rm) // 25)
rm_ch = rm[:nb * 25].reshape(nb, 25).mean(axis=1)
A119 = np.loadtxt('alpha_traj_electricity.csv', delimiter=',')[:, CH]
n_chunks = min(len(A119), nb)

fig, axs = plt.subplots(1, 3, figsize=(TXT, 1.55), layout='constrained')
ax = axs[0]
ax.fill_between(range(W), ec_tr / 1e3, ec_te / 1e3, color=OURS, alpha=0.15, lw=0)
ax.plot(range(W), ec_tr / 1e3, color=FROZEN, lw=1.1, label='Training period')
ax.plot(range(W), ec_te / 1e3, color=BLUE, lw=1.1, label='Test period')
for d in range(1, 7):
    ax.axvline(d * 24, color='#BDBDBD', lw=0.3, zorder=0)
ax.set_xlim(0, W - 1)
ax.set_xticks([12 + 24 * d for d in range(7)])
ax.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
ax.tick_params(axis='x', length=0)
ax.set_ylabel(r'Load ($\times 10^3$)')
ax.set_ylim(10.3, 12.75)
ax.set_yticks([10.5, 11.0, 11.5, 12.0])
ax.grid(False)
ax.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.02))
title(ax, 'a', 'Average weekly cycle of channel %d' % CH)

ax = axs[1]
gap = np.cumsum(fm[:n_chunks] - rm_ch[:n_chunks]) * 25
ax.fill_between(range(n_chunks), 0, gap, color=OURS, alpha=0.15, lw=0)
ax.plot(range(n_chunks), gap, color=OURS, lw=1.5)
ax.set_xlim(0, n_chunks - 1)
ax.set_ylim(0, None)
ax.set_xlabel('Test chunk')
ax.set_ylabel('Cumulative excess MSE')
title(ax, 'b', 'Frozen minus adapted error, accumulated')

ax = axs[2]
k = 9
rsm = np.convolve(rm_ch[:n_chunks], np.ones(k) / k, mode='valid')
fsm = np.convolve(fm[:n_chunks], np.ones(k) / k, mode='valid')
xs = np.arange(k - 1, n_chunks)
ax.fill_between(xs, rsm, fsm, where=fsm >= rsm, color=OURS, alpha=0.15, lw=0)
ax.plot(xs, fsm, color=FROZEN, lw=1.1, label='Frozen')
ax.plot(xs, rsm, color=OURS, lw=1.5, label=r'\textsc{ReCycle}')
ax.set_xlim(0, n_chunks - 1)
ax.set_xlabel('Test chunk')
ax.set_ylabel('MSE (9-chunk mean)')
ax.legend(loc='upper right', ncol=2)
title(ax, 'c', 'Per-chunk error, frozen vs.\\ adapted')
for a in axs[1:]:
    a.set_xticks([0, 50, 100, 150])
save(fig, 'fig1_overview')

# ======================= Sensitivity (column width) =======================
SENS = json.load(open('subexp_final.json'))
fig, axs = plt.subplots(1, 2, figsize=(COL, 1.62), layout='constrained', sharey=True)
for ax, key, xs_, dflt, xl, tg, tt in [
        (axs[0], 'eta', [5, 50, 200, 500], 200, r'Hedge rate $\eta$', 'a', r'Hedge rate'),
        (axs[1], 'hist', [5, 15, 30], 15, 'Estimation windows', 'b', 'History length')]:
    for kname, lab, c, m in zip(DSKEY, DSLAB, DS6, MK6):
        ys = SENS[key][kname]
        best = min(ys)
        pen = [(y - best) / best * 100 for y in ys]
        ax.plot(xs_, pen, marker=m, color=c, lw=1.0, ms=3.2, mfc='white', mec=c, mew=0.8,
                label=lab)
    ax.axvline(dflt, color='#4D4D4D', lw=0.7, ls=(0, (3, 2)), zorder=0)
    ax.set_xscale('log')
    ax.set_xticks(xs_)
    ax.set_xticklabels([str(v) for v in xs_])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis='x', which='minor', length=0)
    ax.set_xlabel(xl)
    ax.set_ylim(-0.15, 2.6)
    title(ax, tg, tt)
axs[0].set_xlim(3.8, 700)
axs[1].set_xlim(4.2, 36)
axs[0].set_ylabel(r'MSE penalty (\%)')
axs[0].text(200 * 1.08, 2.45, 'default', fontsize=6.5, color='#4D4D4D', va='top')
axs[1].text(15 * 1.06, 2.45, 'default', fontsize=6.5, color='#4D4D4D', va='top')
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc='outside upper center', ncol=6, fontsize=6.5, columnspacing=0.6,
           handlelength=1.2, handletextpad=0.3)
save(fig, 'fig_sensitivity')

# ======================= Oracle accounting + quadratic objective (column) ===========
frozen_v = [.3005, .1778, .2430, .8647, .5030, .5052]
causal_v = [.2710, .1702, .2221, .7998, .4503, .4764]
static_v = [.2629, .1719, .2216, .7996, .4359, .4628]
closed = [(f - c) / (f - s) * 100 for f, c, s in zip(frozen_v, causal_v, static_v)]
print('gap closed %:', [round(c, 1) for c in closed], 'median %.1f' % np.median(closed))

D = np.load('basis_ch117_elec.npz')
Sgg, Shh, Sgh = float(D['Sgg']), float(D['Shh']), float(D['Sgh'])
Sgr, Shr, Srr = float(D['Sgr']), float(D['Shr']), float(D['Srr'])
Qm = np.array([[Sgg, Sgh], [Sgh, Shh]])
cc = np.array([Sgr, Shr])
th_opt = np.linalg.solve(Qm, cc)
ag = np.linspace(0.6, 3.95, 260)
bg = np.linspace(-0.85, 1.15, 260)
AA, BB = np.meshgrid(ag, bg)
LL = (Srr - 2 * AA * Sgr - 2 * BB * Shr + AA ** 2 * Sgg + BB ** 2 * Shh
      + 2 * AA * BB * Sgh) / float(D['n'])


def gd_path(lr, steps):
    th = np.array([1.0, 0.0])
    path = [th.copy()]
    for _ in range(steps):
        th = th - lr * 2 * (Qm @ th - cc)
        path.append(th.copy())
    return np.array(path)


lmax = float(D['eig'][1])
p_slow = gd_path(0.22 / lmax, 14)
p_osc = gd_path(0.93 / lmax, 14)

fig, axs = plt.subplots(1, 2, figsize=(COL, 1.72), layout='constrained',
                        gridspec_kw={'width_ratios': [1.0, 1.05]})
ax = axs[0]
ypos = np.arange(len(DSLAB))[::-1]
ax.barh(ypos, [min(c, 135) for c in closed], height=0.62, color=BLUE, alpha=0.85, lw=0)
for y, c in zip(ypos, closed):
    ax.text(min(c, 135) - 3, y, r'%.0f\%%' % c, va='center', ha='right', fontsize=6.3,
            color='white')
ax.axvline(100, color='#4D4D4D', lw=0.7, ls=(0, (3, 2)), zorder=3)
ax.text(102, ypos[-1] - 0.62, 'static fit', fontsize=6.0, color='#4D4D4D', va='center')
ax.set_yticks(ypos)
ax.set_yticklabels(DSLAB)
ax.tick_params(axis='y', length=0)
ax.set_ylim(-1.1, 5.5)
ax.set_xlim(0, 165)
ax.set_xticks([0, 50, 100, 150])
ax.set_xlabel(r'Frozen$\to$static gap closed (\%)')
ax.grid(axis='x')
ax.grid(False, axis='y')
title(ax, 'a', 'Oracle accounting')

ax = axs[1]
ax.contourf(AA, BB, LL, levels=14, cmap='Greys', alpha=0.55)
ax.contour(AA, BB, LL, levels=14, colors='white', linewidths=0.35)
ax.plot(p_osc[:, 0], p_osc[:, 1], '-o', color=BLUE, lw=0.8, ms=1.8, mfc='white', mew=0.5,
        label='GD, large step')
ax.plot(p_slow[:, 0], p_slow[:, 1], '-o', color=PINK, lw=0.8, ms=1.8, mfc='white', mew=0.5,
        label='GD, small step')
ax.plot(1.0, 0.0, 'o', ms=3.5, color='black', zorder=6, label='Identity')
ax.plot(th_opt[0], th_opt[1], '*', ms=8, color=OURS, mec='white', mew=0.4, zorder=7,
        label='Closed form')
ax.set_xlabel(r'Scale $\alpha$')
ax.set_ylabel(r'Offset $\beta$', labelpad=1)
ax.set_xticks([1, 2, 3])
ax.set_yticks([-0.5, 0, 0.5, 1.0])
ax.grid(False)
ax.legend(loc='lower right', fontsize=5.8, handlelength=1.0, borderaxespad=0.1,
          labelspacing=0.2)
title(ax, 'b', 'Estimation objective')
print('landscape optimum (%.2f, %.2f)' % tuple(th_opt))
save(fig, 'fig_oracle')

# ======================= ACF of the per-chunk oracle parameters (column) ============
maxlag = 10


def channel_acf(M):
    acfs = []
    for ch in range(M.shape[1]):
        x = M[:, ch] - M[:, ch].mean()
        den = (x * x).sum()
        if den > 0:
            acfs.append([(x[:-k] * x[k:]).sum() / den for k in range(1, maxlag + 1)])
    return np.mean(acfs, axis=0)


fig, axs = plt.subplots(1, 2, figsize=(COL, 1.55), layout='constrained', sharey=True)
band = 1.96 / np.sqrt(30)
for ax, step, tg, tt in [(axs[0], 1, 'a', 'As measured'), (axs[1], 4, 'b', 'Non-overlapping')]:
    ax.axhspan(-band, band, color='#E5E5E5', lw=0, zorder=0)
    ax.axhline(0, color='#9E9E9E', lw=0.5)
    l1 = {}
    for kname, lab, c, m in zip(DSKEY, DSLAB, DS6, MK6):
        M = np.loadtxt('ab_oracle_%s.csv' % kname, delimiter=',')[::step]
        ac = channel_acf(M)
        l1[kname] = round(float(ac[0]), 2)
        ax.plot(range(1, maxlag + 1), ac, marker=m, color=c, lw=0.9, ms=2.6, mfc='white',
                mec=c, mew=0.7, label=lab)
    print('acf step %d lag-1:' % step, l1)
    ax.set_xlabel('Lag (chunks)')
    ax.set_xticks([1, 4, 7, 10])
    ax.set_xlim(0.5, 10.5)
    title(ax, tg, tt)
axs[0].set_ylim(-0.45, 0.7)
axs[0].set_ylabel('Autocorrelation')
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc='outside upper center', ncol=6, fontsize=6.5, columnspacing=0.6,
           handlelength=1.2, handletextpad=0.3)
save(fig, 'fig_acf')

# ======================= Audit (column) =======================
Q0all = np.load('Q0_electricity.npy')
Qtall = np.load('Qt_final_electricity.npy')
Aall = np.loadtxt('alpha_traj_electricity.csv', delimiter=',')
CC = Q0all.shape[1]
fig, axs = plt.subplots(1, 2, figsize=(COL, 1.62), layout='constrained',
                        gridspec_kw={'width_ratios': [1.15, 1.0]})
ax = axs[0]
g117 = (min(Q0all[:, 117].min(), Qtall[:, 117].min()), max(Q0all[:, 117].max(), Qtall[:, 117].max()))
g81 = (min(Q0all[:, 81].min(), Qtall[:, 81].min()), max(Q0all[:, 81].max(), Qtall[:, 81].max()))
off81 = g117[0] - 0.7 - g81[1]
for ch, off, c in [(117, 0.0, BLUE), (81, off81, PINK)]:
    ax.plot(range(W), Q0all[:, ch] + off, color=FROZEN, lw=0.9)
    ax.plot(range(W), Qtall[:, ch] + off, color=c, lw=1.1)
    top = max(Q0all[:, ch].max(), Qtall[:, ch].max()) + off
    ax.text(3, top + 0.12, 'ch.~%d' % ch, color=c, fontsize=6.5, va='bottom')
for d in range(1, 7):
    ax.axvline(d * 24, color='#BDBDBD', lw=0.3, zorder=0)
ax.set_ylim(off81 + g81[0] - 0.2, g117[1] + 1.55)
ax.set_xlim(0, W - 1)
ax.set_xticks([0, 48, 96, 144])
ax.set_yticks([])
ax.spines['left'].set_visible(False)
ax.grid(False)
ax.set_xlabel('Hour of week')
ax.legend(handles=[Line2D([], [], color=FROZEN, lw=0.9, label='Frozen $Q$'),
                   Line2D([], [], color='black', lw=1.1, label='Recalibrated')],
          loc='upper right', ncol=1, fontsize=6.3, borderaxespad=0.0)
title(ax, 'a', 'Cycle before/after')

ax = axs[1]
a_drift = np.abs(Aall - 1).mean(axis=0)
srt = np.sort(a_drift)
ax.fill_between(range(1, CC + 1), srt, color=BLUE, alpha=0.15, lw=0)
ax.plot(range(1, CC + 1), srt, color=BLUE, lw=1.2)
ax.axhline(1.0, color='#4D4D4D', lw=0.7, ls=(0, (3, 2)))
ax.text(8, 1.02, r'clamp bound $\alpha{=}2$', fontsize=6.3, color='#4D4D4D', va='bottom')
ax.set_xlim(1, CC)
ax.set_ylim(0, 1.18)
ax.set_xticks([1, 100, 200, 300])
ax.set_xlabel('Channel (ranked)')
ax.set_ylabel(r'Mean $|\hat\alpha-1|$')
title(ax, 'b', 'Drift per channel')
print('audit: min %.2f p10 %.2f median %.2f' % (a_drift.min(), np.percentile(a_drift, 10),
                                                np.median(a_drift)))
save(fig, 'fig_audit')

# ======================= Appendix: drift monitor 2x2 (column) =======================
fig, axs = plt.subplots(2, 2, figsize=(COL, 2.55), layout='constrained')
for r, (d, lab) in enumerate([('electricity', 'Electricity'), ('traffic', 'Traffic')]):
    AA_ = np.loadtxt('alpha_traj_%s.csv' % d, delimiter=',')
    fmm = np.load('frozen_mse_%s.npy' % d).mean(axis=1)
    rmm = np.load('chunk_mse_%s.npy' % d)
    nbb = min(len(fmm), len(rmm) // 25)
    rc = rmm[:nbb * 25].reshape(nbb, 25).mean(axis=1)
    nn = min(len(AA_), nbb)
    drift = np.abs(AA_[:nn] - 1).mean(axis=1)
    impr = (fmm[:nn] - rc[:nn]) / fmm[:nn] * 100
    print('%s: frozen beaten on %.0f%% of chunks, mean reduction %.1f%%' % (
        d, 100 * (impr > 0).mean(), impr.mean()))
    ax = axs[r, 0]
    ax.plot(range(nn), drift, color=BLUE, lw=1.1)
    ax.set_xlim(0, nn - 1)
    ax.set_ylabel(r'Mean $|\hat\alpha_t-1|$')
    title(ax, 'abcd'[2 * r], '%s: drift signal' % lab)
    ax = axs[r, 1]
    ax.axhline(0, color='#9E9E9E', lw=0.5)
    ax.plot(range(nn), impr, '.', color=FROZEN, ms=2.0, alpha=0.6)
    ax.plot(range(k - 1, nn), np.convolve(impr, np.ones(k) / k, mode='valid'), color=OURS,
            lw=1.3)
    ax.set_xlim(0, nn - 1)
    ax.set_ylabel(r'MSE reduction (\%)')
    title(ax, 'abcd'[2 * r + 1], '%s: gain' % lab)
for ax in axs[1]:
    ax.set_xlabel('Test chunk')
save(fig, 'fig_drift_monitor')
print('ALL FIGURES DONE')
