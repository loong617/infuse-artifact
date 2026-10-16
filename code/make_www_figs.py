# -*- coding: utf-8 -*-
"""WWW 2027 revision figure system.
One design system, every panel redrawn; outputs ONLY the 13 PDFs used by main_www.tex.

Design rules (carried over from make_final_figs.py, cleaned):
- global color semantics; 3 visual tiers (hero 2.0 / support 1.1 / background 0.5)
- unified font sizes at TRUE rendered width (SIZING CONTRACT)
- all axis text black (no colored axis labels); twin-axis ticks colored only where needed
- fixed canvas + subplots_adjust (savefig.bbox=None) so panel sizes are exact
Sources: empirical cycle from ../data (new, replaces model-output Q0-vs-Qt panel);
sensitivity from subexp_final.json (5-seed means); oracle values verified against
results_final8.csv / results_sota.csv / rq34_stdout.log (seed 2024); ACF from
ab_oracle_*.csv; audit + drift monitor from per-chunk dumps.
"""
import json
import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    import scienceplots
    plt.style.use(['science', 'no-latex', 'ieee'])
except Exception:
    pass

OUT = '../paper/figs'
W_TEASE = 2.17
W_HALF = 1.63

FULL_RC = {'font.size': 9, 'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
           'legend.fontsize': 8}
HALF_RC = {'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
           'legend.fontsize': 6.5}
plt.rcParams.update({
    **FULL_RC,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': None,
    'axes.grid': True, 'grid.alpha': 0.30, 'grid.linewidth': 0.3,
    'axes.spines.top': False, 'axes.spines.right': False,
    'font.family': 'DejaVu Sans', 'pdf.fonttype': 42,
    'axes.axisbelow': True,
    'text.color': 'black', 'axes.labelcolor': 'black',
    'xtick.color': 'black', 'ytick.color': 'black',
})
def use_half():
    plt.rcParams.update(HALF_RC)
def use_full():
    plt.rcParams.update(FULL_RC)

C = {'ours': '#D55E00', 'frozen': '#9CA3AF', 'ref': '#56B4E9',
     'data': '#0072B2', 'aux': '#009E73'}
PAL6 = ['#D55E00', '#0072B2', '#009E73', '#CC79A7', '#56B4E9', '#E69F00']

def grid_y(ax):
    ax.grid(axis='y')
    ax.grid(False, axis='x')

# ================= Figure 1: the deployment story =================
CH = 119
W = 168

# (a) HERO = the drift gap between the training-period and test-period cycles
#     (pure data, no model output; replaces the old Q0-vs-Qt panel)
rows = open('../data/electricity/electricity.csv').read().splitlines()[1:]
stamps = [r.split(',')[0] for r in rows]
d0 = datetime.datetime.strptime(stamps[0], '%Y-%m-%d %H:%M:%S')
ho0 = d0.weekday() * 24 + d0.hour
Xv = np.genfromtxt('../data/electricity/electricity.csv', delimiter=',', skip_header=1)[:, 1:]
n_tot = len(Xv)
ho = (np.arange(n_tot) + ho0) % W
tr_idx = np.arange(0, int(n_tot * 0.7))
te_idx = np.arange(int(n_tot * 0.8), n_tot)
ec_tr = np.array([Xv[tr_idx, CH][ho[tr_idx] == p].mean() for p in range(W)])
ec_te = np.array([Xv[te_idx, CH][ho[te_idx] == p].mean() for p in range(W)])
amp_tr = ec_tr.max() - ec_tr.min()
amp_te = ec_te.max() - ec_te.min()
print('f1_a: ch%d train amp %.0f -> test amp %.0f (%.0f%% drop), level shift %.0f' %
      (CH, amp_tr, amp_te, 100 * (1 - amp_te / amp_tr), ec_te.mean() - ec_tr.mean()))

fig, ax = plt.subplots(figsize=(W_TEASE, 1.45))
fig.subplots_adjust(left=0.235, right=0.965, top=0.845, bottom=0.235)
ax.fill_between(range(W), ec_tr, ec_te, color=C['ours'], alpha=0.16, lw=0)
ax.plot(range(W), ec_tr, color=C['frozen'], lw=1.1)
ax.plot(range(W), ec_te, color=C['data'], lw=1.1, ls='--')
for d in range(1, 7):
    ax.axvline(d * 24, color=C['frozen'], lw=0.2, alpha=0.25)
ax.set_xlabel('Hour of week')
ax.set_ylabel('Load (MW)')
ax.set_xlim(0, W - 1)
ax.set_xticks([0, 48, 96, 144])
from matplotlib.ticker import FuncFormatter
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: '%.1f' % (v / 1000)))
ax.legend(handles=[Line2D([], [], color=C['frozen'], lw=1.1, label='train period'),
                   Line2D([], [], color=C['data'], lw=1.1, ls='--', label='test period')],
          loc='lower left', bbox_to_anchor=(0.0, 1.01), ncol=2, fontsize=7, frameon=False,
          handlelength=1.4, columnspacing=0.8, handletextpad=0.4)
grid_y(ax)
fig.savefig(OUT + '/f1_a.pdf', facecolor='white')
plt.close()

# (b) HERO = monotonically rising cumulative excess of frozen over adapted
fm = np.load('frozen_mse_electricity.npy').mean(axis=1)
rm = np.load('chunk_mse_electricity.npy')
nb = min(len(fm), len(rm) // 25)
rm_ch = rm[:nb * 25].reshape(nb, 25).mean(axis=1)
A = np.loadtxt('alpha_traj_electricity.csv', delimiter=',')[:, CH]
n_chunks = min(len(A), nb)
fig, ax = plt.subplots(figsize=(W_TEASE, 1.45))
fig.subplots_adjust(left=0.235, right=0.965, top=0.845, bottom=0.235)
gap = np.cumsum(fm[:n_chunks] - rm_ch[:n_chunks]) * 25
ax.plot(range(n_chunks), fm[:n_chunks], color=C['frozen'], lw=0.4, alpha=0.35)
ax.plot(range(n_chunks), gap, color=C['ours'], lw=2.0, ls='-.')
ax.set_xlabel('Serving chunk')
ax.set_ylabel('Cum. excess MSE')
ax.set_xlim(0, n_chunks - 1)
grid_y(ax)
fig.savefig(OUT + '/f1_b.pdf', facecolor='white')
plt.close()

# (c) HERO = adapted MSE below frozen; causal alpha on the twin axis
fig, ax = plt.subplots(figsize=(W_TEASE, 1.45))
fig.subplots_adjust(left=0.115, right=0.87, top=0.845, bottom=0.235)
ax.plot(range(n_chunks), A, color=C['data'], lw=1.1, alpha=0.85, label=r'$\hat\alpha_t$')
ax.axhline(1.0, color=C['frozen'], lw=0.6, ls=':')
ax2 = ax.twinx()
k = 9
rsm = np.convolve(rm_ch[:n_chunks], np.ones(k) / k, mode='valid')
fsm = np.convolve(fm[:n_chunks], np.ones(k) / k, mode='valid')
ax2.fill_between(range(8, n_chunks), rsm, fsm, color=C['ours'], alpha=0.10, lw=0)
ax2.plot(range(8, n_chunks), fsm, color=C['frozen'], lw=1.1, ls='--', label='frozen MSE')
ax2.plot(range(8, n_chunks), rsm, color=C['ours'], lw=2.0, label='ReCycle MSE')
ax2.tick_params(axis='y', labelsize=7)
ax2.spines['right'].set_visible(True)
ax2.grid(False)
ax.set_xlabel('Serving chunk')
# no left ylabel: the legend entry (same blue) + caption "(blue, left axis)" carry it;
# the rotated mathtext label is taller than a plain one and clips off-canvas here
ax.set_xlim(0, n_chunks - 1)
ax.set_ylim(0.85, 2.4)
ax.grid(False, axis='x')
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc='lower left', bbox_to_anchor=(0.0, 1.01), ncol=3, fontsize=7,
          frameon=False, handlelength=1.0, columnspacing=0.7, handletextpad=0.4)
fig.savefig(OUT + '/f1_c.pdf', facecolor='white')
plt.close()
print('OK f1_a/b/c')

# ================= Figure 2: sensitivity (eta + windows), 5-seed means from json =================
use_half()
SENS = json.load(open('subexp_final.json'))
NAME = {'ETTh2': 'ETTh2', 'ETTm2': 'ETTm2', 'electricity': 'Elec.',
        'traffic': 'Traffic', 'PEMS03': 'PEMS03', 'PEMS04': 'PEMS04'}
ORDER = ['ETTh2', 'ETTm2', 'electricity', 'traffic', 'PEMS03', 'PEMS04']

def sens_panel(data, xs, out, frozen_x, label_kw, xlabel, xlim, ylim, tick_pos=None):
    fig, ax = plt.subplots(figsize=(W_HALF, 1.95))
    fig.subplots_adjust(left=0.24, right=0.955, top=0.735, bottom=0.20)
    for (name, ys), c, m in zip(data.items(), PAL6, ['o', 's', '^', 'D', 'v', 'P']):
        best = min(ys)
        pen = [(y - best) / best * 100 for y in ys]
        ax.plot(xs, pen, marker=m, color=c, lw=1.0, ms=3, mfc='white', mec=c, mew=0.8,
                label=NAME[name])
    ax.axhline(0, color=C['frozen'], lw=0.6, ls=':')
    ax.axvline(frozen_x, ymin=0, ymax=0.80, color=C['frozen'], lw=0.8, ls='--', alpha=0.8)
    ax.tick_params(axis='x', which='both', top=False)
    ax.text(**label_kw)
    # anchor at the FIGURE left edge: a centered 3-column legend (101pt) is wider
    # than the 117pt half-column canvas and clips PEMS03/PEMS04 at the right edge
    ax.legend(ncol=3, fontsize=6.5, frameon=False, loc='lower left',
              bbox_to_anchor=(-0.336, 1.0),
              columnspacing=0.5, handlelength=1.0, handletextpad=0.3)
    ax.set_xscale('log')
    # tick_pos may be a subset of xs when two labels would collide on the log axis
    # at half-column width (e.g. 200 vs 500); the frozen-value vline carries its
    # own annotation, so its tick can be dropped
    ticks = xs if tick_pos is None else tick_pos
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(v) for v in ticks])
    from matplotlib.ticker import NullFormatter
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('MSE penalty (%)')
    grid_y(ax)
    fig.savefig(OUT + '/' + out, facecolor='white')
    plt.close()

sens_panel({k: SENS['eta'][k] for k in ORDER}, [5, 50, 200, 500], 'eta_sensitivity.pdf',
           200, dict(x=212, y=0.12, s='frozen $\\eta$', fontsize=6.5, color='#4B5563',
                     ha='left', va='bottom'),
           'Hedge rate $\\eta$', (3.8, 3200), (-0.15, 2.75), tick_pos=[5, 50, 500])
sens_panel({k: SENS['hist'][k] for k in ORDER}, [5, 15, 30], 'win_sensitivity.pdf',
           15, dict(x=15, y=1.86, s='frozen windows', fontsize=6.5, color='#4B5563',
                    ha='center', va='top'),
           'Estimation windows', (4.2, 42), (-0.12, 2.05))
print('OK eta/win sensitivity')

# ================= Figure 3: (a) oracle dumbbell + (b) exact-quadratic landscape ===
frozen_v = [.3005, .1778, .2430, .8647, .5030, .5052]
causal_v = [.2710, .1702, .2221, .7998, .4503, .4764]
static_v = [.2629, .1719, .2216, .7996, .4359, .4628]
names = ['ETTh2', 'ETTm2', 'Elec.', 'Traffic', 'PEMS03', 'PEMS04']
closed = [(f - c) / (f - s) * 100 for f, c, s in zip(frozen_v, causal_v, static_v)]
print('dumbbell closed%%:', [round(c, 1) for c in closed])
use_half()
fig, ax = plt.subplots(figsize=(W_HALF, 1.66))
fig.subplots_adjust(left=0.29, right=0.97, top=0.93, bottom=0.24)
ypos = np.arange(len(names))[::-1]
for y, f, c_, s_, cl in zip(ypos, frozen_v, causal_v, static_v, closed):
    ax.plot([0, 100], [y, y], color=C['ref'], lw=4.0, alpha=0.45, solid_capstyle='round')
    ax.plot([0, 0], [y - 0.18, y + 0.18], color=C['frozen'], lw=1.6)
    xx = min(cl, 132)
    ax.plot(xx, y, 'o', ms=5.5, color=C['ours'], zorder=5)
    if xx > 120:
        # ETTm2 sits far right: centered above its dot keeps the label inside and
        # clear of the 100% vline that a left-side 3-digit label would straddle
        ax.annotate('%.0f%%' % cl, xy=(xx, y), xytext=(0, 7), textcoords='offset points',
                    fontsize=7, color=C['ours'], fontweight='bold', ha='center', va='bottom')
    else:
        # left of the dot, vertically centered in its own row slot (row pitch 11.8pt,
        # adjacent-row dots >=8.3pt away, label band +/-4.5pt -> no collisions possible)
        ax.annotate('%.0f%%' % cl, xy=(xx, y), xytext=(-8, 2), textcoords='offset points',
                    fontsize=7, color=C['ours'], fontweight='bold', ha='right', va='center')
ax.axvline(100, color=C['frozen'], lw=0.9, ls=':')
ax.text(103, 6.55, 'static' + chr(10) + 'fit', fontsize=6, color='#4B5563', va='top')
ax.set_yticks(ypos)
ax.set_yticklabels(names)
ax.set_xlabel('Gap closed (%)', fontsize=7.5)
ax.set_xlim(-4, 148)
ax.set_ylim(-0.6, 6.7)
grid_y(ax)
ax.grid(False, axis='x')
fig.savefig(OUT + '/oracle_dumbbell.pdf', facecolor='white')
plt.close()
print('OK oracle_dumbbell (half-width)')

# (b) the estimation objective of one real adaptation, restricted to the subspace:
#     exact quadratic by Lemma 1; closed-form star vs gradient paths from the identity
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
print('landscape: optimum (%.2f, %.2f); slow path ends (%.2f, %.2f); osc ends (%.2f, %.2f)'
      % (th_opt[0], th_opt[1], p_slow[-1][0], p_slow[-1][1], p_osc[-1][0], p_osc[-1][1]))

fig, ax = plt.subplots(figsize=(W_HALF, 1.66))
fig.subplots_adjust(left=0.185, right=0.97, top=0.93, bottom=0.24)
ax.pcolormesh(AA, BB, LL, cmap='Greys', shading='auto', rasterized=True)
cs = ax.contour(AA, BB, LL, levels=6, colors='#FFFFFF', linewidths=0.5, alpha=0.9)
ax.plot(p_osc[:, 0], p_osc[:, 1], '-', color=C['data'], lw=1.0, ms=2.2, marker='o',
        mfc='white', mec=C['data'], mew=0.7, zorder=5)
ax.plot(p_slow[:, 0], p_slow[:, 1], '--', color='#CC79A7', lw=1.0, ms=2.2, marker='o',
        mfc='white', mec='#CC79A7', mew=0.7, zorder=5)
ax.plot(1.0, 0.0, 'o', ms=4.5, color=C['frozen'], zorder=6)
ax.plot(th_opt[0], th_opt[1], '*', ms=11, color=C['ours'], zorder=7)
ax.text(1.0, -0.30, 'identity', fontsize=6, color='#4B5563', ha='center', va='top')
ax.text(th_opt[0] - 0.08, th_opt[1] - 0.30, 'closed-form', fontsize=6, color=C['ours'],
        ha='right', va='top', fontweight='bold')
# path identities are carried by the caption (a near-critical step zigzags, a small
# step crawls); in-panel text labels are ~30pt wide at this panel size and collide
ax.set_xlabel(r'$\alpha$ (cycle scale)', fontsize=7.5)
ax.set_ylabel(r'$\beta$ (cycle offset)', fontsize=7.5)
ax.set_xticks([1, 2, 3])
ax.set_yticks([-0.5, 0, 0.5])
ax.grid(False)
fig.savefig(OUT + '/landscape.pdf', facecolor='white')
plt.close()
print('OK landscape (exact quadratic bowl)')

# ================= Figure 4: drift-process ACF, before vs after decimation =======
DS = ['ETTh2', 'ETTm2', 'electricity', 'traffic', 'PEMS03', 'PEMS04']
DSLAB = ['ETTh2', 'ETTm2', 'Elec.', 'Traffic', 'PEMS03', 'PEMS04']
maxlag = 10

def channel_acf(M):
    acfs = []
    for ch in range(M.shape[1]):
        x = M[:, ch] - M[:, ch].mean()
        den = (x * x).sum()
        if den > 0:
            acfs.append([(x[:-k] * x[k:]).sum() / den for k in range(1, maxlag + 1)])
    return np.mean(acfs, axis=0)

def acf_panel(src_fn, out, with_legend):
    fig, ax = plt.subplots(figsize=(W_HALF, 1.34))
    fig.subplots_adjust(left=0.205, right=0.965, top=0.70, bottom=0.26)
    lag1 = {}
    for ds, lab, c in zip(DS, DSLAB, PAL6):
        M = src_fn(ds)
        ac = channel_acf(M)
        lag1[ds] = ac[0]
        ax.plot(range(1, maxlag + 1), ac, 'o-', color=c, lw=1.0, ms=2.5, label=lab)
    print('%s lag-1:' % out, {k: round(v, 3) for k, v in lag1.items()})
    band = 1.96 / np.sqrt(30)
    ax.axhspan(-band, band, color=C['frozen'], alpha=0.22, lw=0)
    ax.axhline(0, color='#9CA3AF', lw=0.6, ls=':')
    ax.set_xlabel('Lag (chunks)', fontsize=7.5)
    ax.set_ylabel('ACF', fontsize=7.5)
    ax.set_ylim(-0.4, 0.7)
    ax.set_yticks([-0.4, -0.2, 0, 0.2, 0.4, 0.6])
    if with_legend:
        ax.legend(ncol=3, fontsize=6, frameon=False, loc='lower left',
                  bbox_to_anchor=(-0.10, 1.0), columnspacing=0.5, handlelength=1.0,
                  handletextpad=0.3)
    grid_y(ax)
    fig.savefig(OUT + '/' + out, facecolor='white')
    plt.close()

use_half()
# (a) as measured: consecutive leaked fits on Electricity/Traffic share 71 of 96
#     target steps (chunk stride 25 < horizon 96) -> the shared data manufactures
#     correlation (lag-1 0.57/0.58)
acf_panel(lambda ds: np.loadtxt('ab_oracle_%s.csv' % ds, delimiter=','),
          'acf_a.pdf', with_legend=True)
# (b) decimated to non-overlapping strides: the artifact vanishes; all six series
#     sit inside the 95% band (drawn at n=30; shorter decimated series have wider
#     bands up to +-0.74 and also stay within them)
acf_panel(lambda ds: np.loadtxt('ab_oracle_%s.csv' % ds, delimiter=',')[::4],
          'acf_b.pdf', with_legend=False)
print('OK acf_a (raw) + acf_b (decimated)')

# ================= Figure 5: audit (recalibrated cycles + channel ranking) =================
Q0all = np.load('Q0_electricity.npy')
Qtall = np.load('Qt_final_electricity.npy')
Aall = np.loadtxt('alpha_traj_electricity.csv', delimiter=',')
_, CC = Q0all.shape

use_half()
fig, ax = plt.subplots(figsize=(W_HALF, 1.58))
fig.subplots_adjust(left=0.245, right=0.965, top=0.78, bottom=0.235)
g117_max = max(Q0all[:, 117].max(), Qtall[:, 117].max())
g117_min = min(Q0all[:, 117].min(), Qtall[:, 117].min())
g81_max = max(Q0all[:, 81].max(), Qtall[:, 81].max())
g81_min = min(Q0all[:, 81].min(), Qtall[:, 81].min())
off81 = g117_min - 0.85 - g81_max
offs = {117: 0.0, 81: off81}
lab_ys = {117: g117_max + 0.34, 81: off81 + g81_max + 0.34}
ax.set_ylim(off81 + g81_min - 0.15, g117_max + 0.34 + 0.48)
for ch in (117, 81):
    off, c = offs[ch], (C['data'] if ch == 117 else '#CC79A7')
    ax.plot(range(W), Q0all[:, ch] + off, color=C['frozen'], lw=1.0)
    ax.plot(range(W), Qtall[:, ch] + off, color=c, lw=1.4, ls='--')
    ax.fill_between(range(W), Q0all[:, ch] + off, Qtall[:, ch] + off, color=c, alpha=0.12, lw=0)
    ax.text(2, lab_ys[ch], 'ch. %d' % ch, fontsize=7, color=c, fontweight='bold', va='bottom')
for d in range(1, 7):
    ax.axvline(d * 24, color=C['frozen'], lw=0.2, alpha=0.25)
ax.legend(handles=[Line2D([], [], color=C['frozen'], lw=1.0, label='frozen $Q$'),
                   Line2D([], [], color=C['data'], lw=1.4, ls='--', label='recalibrated')],
          loc='lower left', bbox_to_anchor=(-0.28, 1.01), ncol=2, fontsize=6.5, frameon=False,
          handlelength=1.2, columnspacing=0.8, handletextpad=0.4)
ax.set_xlabel('Hour of week')
ax.set_ylabel('Cycle value (offset)')
ax.set_xlim(0, W - 1)
grid_y(ax)
ax.grid(False, axis='x')
fig.savefig(OUT + '/audit_a.pdf', facecolor='white')
plt.close()

fig, ax = plt.subplots(figsize=(W_HALF, 1.58))
fig.subplots_adjust(left=0.285, right=0.965, top=0.86, bottom=0.235)
a_drift = np.abs(Aall - 1).mean(axis=0)
srt = np.sort(a_drift)
ax.fill_between(range(1, CC + 1), srt, color=C['ref'], alpha=0.30, lw=0)
ax.plot(range(1, CC + 1), srt, color=C['data'], lw=1.6)
ax.axhline(1.0, color=C['frozen'], lw=0.8, ls=':')
ax.text(CC * 0.90, 0.99, 'saturation', fontsize=6.5, color='#4B5563', ha='right', va='bottom')
ax.set_xlabel('Channel rank')
ax.set_ylabel(r'mean $|\hat\alpha - 1|$')
ax.set_ylim(0, 1.12)
grid_y(ax)
ax.grid(False, axis='x')
fig.savefig(OUT + '/audit_b.pdf', facecolor='white')
plt.close()
print('OK audit_a/b  (drift pct: 10th=%.2f median=%.2f min=%.2f)' %
      (np.percentile(a_drift, 10), np.median(a_drift), a_drift.min()))

# ================= Appendix figure: drift monitor (4 panels) =================
for d, c in [('electricity', C['aux']), ('traffic', '#CC79A7')]:
    AA = np.loadtxt('alpha_traj_%s.csv' % d, delimiter=',')
    fmm = np.load('frozen_mse_%s.npy' % d).mean(axis=1)
    rmm = np.load('chunk_mse_%s.npy' % d)
    nbb = min(len(fmm), len(rmm) // 25)
    rc = rmm[:nbb * 25].reshape(nbb, 25).mean(axis=1)
    nn = min(len(AA), nbb)
    drift = np.abs(AA[:nn] - 1).mean(axis=1)
    impr = (fmm[:nn] - rc[:nn]) / fmm[:nn] * 100
    tag = 'elec' if d == 'electricity' else 'traf'
    print('%s: frozen beaten on %.0f%% of chunks, mean red %.1f%%' %
          (d, 100 * (impr > 0).mean(), impr.mean()))

    fig, ax = plt.subplots(figsize=(W_HALF, 1.22))
    fig.subplots_adjust(left=0.28, right=0.955, top=0.87, bottom=0.27)
    ax.plot(range(nn), drift, color=c, lw=1.6)
    ax.set_ylabel(r'$|\hat\alpha-1|$', fontsize=7.5)
    ax.set_xlabel('Test chunk', fontsize=7.5)
    ax.set_xlim(0, nn - 1)
    grid_y(ax)
    fig.savefig(OUT + '/tl_%s_drift.pdf' % tag, facecolor='white')
    plt.close()

    fig, ax = plt.subplots(figsize=(W_HALF, 1.22))
    fig.subplots_adjust(left=0.28, right=0.955, top=0.87, bottom=0.27)
    ax.plot(range(nn), impr, '.', color=C['frozen'], ms=2, alpha=0.40)
    sm = np.convolve(impr, np.ones(9) / 9, mode='valid')
    ax.plot(range(8, nn), sm, color=C['ours'], lw=1.8)
    ax.axhline(0, color='#9CA3AF', lw=0.6, ls=':')
    ax.set_ylabel('MSE red. (%)', fontsize=7.5)
    ax.set_xlabel('Test chunk', fontsize=7.5)
    ax.set_xlim(0, nn - 1)
    grid_y(ax)
    fig.savefig(OUT + '/tl_%s_impr.pdf' % tag, facecolor='white')
    plt.close()
print('OK tl panels')

# clean pairwise corr (electricity), for the pooling motivation sentence
A = np.loadtxt('alpha_traj_electricity.csv', delimiter=',')
Cm = np.corrcoef(A.T)
off = Cm[np.triu_indices_from(Cm, 1)]
print('electricity mean pairwise corr (clean): %.3f' % np.nanmean(off))
print('ALL FIGURES DONE -> ' + OUT)
