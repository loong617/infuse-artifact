# -*- coding: utf-8 -*-
"""Build every table (.tex) and figure (.pdf) of the paper from the raw result files, and print every number
quoted in the text.

Inputs:  ../baselines/COSA_official/grid_itdl.csv, grid_patch.csv, grid_noadj.csv  (official harness)
         res_grid.jsonl  (eval_all on dumps/grid/*.npz: frozen backbones)
         res_tta.jsonl   (eval_all on dumps/grid/tta/*.npz: adapters' emitted forecasts, revision disabled)
         res_fm.jsonl    (eval_all on dumps/fm/*.npz: Chronos-Bolt)
"""
import csv
import json
import os
from collections import defaultdict

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

PAPER = '../paper/'
GRID = '../baselines/COSA_official/'
DS = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'exchange_rate', 'weather']
DSN = {'ETTh1': 'ETTh1', 'ETTh2': 'ETTh2', 'ETTm1': 'ETTm1', 'ETTm2': 'ETTm2', 'exchange_rate': 'Exchange',
       'weather': 'Weather', 'azure_functions': 'Azure'}
HS = ['96', '192', '336', '720']
BB = ['iTransformer', 'PatchTST', 'DLinear']
FMS = ['bolt-small', 'bolt-base']
FMN = {'bolt-small': 'Chronos-Bolt-S', 'bolt-base': 'Chronos-Bolt-B'}
TTA = ['TAFAS', 'PETSA', 'COSA']               # adapters with a revision step (adaptation-only variants exist)
ADP = TTA + ['DADAPTER']                       # all adapters (delta-Adapter: no revision step)
STACK = ['tafas_fuse', 'petsa_fuse', 'cosa_fuse', 'dadapter_fuse']
COLS = ['frozen', 'TAFAS', 'PETSA', 'COSA', 'DADAPTER', 'refresh', 'update', 'fuse'] + STACK
COLN = {'frozen': 'Frozen', 'TAFAS': 'TAFAS', 'PETSA': 'PETSA', 'COSA': 'COSA', 'DADAPTER': r'$\delta$-Ad.',
        'refresh': 'Refresh', 'update': r'\InfuseU{}', 'fuse': r'\Infuse{}', 'tafas_fuse': 'TAFAS',
        'petsa_fuse': 'PETSA', 'cosa_fuse': 'COSA', 'dadapter_fuse': r'$\delta$-Ad.'}
COLN_FLAT = dict(COLN, tafas_fuse=r'T+\Infuse{}', petsa_fuse=r'P+\Infuse{}', cosa_fuse=r'C+\Infuse{}',
                 dadapter_fuse=r'$\delta$+\Infuse{}')


def best_adapter(row):
    return min(row[t][0] for t in ADP if t in row)

plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage[T1]{fontenc}\usepackage{libertine}\usepackage[libertine]{newtxmath}',
    'font.family': 'serif', 'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'axes.grid.axis': 'y', 'grid.linewidth': 0.4, 'grid.color': '#D9D9D9',
    'axes.axisbelow': True, 'axes.titlelocation': 'left', 'axes.titlepad': 4, 'legend.frameon': False,
    'figure.dpi': 300, 'savefig.dpi': 300, 'pdf.fonttype': 42,
})
C_OURS, C_REF, C_GREY, C_BLUE = '#D55E00', '#E69F00', '#7F7F7F', '#0072B2'
C_TTA = {'TAFAS': '#56B4E9', 'PETSA': '#0072B2', 'COSA': '#009E73', 'DADAPTER': '#CC79A7'}


def rel(a, b):
    return 100.0 * (a - b) / b


def fmt(v):
    return ('%.3f' % v)[1:] if v < 1 else '%.3f' % v


def load_grid():
    R = defaultdict(dict)
    import glob as _glob
    for fn in ['grid_itdl.csv', 'grid_patch.csv', 'grid_patch_small.csv', 'grid_patch_long.csv', 'grid_patch192.csv',
               'grid_noadj.csv', 'grid_azure.csv', 'grid_azure_patch.csv', 'grid_azure_noadj.csv'] + \
            sorted(os.path.basename(p) for p in _glob.glob(GRID + 'grid_dad*.csv')):
        p = GRID + fn
        if not os.path.exists(p):
            continue
        for r in csv.reader(open(p)):
            if len(r) < 7 or not r[5]:
                continue
            R[(r[0], r[1], r[2])][r[4]] = (float(r[5]), float(r[6]) if r[6] else np.nan)
    # the official adapters print only MSE; a second run of the same official configuration records MAE as well
    dev = []
    for fn in ('grid_mae.csv', 'grid_mae_p.csv'):
        p = GRID + fn
        if not os.path.exists(p):
            continue
        for r in csv.reader(open(p)):
            if len(r) < 7 or not r[5] or r[3] != '0':
                continue
            k = (r[0], r[1], r[2])
            if r[4] in R[k] and r[6]:
                dev.append(abs(float(r[5]) - R[k][r[4]][0]) / R[k][r[4]][0])
                R[k][r[4]] = (R[k][r[4]][0], float(r[6]))
    if dev:
        print('official adapters re-run for MAE: %d runs, max relative MSE deviation %.2e' % (len(dev), max(dev)))
    return R


def load_eval(fn):
    E = {}
    if os.path.exists(fn):
        for l in open(fn):
            d = json.loads(l)
            E[os.path.basename(d['path'])[:-4]] = d
    return E


def add_elf(row, name, EL):
    """ELF (emission-time online adaptation) and ELF + refresh / INFUSE-U / INFUSE, from res_elf.jsonl."""
    el = EL.get('ELF_' + name)
    if el is not None:
        for m, c in (('frozen', 'elf'), ('refresh', 'elf_refresh'), ('update', 'elf_update'), ('fuse', 'elf_fuse')):
            row[c] = tuple(el[m])
    return row


def merged(dsets=DS):
    R, EV, ET = load_grid(), load_eval('res_grid.jsonl'), load_eval('res_tta.jsonl')
    EL = load_eval('res_elf.jsonl')
    M = {}
    for bb in BB:
        for ds in dsets:
            for H in HS:
                k = (bb, ds, H)
                ev = EV.get('%s_%s_%s' % k)
                if k not in R or ev is None or not all(t in R[k] for t in TTA):
                    continue
                row = dict(R[k])
                row['frozen'] = tuple(ev['frozen'])               # identical to the harness (checked below)
                row['frozen_harness'] = R[k].get('frozen')
                for m in ('refresh', 'update', 'fuse', 'fuse_eq', 'fuse_bg'):
                    row[m] = tuple(ev[m])
                row['ev'] = ev
                for t in ADP:
                    et = ET.get(('%s-noadj_%s_%s_%s' if t in TTA else '%s_%s_%s_%s') % ((t,) + k))
                    if et is not None:
                        row[t.lower() + '_fuse'] = tuple(et['fuse'])
                        row[t.lower() + '_refresh'] = tuple(et['refresh'])
                        row[t.lower() + '_noadj_dump'] = tuple(et['frozen'])
                M[k] = add_elf(row, '%s_%s_%s' % k, EL)
    return M


# ------------------------------------------------------------------------------------------- tables
def table_main(M):
    cols = [c for c in COLS if any(c in M[k] for k in M)]
    stk = [c for c in cols if c in STACK]
    n_single = len(cols) - len(stk)
    lines = [r'\begin{table*}[t]', r'\centering',
             r'\caption{MSE under the TTA protocol, averaged over the four horizons (per-horizon MSE and MAE in '
             r'Appendix~\ref{app:full}). TAFAS, PETSA, COSA, and $\delta$-Adapter ($\delta$-Ad.) use the '
             r'official procedures and settings. Under the batch TTA protocol, Refresh and \Infuse{} reuse the '
             r'batch-end forecast already computed by the protocol (zero extra model calls). '
             r'The last four columns apply \Infuse{} to the forecasts emitted by each adapter (TAFAS, PETSA, and COSA '
             r'with their own revision disabled, while $\delta$-Adapter has no revision step). Best and second-best single '
             r'method in \textbf{bold} and \underline{underlined}, and best adapter+\Infuse{} combination in '
             r'\textbf{bold}. The last rows give the mean relative change of MSE and MAE with respect to the frozen '
             r'forecaster over all settings and the number of settings in which each single method is best (MSE).}',
             r'\label{tab:main}', r'\small', r'\setlength{\tabcolsep}{3.5pt}',
             r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}ll %s}' % ('c' * len(cols)), r'\toprule']
    if stk:
        lines += [r'& & \multicolumn{%d}{c}{Single method} & \multicolumn{%d}{c}{Adapter + \Infuse{}} \\'
                  % (n_single, len(stk)),
                  r'\cmidrule(lr){3-%d}\cmidrule(lr){%d-%d}' % (n_single + 2, n_single + 3, len(cols) + 2)]
    lines += [r'Backbone & Dataset & ' + ' & '.join(COLN[c] for c in cols) + r' \\', r'\midrule']
    for bb in BB:
        rows = []
        for ds in DS:
            ks = [(bb, ds, H) for H in HS if (bb, ds, H) in M]
            if len(ks) == 0:
                continue
            vals = {c: (np.mean([M[k][c][0] for k in ks]) if all(c in M[k] for k in ks) else np.nan) for c in cols}
            order = sorted([c for c in cols if c != 'frozen' and c not in STACK and not np.isnan(vals[c])],
                           key=lambda c: vals[c])
            order_s = sorted([c for c in stk if not np.isnan(vals[c])], key=lambda c: vals[c])
            # ties at the printed precision receive the same mark
            first = fmt(vals[order[0]]) if order else None
            second = next((fmt(vals[c]) for c in order if fmt(vals[c]) != first), None)
            first_s = fmt(vals[order_s[0]]) if order_s else None
            cells = []
            for c in cols:
                v = vals[c]
                s = '--' if np.isnan(v) else fmt(v)
                if c == 'frozen' or np.isnan(v):
                    pass
                elif (c in stk and s == first_s) or (c not in stk and s == first):
                    s = r'\textbf{%s}' % s
                elif c not in stk and s == second:
                    s = r'\underline{%s}' % s
                cells.append(s)
            tag = '' if len(ks) == 4 else '$^{%d}$' % len(ks)
            rows.append('%s%s & %s' % (DSN[ds], tag, ' & '.join(cells)))
        for i, r_ in enumerate(rows):
            lines.append(('%s & ' % (r'\multirow{%d}{*}{%s}' % (len(rows), bb) if i == 0 else '')) + r_ + r' \\')
        lines.append(r'\midrule')
    wins = defaultdict(int)
    relsum, relmae = defaultdict(list), defaultdict(list)
    single = [c for c in cols if c != 'frozen' and c not in STACK]
    for k, row in M.items():
        best = min((c for c in single if c in row), key=lambda c: row[c][0])
        wins[best] += 1
        for c in cols[1:]:
            if c in row:
                relsum[c].append(rel(row[c][0], row['frozen'][0]))
                relmae[c].append(rel(row[c][1], row['frozen'][1]))
    lines.append(r'\multicolumn{2}{l}{Mean rel.\ change, MSE} & -- & ' +
                 ' & '.join(r'$%+.1f\%%$' % np.mean(relsum[c]) for c in cols[1:]) + r' \\')
    lines.append(r'\multicolumn{2}{l}{Mean rel.\ change, MAE} & -- & ' +
                 ' & '.join(r'$%+.1f\%%$' % np.nanmean(relmae[c]) for c in cols[1:]) + r' \\')
    for c, nm in (('fuse', 'MaeRelInfuse'), ('TAFAS', 'MaeRelTAFAS'), ('PETSA', 'MaeRelPETSA'), ('COSA', 'MaeRelCOSA'),
                  ('DADAPTER', 'MaeRelDAD')):
        if relmae.get(c):
            num(nm, float(np.nanmean(relmae[c])))
    print('MAE rel change:', {c: round(float(np.nanmean(v)), 2) for c, v in relmae.items()},
          'nan count:', {c: int(np.isnan(v).sum()) for c, v in relmae.items()})
    lines.append(r'\multicolumn{2}{l}{\#Best (of %d)} & -- & ' % len(M) +
                 ' & '.join(str(wins[c]) if c in single else '--' for c in cols[1:]) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular*}', r'\end{table*}']
    open(PAPER + 'tab_main.tex', 'w').write('\n'.join(lines) + '\n')
    return wins, relsum


def table_azure(MA, EVF):
    """Azure Functions: MSE averaged over the four horizons, per forecaster (rows: methods)."""
    ds = 'azure_functions'
    cols = [(bb, bb if bb != 'iTransformer' else 'iTrans.') for bb in BB] + [(fm, FMN[fm].replace('Chronos-', ''))
                                                                              for fm in FMS]
    meths = ['frozen', 'TAFAS', 'PETSA', 'COSA', 'DADAPTER', 'refresh', 'update', 'fuse'] + STACK
    mn = {'frozen': 'Frozen / zero-shot', 'TAFAS': 'TAFAS', 'PETSA': 'PETSA', 'COSA': 'COSA',
          'DADAPTER': r'$\delta$-Adapter', 'refresh': 'Refresh', 'update': r'\InfuseU{}', 'fuse': r'\Infuse{}',
          'tafas_fuse': r'TAFAS+\Infuse{}', 'petsa_fuse': r'PETSA+\Infuse{}', 'cosa_fuse': r'COSA+\Infuse{}',
          'dadapter_fuse': r'$\delta$-Adapter+\Infuse{}'}
    val, rels = {}, defaultdict(list)
    for c, _ in cols:
        recs = [MA.get((c, ds, H)) for H in HS] if c in BB else [EVF.get('%s_%s_%s' % (c, ds, H)) for H in HS]
        if any(r is None for r in recs):
            continue
        for m in meths:
            if all(m in r for r in recs):
                val[(c, m)] = np.mean([r[m][0] for r in recs])
                if m != 'frozen':
                    rels[(c in BB, m)] += [rel(r[m][0], r['frozen'][0]) for r in recs]
    if not val:
        return rels
    cols = [(c, n) for c, n in cols if (c, 'frozen') in val]
    bb_cols = [(c, n) for c, n in cols if c in BB]
    fm_cols = [(c, n) for c, n in cols if c not in BB]
    lines = [r'\begin{table}[t]', r'\centering',
             r'\caption{Azure Functions (per-minute invocations of 50 applications) under the TTA protocol: MSE '
             r'averaged over the four horizons. Best single method per forecaster in \textbf{bold}.}',
             r'\label{tab:azure}', r'\footnotesize', r'\setlength{\tabcolsep}{2.5pt}',
             r'\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}l %s}' % ('c' * len(cols)), r'\toprule',
             r'Method & ' + ' & '.join(n for _, n in cols) + r' \\', r'\midrule']
    top_meths = ['frozen', 'refresh', 'update', 'fuse']
    bot_meths = ['TAFAS', 'PETSA', 'COSA', 'DADAPTER'] + STACK
    for m in top_meths:
        cells = []
        for c, _ in cols:
            single = [mm for mm in meths[1:] if mm not in STACK and (c, mm) in val]
            best = min(single, key=lambda mm: val[(c, mm)])
            cells.append((r'\textbf{%s}' if m == best else '%s') % fmt(val[(c, m)]))
        lines.append('%s & %s \\\\' % (mn[m], ' & '.join(cells)))
    lines.append(r'\midrule')
    for idx, m in enumerate(bot_meths):
        if m == STACK[0]:
            lines.append(r'\cmidrule(lr){1-%d}' % (len(bb_cols) + 1))
        cells = []
        for c, _ in bb_cols:
            single = [mm for mm in meths[1:] if mm not in STACK and (c, mm) in val]
            best = min(single, key=lambda mm: val[(c, mm)])
            cells.append((r'\textbf{%s}' if m == best else '%s') % fmt(val[(c, m)]))
        if fm_cols:
            if idx == 0:
                cells.append(r'\multicolumn{%d}{c}{\multirow{%d}{*}{\makecell{\emph{black-box}\\{\scriptsize(no gradients)}}}}'
                             % (len(fm_cols), len(bot_meths)))
            else:
                cells.append(r'\multicolumn{%d}{c}{}' % len(fm_cols))
        lines.append('%s & %s \\\\' % (mn[m], ' & '.join(cells)))
    lines += [r'\bottomrule', r'\end{tabular*}', r'\end{table}']
    open(PAPER + 'tab_azure.tex', 'w').write('\n'.join(lines) + '\n')
    # per-horizon MSE / MAE for the trained backbones on Azure Functions (appendix; foundation models are in tab_fm_full)
    fl = [r'\begin{table*}[t]', r'\centering',
          r'\caption{Per-horizon MSE / MAE of the trained backbones on Azure Functions under the TTA protocol. '
          r'T/P/C/$\delta$+\Infuse{}: \Infuse{} applied to the forecasts of TAFAS, PETSA, and COSA with their own '
          r'revision disabled, and of $\delta$-Adapter. Best single method in \textbf{bold}.}',
          r'\label{tab:azure_full}', r'\scriptsize\renewcommand{\arraystretch}{0.86}', r'\setlength{\tabcolsep}{1.6pt}',
          r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}ll %s}' % ('c' * len(meths)), r'\toprule',
          r'Backbone & $H$ & ' + ' & '.join(COLN_FLAT[m] for m in meths) + r' \\', r'\midrule']
    for c, n in bb_cols:
        for H in HS:
            r = MA.get((c, ds, H))
            if r is None:
                continue
            single = [m for m in meths[1:] if m not in STACK and m in r]
            best = min(single, key=lambda m: r[m][0])
            cells = [(r'\textbf{%s/%s}' if m == best else '%s/%s') % (fmt(r[m][0]), fmt(r[m][1])) for m in meths]
            fl.append('%s & %s & %s \\\\' % (n if H == HS[0] else '', H, ' & '.join(cells)))
        fl.append(r'\midrule')
    fl[-1] = r'\bottomrule'
    fl += [r'\end{tabular*}', r'\end{table*}']
    open(PAPER + 'tab_azure_full.tex', 'w').write('\n'.join(fl) + '\n')
    for (trained, m), v in rels.items():
        nm = {'TAFAS': 'TAFAS', 'PETSA': 'PETSA', 'COSA': 'COSA', 'DADAPTER': 'DAD', 'refresh': 'Refresh',
              'update': 'InfuseU', 'fuse': 'Infuse', 'tafas_fuse': 'TafasInfuse', 'petsa_fuse': 'PetsaInfuse',
              'cosa_fuse': 'CosaInfuse', 'dadapter_fuse': 'DadInfuse'}[m]
        num('Az%s%s' % ('Tr' if trained else 'FM', nm), float(np.mean(v)))
    print('Azure rel change:', {('trained' if t else 'fm', m): round(float(np.mean(v)), 2) for (t, m), v in rels.items()})
    # serving with a call budget on Azure (H=96)
    for r_ in (12, 24):
        inc_h, inc_f = [], []
        for c, _ in cols:
            ev = MA[(c, ds, '96')]['ev'] if c in BB else EVF.get('%s_%s_96' % (c, ds))
            sv = (ev or {}).get('service')
            if not sv or str(r_) not in sv:
                continue
            inc_h.append(rel(sv[str(r_)][0], sv['1'][0]))
            inc_f.append(rel(sv[str(r_)][3] if len(sv[str(r_)]) > 3 else sv[str(r_)][1], sv['1'][0]))
        if inc_h:
            w = {12: 'Twelve', 24: 'TwentyFour'}[r_]
            num('AzBudgetHold' + w, float(np.mean(inc_h)))
            num('AzBudgetInfuse' + w, float(np.mean(inc_f)))
            print('Azure serving r=%d over %d: hold %+.1f%%, INFUSE %+.1f%%' % (r_, len(inc_h), np.mean(inc_h),
                                                                               np.mean(inc_f)))
    for fm, tg in (('bolt-small', 'S'), ('bolt-base', 'B')):
        sv = (EVF.get('%s_%s_96' % (fm, ds)) or {}).get('service')
        if sv and '48' in sv:
            num('AzBolt%sFortyEight' % tg, rel(sv['48'][3], sv['1'][0]))
    return rels


def family_groups(M, EVF):
    """Setting groups of the summary table: trained backbones on the benchmark, Chronos-Bolt on the benchmark, and
    every forecaster on Azure Functions. Rows are dicts method -> (MSE, MAE)."""
    EL = load_eval('res_elf.jsonl')
    MA = merged(['azure_functions'])

    def fm_row(name, r):
        return add_elf({m: tuple(r[m]) for m in ('frozen', 'refresh', 'update', 'fuse')}, name, EL)
    fm = {n: fm_row(n, r) for n, r in EVF.items() if any(n.startswith('%s_%s_' % (f, d)) for f in FMS for d in DS)}
    az = dict({'%s_%s_%s' % k: r for k, r in MA.items()},
              **{n: fm_row(n, r) for n, r in EVF.items() if '_azure_functions_' in n})
    return [('trained', 'Trained', list(M.values())), ('fm', 'Chronos-Bolt', list(fm.values())),
            ('azure', 'Azure Func.', list(az.values()))]


def table_family(M, EVF):
    """Summary across method families: emission-time online adaptation (gradient adapters, gradient-free ELF), in-flight
    revision of the frozen forecaster, and both combined. Mean relative MSE change w.r.t. the frozen forecaster."""
    G = family_groups(M, EVF)
    rows = [(r'\emph{Emission-time online adaptation}', None, None),
            ('DADAPTER', r'$\delta$-Adapter', False), ('elf', r'ELF', True),
            (r'\emph{In-flight revision of the frozen forecaster}', None, None),
            ('refresh', 'Refresh', True), ('update', r'\InfuseU{} (no model call)', True), ('fuse', r'\Infuse{}', True),
            (r'\emph{Adaptation + in-flight revision}', None, None),
            ('dadapter_refresh', r'$\delta$-Adapter + Refresh', False), ('dadapter_fuse', r'$\delta$-Adapter + \Infuse{}', False),
            ('elf_refresh', 'ELF + Refresh', True), ('elf_fuse', r'ELF + \Infuse{}', True)]
    val = {}
    for g, _, recs in G:
        for key, _, _ in rows:
            if key is None or not key or key.startswith('\\'):
                continue
            rs = [r for r in recs if key in r]
            if len(rs) == len(recs) and recs:
                val[(g, key)] = float(np.mean([rel(r[key][0], r['frozen'][0]) for r in rs]))
    lines = [r'\begin{table}[t]', r'\centering',
             r'\caption{Mean relative MSE change (\%) with respect to the frozen or zero-shot forecaster under the TTA '
             r'protocol, by method family, over the trained-backbone benchmark settings, the Chronos-Bolt benchmark '
             r'settings, and all Azure Functions settings ($n$ in parentheses). G.-f.: gradient-free and applicable to '
             r'black-box forecasters. n/a: gradient-based adapters need white-box access and do not apply to Chronos-Bolt. '
             r'$\delta$-Adapter is the strongest of TAFAS, PETSA, COSA, and $\delta$-Adapter (Table~\ref{tab:main}, '
             r'Azure in Table~\ref{tab:azure}). Best per column in \textbf{bold}.}',
             r'\label{tab:family}', r'\footnotesize', r'\setlength{\tabcolsep}{2.2pt}',
             r'\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lc ccc}', r'\toprule',
             r'Method & G.-f. & ' + ' & '.join('%s (%d)' % (nm, len(recs)) for _, nm, recs in G) + r' \\', r'\midrule']
    best = {g: min((v for (gg, _), v in val.items() if gg == g), default=None) for g, _, _ in G}
    for ri, (key, lab, gf) in enumerate(rows):
        if lab is None:
            if ri:
                lines.append(r'\midrule')
            lines.append(r'\multicolumn{5}{l}{%s} \\' % key)
            continue
        cells = []
        for g, _, _ in G:
            v = val.get((g, key))
            cells.append('' if v is None else ((r'$\mathbf{%+.1f}$' if abs(v - best[g]) < 1e-9 else r'$%+.1f$') % v))
        if cells[1:] == ['', '']:
            cells = cells[:1] + [r'\multicolumn{2}{c}{\emph{n/a}}']
        lines.append(r'\quad %s & %s & %s \\' % (lab, r'\checkmark' if gf else '', ' & '.join(cells)))
    lines += [r'\bottomrule', r'\end{tabular*}', r'\end{table}']
    open(PAPER + 'tab_family.tex', 'w').write('\n'.join(lines) + '\n')
    tag = {'trained': 'Tr', 'fm': 'FM', 'azure': 'Az'}
    for (g, key), v in val.items():
        nm = {'elf': 'Elf', 'elf_refresh': 'ElfRefresh', 'elf_fuse': 'ElfInfuse', 'dadapter_refresh': 'DadRefresh',
              'dadapter_fuse': 'DadInfuse', 'fuse': 'Infuse', 'update': 'InfuseU', 'refresh': 'Refresh'}.get(key)
        if nm:
            num('Fam%s%s' % (tag[g], nm), v)
    # INFUSE stacked on ELF vs ELF alone, and standalone INFUSE vs ELF, per setting
    for g, _, recs in G:
        rs = [r for r in recs if 'elf_fuse' in r]
        if rs:
            num('ElfStack%s' % tag[g], float(np.mean([rel(r['elf_fuse'][0], r['elf'][0]) for r in rs])))
            num('ElfStackBetter%s' % tag[g], int(sum(r['elf_fuse'][0] < r['elf'][0] for r in rs)))
            num('InfuseBeatsElf%s' % tag[g], int(sum(r['fuse'][0] < r['elf'][0] for r in rs)))
            num('N%s' % tag[g], len(rs))
            print('%s: ELF+INFUSE vs ELF %+.2f%% (better in %d/%d), INFUSE better than ELF in %d' % (
                g, np.mean([rel(r['elf_fuse'][0], r['elf'][0]) for r in rs]), sum(r['elf_fuse'][0] < r['elf'][0] for r in rs),
                len(rs), sum(r['fuse'][0] < r['elf'][0] for r in rs)))
    print('family table:', {k: round(v, 2) for k, v in val.items()})
    return val


def table_naive():
    """Naive (last value) reference: MSE averaged over the horizons, per dataset."""
    NV = load_eval('res_naive.jsonl')
    EV = load_eval('res_grid.jsonl')
    if not NV:
        return
    dsets = DS + ['azure_functions']
    lines = [r'\begin{table}[H]', r'\centering',
             r'\caption{Naive (last observed value) reference under the TTA protocol: MSE averaged over the four '
             r'horizons. Best frozen backbone and best backbone + \Infuse{} per dataset for comparison.}',
             r'\label{tab:naive}', r'\scriptsize', r'\setlength{\tabcolsep}{1.8pt}',
             r'\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}l %s}' % ('c' * len(dsets)), r'\toprule',
             r'Method & ' + ' & '.join({'exchange_rate': 'Exch.', 'weather': 'Weath.', 'azure_functions': 'Azure'}.get(d, d)
                                     for d in dsets) + r' \\', r'\midrule']
    rowv = defaultdict(dict)
    for ds in dsets:
        nv = [NV.get('naive_%s_%s' % (ds, H)) for H in HS]
        if any(r is None for r in nv):
            continue
        for m, lab in (('frozen', 'Naive'), ('refresh', 'Naive + Refresh'), ('fuse', r'Naive + \Infuse{}')):
            rowv[lab][ds] = np.mean([r[m][0] for r in nv])
        for m, lab in (('frozen', 'Best frozen backbone'), ('fuse', r'Best backbone + \Infuse{}')):
            vs = [np.mean([EV['%s_%s_%s' % (bb, ds, H)][m][0] for H in HS]) for bb in BB
                  if all('%s_%s_%s' % (bb, ds, H) in EV for H in HS)]
            if vs:
                rowv[lab][ds] = min(vs)
    for lab, d in rowv.items():
        lines.append('%s & %s \\\\' % (lab, ' & '.join(fmt(d[ds]) if ds in d else '--' for ds in dsets)))
    lines += [r'\bottomrule', r'\end{tabular*}', r'\end{table}']
    open(PAPER + 'tab_naive.tex', 'w').write('\n'.join(lines) + '\n')
    print('naive:', {lab: {k: round(v, 3) for k, v in d.items()} for lab, d in rowv.items()})


def fig_autoscale():
    """Downstream autoscaling on Azure Functions (autoscale.json): over-provisioning vs SLA-violation rate."""
    if not os.path.exists('autoscale.json'):
        return
    A = json.load(open('autoscale.json'))
    R, d = A['res'], '10'
    fcs = [('iTransformer', 'iTrans.'), ('PatchTST', 'PatchTST'), ('DLinear', 'DLinear'), ('bolt-small', 'Bolt-S'),
           ('bolt-base', 'Bolt-B')]
    styles = [('hold_r1', r'frozen, call every min.', '#4D4D4D', None), ('hold_r12', r'frozen, hold ($r=12$)', 'white', '////'),
              ('fuse_r12', r'frozen + \textsc{Infuse} ($r=12$)', C_OURS, None),
              ('ELF_r1', r'ELF, call every min.', '#9ECAE1', None), ('ELFhold_r12', r'ELF, hold ($r=12$)', 'white', '\\\\\\\\'),
              ('ELFfuse_r12', r'ELF + \textsc{Infuse} ($r=12$)', C_BLUE, None)]
    fig, ax = plt.subplots(figsize=(3.33, 1.75), layout='constrained')
    w = 0.13
    for si, (key, lab, col, hatch) in enumerate(styles):
        xs = [i + (si - 2.5) * w for i, (fc, _) in enumerate(fcs) if key in R.get(fc, {})]
        ys = [100 * R[fc][key][d]['over_at_0.05'] for fc, _ in fcs if key in R.get(fc, {})]
        ec = {'hold_r12': '#4D4D4D', 'ELFhold_r12': C_BLUE}.get(key, 'none')
        ax.bar(xs, ys, w, color=col, edgecolor=ec, hatch=hatch, lw=0.5 if hatch else 0, label=lab)
        if key == 'fuse_r12':
            for x_, (fc, _) in zip(xs, [f for f in fcs if key in R.get(f[0], {})]):
                y_ = 100 * R[fc][key][d]['over_at_0.05']
                b_ = 100 * R[fc]['hold_r1'][d]['over_at_0.05']
                chg = 100 * (y_ - b_) / b_
                ax.text(x_, y_ + 0.35, (r'$-$%.1f\%%' % abs(chg)) if chg < 0 else ('+%.1f\\%%' % chg),
                        ha='center', va='bottom', fontsize=4.6, fontweight='bold', color=C_OURS, rotation=90)
    ax.set_xticks(range(len(fcs))); ax.set_xticklabels([n for _, n in fcs]); ax.tick_params(axis='x', length=0)
    ax.set_ylim(16, 35.5)
    ax.set_ylabel(r'Idle capacity at 5\% SLA viol.\ (\%)', fontsize=6.5)
    ax.legend(loc='upper left', ncol=2, fontsize=5.4, handlelength=1.2, columnspacing=0.7, handletextpad=0.3,
              borderaxespad=0.2)
    fig.savefig(PAPER + 'figs/fig_autoscale.pdf')
    plt.close(fig)
    # over-provisioning at a 5% violation rate, averaged over the five forecasters
    keys = ['hold_r1', 'fuse_r1', 'hold_r12', 'update_r12', 'fuse_r12', 'fuse_pt_r12', 'hold_r24', 'fuse_r24', 'ELF_r1',
            'DADAPTER_r1', 'ELFhold_r12', 'ELFfuse_r12', 'ELFfuse_r1', 'ELFhold_r24', 'ELFfuse_r24']
    word = {'1': 'One', '12': 'Twelve', '24': 'TwentyFour'}

    def mac(k):                                    # e.g. hold_r12 -> HoldRTwelve
        m, r = k.rsplit('_r', 1)
        m = {'fuse_pt': 'FusePt'}.get(m, m.capitalize())
        return m + 'R' + word[r]
    for suf, grp in (('', list(R)), ('Tr', [fc for fc in R if fc in BB]), ('FM', [fc for fc in R if fc in FMS])):
        for k in keys:
            fcs = [fc for fc in grp if k in R[fc]]
            if not fcs:
                continue
            x = np.array([R[fc][k][d]['over_at_0.05'] for fc in fcs])
            base = np.array([R[fc]['hold_r1'][d]['over_at_0.05'] for fc in fcs])
            num('As%s%s' % (mac(k), suf), 100 * float(np.nanmean(x)))
            num('AsRel%s%s' % (mac(k), suf), float(np.nanmean(100 * (x - base) / base)))
    print('autoscale over-provisioning at 5% violations (d=10):',
          {fc: {k: round(100 * R[fc][k][d]['over_at_0.05'], 2) for k in R[fc]} for fc in R})


def table_full(M):
    cols = [c for c in COLS if any(c in M[k] for k in M)]
    head = [r'\begin{table*}[t]', r'\centering', None, None, r'\scriptsize\renewcommand{\arraystretch}{0.86}',
            r'\setlength{\tabcolsep}{1.6pt}',
            r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}ll %s}' % ('c' * len(cols)), r'\toprule',
            r'Data & $H$ & ' + ' & '.join(COLN_FLAT[c] for c in cols) + r' \\', r'\midrule']
    for bb in BB:
        lines = list(head)
        lines[2] = (r'\caption{Per-horizon MSE / MAE under the TTA protocol, %s backbone. T/P/C/$\delta$+\Infuse{}: '
                    r'\Infuse{} applied to the forecasts of TAFAS, PETSA, and COSA with their own revision disabled, and '
                    r'of $\delta$-Adapter. Best single method in \textbf{bold}.}' % bb)
        lines[3] = r'\label{tab:full_%s}' % bb
        for ds in DS:
            for H in HS:
                k = (bb, ds, H)
                if k not in M:
                    continue
                row = M[k]
                av = sorted([c for c in cols[1:] if c in row and c not in STACK], key=lambda c: row[c][0])
                cells = []
                for c in cols:
                    if c not in row:
                        cells.append('--')
                        continue
                    s = '%s/%s' % (fmt(row[c][0]), fmt(row[c][1]))
                    cells.append(r'\textbf{%s}' % s if av and c == av[0] else s)
                lines.append('%s & %s & %s \\\\' % (DSN[ds] if H == HS[0] else '', H, ' & '.join(cells)))
            lines.append(r'\midrule')
        lines[-1] = r'\bottomrule'
        lines += [r'\end{tabular*}', r'\end{table*}']
        open(PAPER + 'tab_full_%s.tex' % bb, 'w').write('\n'.join(lines) + '\n')


def table_fm(EVF):
    meths = ['frozen', 'refresh', 'update', 'fuse']
    mn = {'frozen': 'Zero-shot', 'refresh': 'Refresh', 'update': r'\InfuseU{}', 'fuse': r'\Infuse{}'}
    dsh = {'ETTh1': 'ETTh1', 'ETTh2': 'ETTh2', 'ETTm1': 'ETTm1', 'ETTm2': 'ETTm2', 'exchange_rate': 'Exch.',
           'weather': 'Weath.'}
    lines = [r'\begin{table}[t]', r'\centering',
             r'\caption{Black-box foundation models (zero-shot Chronos-Bolt) under the TTA protocol: MSE averaged '
             r'over the four horizons, and mean relative change over the 24 settings of each model. Gradient-based '
             r'adapters do not apply.}',
             r'\label{tab:fm}', r'\footnotesize', r'\setlength{\tabcolsep}{2pt}',
             r'\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}l cccccc r}', r'\toprule',
             r'Method & ' + ' & '.join(dsh[d] for d in DS) + r' & Rel. \\', r'\midrule']
    agg = defaultdict(list)
    wins = defaultdict(int)
    for fm in FMS:
        vals = {}
        for ds in DS:
            recs = [r for r in (EVF.get('%s_%s_%s' % (fm, ds, H)) for H in HS) if r is not None]
            if recs:
                vals[ds] = {m: np.mean([r[m][0] for r in recs]) for m in meths}
                for r in recs:
                    for m in meths[1:]:
                        agg[(fm, m)].append(rel(r[m][0], r['frozen'][0]))
                    wins[(fm, min(meths[1:], key=lambda m: r[m][0]))] += 1
        lines.append(r'\multicolumn{8}{l}{\emph{%s}} \\' % FMN[fm])
        for m in meths:
            cells = []
            for ds in DS:
                if ds not in vals:
                    cells.append('--')
                    continue
                best = min(meths[1:], key=lambda mm: vals[ds][mm])
                cells.append((r'\textbf{%s}' if m == best else '%s') % fmt(vals[ds][m]))
            r_ = '--' if m == 'frozen' else (r'$%+.1f\%%$' % np.mean(agg[(fm, m)]) if agg[(fm, m)] else '--')
            lines.append(r'\quad %s & %s & %s \\' % (mn[m], ' & '.join(cells), r_))
        if fm != FMS[-1]:
            lines.append(r'\midrule')
    lines += [r'\bottomrule', r'\end{tabular*}', r'\end{table}']
    open(PAPER + 'tab_fm.tex', 'w').write('\n'.join(lines) + '\n')
    return agg, wins


def table_fm_full(EVF):
    meths = ['frozen', 'refresh', 'update', 'fuse']
    mn = {'frozen': 'Zero-shot', 'refresh': 'Refresh', 'update': r'\InfuseU{}', 'fuse': r'\Infuse{}'}
    nm = len(meths)
    lines = [r'\begin{table*}[t]', r'\centering',
             r'\caption{Per-horizon MSE / MAE of the zero-shot foundation models under the TTA protocol (six benchmark '
             r'datasets, Azure Functions, and large-scale Electricity and Traffic at $H=96$). Best in \textbf{bold}.}',
             r'\label{tab:fm_full}', r'\scriptsize\renewcommand{\arraystretch}{0.82}', r'\setlength{\tabcolsep}{1.5pt}',
             r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}ll %s %s}' % ('c' * nm, 'c' * nm), r'\toprule',
             r'& & ' + ' & '.join(r'\multicolumn{%d}{c}{%s}' % (nm, FMN[fm]) for fm in FMS) + r' \\',
             r'\cmidrule(lr){3-%d}\cmidrule(lr){%d-%d}' % (nm + 2, nm + 3, 2 * nm + 2),
             r'Data & $H$ & ' + ' & '.join(mn[m] for m in meths) + ' & ' + ' & '.join(mn[m] for m in meths) + r' \\',
             r'\midrule']

    def cells(r):
        if r is None:
            return ['--'] * nm
        best = min(meths[1:], key=lambda m: r[m][0])
        return [(r'\textbf{%s/%s}' if m == best else '%s/%s') % (fmt(r[m][0]), fmt(r[m][1])) for m in meths]
    for ds in DS + ['azure_functions']:
        for H in HS:
            row = [c for fm in FMS for c in cells(EVF.get('%s_%s_%s' % (fm, ds, H)))]
            lines.append('%s & %s & %s \\\\' % (DSN[ds] if H == HS[0] else '', H, ' & '.join(row)))
        lines.append(r'\midrule')
    for ds, nmds in (('electricity', 'Electricity'), ('traffic', 'Traffic')):
        recs = [EVF.get('%s_%s_96' % (fm, ds)) for fm in FMS]
        if any(r is not None for r in recs):
            lines.append('%s & 96 & %s \\\\' % (nmds, ' & '.join(c for r in recs for c in cells(r))))
    if lines[-1] == r'\midrule':
        lines.pop()
    lines += [r'\bottomrule', r'\end{tabular*}', r'\end{table*}']
    open(PAPER + 'tab_fm_full.tex', 'w').write('\n'.join(lines) + '\n')


def fmt_p(p):
    if p >= 1e-3:
        return '%.3f' % p
    e = int(np.floor(np.log10(p)))
    return r'%.0f\times10^{%d}' % (p / 10 ** e, e)


def significance(M):
    """Paired Wilcoxon signed-rank tests over the benchmark settings: relative MSE difference between INFUSE (and
    INFUSE-U) and each adapter."""
    from scipy.stats import wilcoxon
    for meth, tag in (('fuse', 'Infuse'), ('update', 'InfuseU')):
        ps = []
        for t in ADP:
            ks = [k for k in M if t in M[k]]
            if len(ks) < 10:
                continue
            d = [rel(M[k][meth][0], M[k][t][0]) for k in ks]
            p = float(wilcoxon(d, alternative='less').pvalue)
            ps.append(p)
            NUM['PWilc%s%s' % (tag, {'DADAPTER': 'DAD'}.get(t, t))] = fmt_p(p)
            print('Wilcoxon %s < %s: n=%d p=%.2e' % (meth, t, len(ks), p))
        if ps:
            NUM['PWilcMax%s' % tag] = fmt_p(max(ps))
    # settings that share a backbone and a dataset are not independent: one value per (backbone, dataset), averaging
    # the relative differences over the four horizons, tested against every adapter and against refresh
    for meth, tag in (('fuse', 'Infuse'), ('update', 'InfuseU')):
        ps = {}
        for t in ADP + ['refresh']:
            groups = defaultdict(list)
            for k in M:
                if t in M[k]:
                    groups[(k[0], k[1])].append(rel(M[k][meth][0], M[k][t][0]))
            d = [np.mean(v) for v in groups.values()]
            if len(d) >= 10:
                ps[t] = float(wilcoxon(d, alternative='less').pvalue)
                print('grouped Wilcoxon %s < %s: n=%d groups, p=%.2e, %d/%d groups better' % (
                    meth, t, len(d), ps[t], sum(v < 0 for v in d), len(d)))
        if ps:
            NUM['PGroupMax%s' % tag] = fmt_p(max(v for t, v in ps.items() if t != 'refresh'))
            if 'refresh' in ps:
                NUM['PGroup%sRefresh' % tag] = fmt_p(ps['refresh'])
            NUM['NGroups'] = '%d' % len({(k[0], k[1]) for k in M})


def table_abl(M, EVF):
    recs = [M[k]['ev'] for k in M] + list(EVF.values())
    RV = load_eval('res_revise.jsonl')

    def collect(rs):
        out = defaultdict(list)
        for r in rs:
            fr = r['frozen'][0]
            out['full'].append(rel(r['update'][0], fr))
            for k, v in r.get('ablate', {}).items():
                out[k].append(rel(v, fr))
            for m in ('refresh', 'fuse', 'fuse_eq', 'fuse_bg'):
                out[m].append(rel(r[m][0], fr))
            rv = RV.get(os.path.basename(r['path'])[:-4])
            for m in ('last', 'mean', 'ar1'):
                if rv is not None:
                    out['rule_' + m].append(rel(rv[m][0], fr))
        return out
    allr = collect([r for r in recs if 'ablate' in r])
    sub = collect([r for r in recs if 'per_channel' in r.get('ablate', {})])
    names = [('rule_last', r'Latest revealed error carried forward'), ('rule_mean', r'Mean revealed error'),
             ('rule_ar1', r'AR(1) error decay'), ('full', r'\InfuseU{} (cone + progressive innovation)'),
             ('single_u', r'\quad w/o cone \& progressive blocks'),
             ('no_shrink', r'\quad w/o shrinkage (OLS)'), ('shrink_0.3', r'\quad fixed shrinkage $\lambda=0.3$'),
             ('forget_0.99', r'\quad forgetting factor $0.99$'), ('per_channel', r'\quad per-channel covariance'),
             ('refresh', r'Refresh'), ('fuse', r'\Infuse{} (cone + excess-precision)'),
             ('single_f', r'\quad w/o excess-precision \& cone ($c=0$)'),
             ('fuse_eq', r'\quad equal weights'), ('fuse_bg', r'\quad Bates--Granger weights (online fit)')]
    n_all = len(allr['full']); n_sub = len(sub['full'])
    lines = [r'\begin{table}[t]', r'\centering',
             r'\caption{Revision rules and ablations: mean relative MSE change with respect to the frozen forecaster '
             r'under the TTA protocol, over all trained-backbone and foundation-model settings and over the $H=96$ '
             r'settings, where the per-channel variant was run. The first block uses no model call.}',
             r'\label{tab:abl}', r'\small',
             r'\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrr}', r'\toprule',
             r'Variant & All ($n=%d$) & $H=96$ ($n=%d$) \\' % (n_all, n_sub), r'\midrule']
    for k, nm in names:
        a = r'$%+.1f\%%$' % np.mean(allr[k]) if allr[k] and k != 'per_channel' else '--'
        b = r'$%+.1f\%%$' % np.mean(sub[k]) if sub[k] else '--'
        lines.append(r'%s & %s & %s \\' % (nm, a, b))
        if k == 'per_channel':
            lines.append(r'\midrule')
    # how often INFUSE-U beats every revision rule
    rv_win = [all(rel(r['update'][0], r['frozen'][0]) < rel(RV[b][m][0], r['frozen'][0]) for m in ('last', 'mean', 'ar1'))
              for r in recs if 'ablate' in r for b in [os.path.basename(r['path'])[:-4]] if b in RV]
    if rv_win:
        num('RuleBeat', int(sum(rv_win)))
        num('RuleN', len(rv_win))
    lines += [r'\bottomrule', r'\end{tabular*}', r'\end{table}']
    open(PAPER + 'tab_abl.tex', 'w').write('\n'.join(lines) + '\n')
    out = {'all': {k: np.mean(v) for k, v in allr.items()}, 'h96': {k: np.mean(v) for k, v in sub.items()}}
    fig_ablation(out, n_all, n_sub)
    num('AblNAll', n_all); num('AblNSub', n_sub)
    return out


# ------------------------------------------------------------------------------------------- figures
def fig_ablation(ab, n_all, n_sub):
    """Compact two-panel horizontal bar chart of MSE reduction (%) over all benchmark settings (n=120):
    (a) zero-call revision (INFUSE-U and ablations/rules), (b) with a fresh model call (INFUSE and fusion ablations)."""
    left_items = [
        ('full', r'\textbf{\textsc{Infuse-U} (full)}', True),
        ('single_u', r'w/o cone \& prog.\ cov.', False),
        ('shrink_0.3', r'fixed shrinkage ($\lambda{=}0.3$)', False),
        ('no_shrink', r'w/o shrinkage (OLS)', False),
        ('forget_0.99', r'forgetting ($\rho{=}0.99$)', False),
        ('rule_mean', r'mean error shift', False),
        ('rule_ar1', r'AR(1) error decay', False),
    ]
    right_items = [
        ('fuse', r'\textbf{\textsc{Infuse} (full)}', True),
        ('single_f', r'w/o excess-prec.\ \& cone', False),
        ('fuse_eq', r'equal weights ($0.5/0.5$)', False),
        ('fuse_bg', r'online Bates--Granger', False),
        ('refresh', r'refresh only', False),
    ]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.38, 1.48), layout='constrained',
                                   gridspec_kw={'width_ratios': [1.05, 1.0]})

    # Panel (a): Zero-call revision (INFUSE-U)
    for yi, (key, lab, ours) in enumerate(left_items):
        v = -float(ab['all'].get(key, 0.0))
        col = C_OURS if ours else ('#7F7F7F' if key.startswith('rule_') else '#4B6A88')
        ax1.barh(yi, v, height=0.66, color=col, edgecolor='none', alpha=0.95 if ours else 0.82)
        ax1.text(v + 0.22, yi, ('%.1f\\%%' % v), va='center', ha='left', fontsize=5.5,
                 fontweight='bold' if ours else 'normal', color=C_OURS if ours else '#222222')
    ax1.set_yticks(range(len(left_items)))
    ax1.set_yticklabels([it[1] for it in left_items], fontsize=5.7)
    ax1.invert_yaxis()
    ax1.tick_params(axis='y', length=0)
    ax1.set_xlim(0, 13.5)
    ax1.set_xticks([0, 5, 10])
    ax1.grid(axis='x'); ax1.grid(False, axis='y')
    ax1.set_xlabel(r'MSE reduction (\%)', fontsize=6.5, labelpad=1.5)
    ax1.set_title(r'\textbf{(a)} 0-call (\textsc{Infuse-U})', fontsize=6.8, pad=3)

    # Panel (b): With a fresh call (INFUSE) + H=96 pooling inset at bottom
    for yi, (key, lab, ours) in enumerate(right_items):
        v = -float(ab['all'].get(key, 0.0))
        col = C_OURS if ours else (C_REF if key == 'refresh' else '#4B6A88')
        ax2.barh(yi, v, height=0.62, color=col, edgecolor='none', alpha=0.95 if ours else 0.85)
        ax2.text(v + 0.25, yi, ('%.1f\\%%' % v), va='center', ha='left', fontsize=5.5,
                 fontweight='bold' if ours else 'normal', color=C_OURS if ours else '#222222')
    # Add H=96 pooling vs per-channel comparison on the bottom two rows of Panel (b)
    v_pool96 = -float(ab['h96'].get('full', 0.0))
    v_pc96 = -float(ab['h96'].get('per_channel', 0.0))
    y_base = len(right_items) + 0.35
    ax2.axhline(y_base - 0.5, color='#CCCCCC', lw=0.5, ls=':')
    for dy, lab, v, ours in ((0, r'pooled cov.\ ($H{=}96$)', v_pool96, True),
                             (1, r'per-channel ($H{=}96$)', v_pc96, False)):
        yy = y_base + dy
        col = C_BLUE if ours else '#7F7F7F'
        ax2.barh(yy, v, height=0.58, color=col, edgecolor='none', alpha=0.88)
        ax2.text(v + 0.25, yy, ('%.1f\\%%' % v), va='center', ha='left', fontsize=5.4,
                 fontweight='bold' if ours else 'normal', color=C_BLUE if ours else '#333333')
    yt_b = list(range(len(right_items))) + [y_base, y_base + 1]
    yl_b = [it[1] for it in right_items] + [r'pooled cov.\ ($H{=}96$)', r'per-channel ($H{=}96$)']
    ax2.set_yticks(yt_b)
    ax2.set_yticklabels(yl_b, fontsize=5.7)
    ax2.set_ylim(-0.55, y_base + 1.55)
    ax2.invert_yaxis()
    ax2.tick_params(axis='y', length=0)
    ax2.set_xlim(0, 17.5)
    ax2.set_xticks([0, 5, 10, 15])
    ax2.grid(axis='x'); ax2.grid(False, axis='y')
    ax2.set_xlabel(r'MSE reduction (\%)', fontsize=6.5, labelpad=1.5)
    ax2.set_title(r'\textbf{(b)} Fusion \& pooling', fontsize=6.8, pad=3)

    fig.savefig(PAPER + 'figs/fig_abl.pdf')
    plt.close(fig)


def fig_teaser(M):
    rows = list(M.values())

    def avg(c):
        v = [rel(r[c][0], r['frozen'][0]) for r in rows if c in r]
        return (np.mean(v), len(v)) if v else (np.nan, 0)
    items = []                                     # (label, value, colour, hatched, starts a new group)
    for t in TTA:
        items.append((t, avg(t)[0], C_TTA[t], False, True))
        items.append((t + ', revision disabled', avg(t + '-noadj')[0], C_TTA[t], True, False))
    items.append((r'$\delta$-Adapter (no revision step)', avg('DADAPTER')[0], C_TTA['DADAPTER'], False, True))
    items += [('Refresh (frozen model)', avg('refresh')[0], C_REF, False, True),
              (r'\textsc{Infuse-U} (no model call)', avg('update')[0], C_OURS, True, False),
              (r'\textsc{Infuse}', avg('fuse')[0], C_OURS, False, False)]
    items += [(r'%s + \textsc{Infuse}' % {'DADAPTER': r'$\delta$-Adapter'}.get(t, t), avg(t.lower() + '_fuse')[0],
               '#A63D00', False, t == ADP[0]) for t in ADP]
    items = [it for it in items if not np.isnan(it[1])]
    fig, ax = plt.subplots(figsize=(3.33, 2.45), layout='constrained')
    ypos = []
    y = 0.0
    for i, (lab, v, c, hatched, new_group) in enumerate(items):
        if i > 0 and new_group:
            y += 0.45
        ypos.append(y)
        ax.barh(y, v, height=0.72, color='white' if hatched else c, edgecolor=c, hatch='/////' if hatched else None,
                linewidth=0.8)
        lab_v = (r'$-$%.1f\%%' if v < 0 else r'%.1f\%%') % abs(v)
        if v < -5 and not hatched:       # long solid bar: value inside, at its far end
            ax.text(v + 0.2, y, lab_v, va='center', ha='left', fontsize=6.3,
                    color=c if hatched else 'white', bbox=dict(fc='white', ec='none', pad=0.6) if hatched else None)
        else:
            ax.text(v - 0.15, y, lab_v, va='center', ha='right', fontsize=6.3)
        y += 1.0
    ax.set_yticks(ypos)
    ax.set_yticklabels([it[0] for it in items], fontsize=6.6)
    ax.invert_yaxis()
    ax.axvline(0, color='black', lw=0.6)
    ax.set_xlabel(r'Mean relative MSE change vs.\ frozen model (\%)')
    ax.set_title(r'\textbf{(b)} Uses of revealed targets on the TTA benchmark', loc='right')
    lo = min(it[1] for it in items)
    ax.set_xlim(lo * 1.12, 0.3)
    ax.grid(axis='x'); ax.grid(False, axis='y')
    ax.tick_params(axis='y', length=0)
    fig.savefig(PAPER + 'figs/fig_teaser.pdf')
    plt.close(fig)
    return {it[0]: it[1] for it in items}


def fig_fresh(M, EVF):
    fig, axs = plt.subplots(1, 2, figsize=(3.33, 1.72), layout='constrained', gridspec_kw={'width_ratios': [1, 1.15]})
    ax = axs[0]
    xs, ys = [], []
    CA = load_eval('res_causal.jsonl')
    for src, recs, mk, col in (('bb', [M[k]['ev'] for k in M], 'o', C_BLUE), ('fm', list(EVF.values()), '^', C_OURS)):
        # prediction from the lead-wise MSE of the calibration period (first part of the stream), measured on the rest
        cs = [CA.get(os.path.basename(r['path'])[:-4]) for r in recs]
        cs = [c for c in cs if c is not None and not c.get('skipped')]
        if not cs:
            continue
        px = [c['pred_ref'] for c in cs]
        py = [c['act_ref'] for c in cs]
        xs += px; ys += py
        ax.scatter(px, py, s=7, marker=mk, color=col, alpha=0.75, lw=0,
                   label='trained' if src == 'bb' else 'foundation')
    lim = [min(xs + ys) - 2, 1]
    ax.plot(lim, lim, color='#4D4D4D', lw=0.6, ls=(0, (3, 2)))
    ax.set_xlim(lim); ax.set_ylim(lim)
    tk = np.arange(0, lim[0], -10)[::-1]
    ax.set_xticks(tk); ax.set_yticks(tk)
    ax.set_xlabel(r'Predicted by Eq.~(1) (\%)')
    ax.set_ylabel(r'Measured (\%)')
    ax.grid(axis='x')
    ax.legend(loc='upper left', fontsize=6, handletextpad=0.1, borderaxespad=0.1)
    xs, ys = np.array(xs), np.array(ys)
    r2 = 1 - ((ys - xs) ** 2).sum() / ((ys - ys.mean()) ** 2).sum()
    num('NCausal', len(xs))
    num('CausalMAE', float(np.abs(ys - xs).mean()))
    ax.set_title(r'\textbf{(a)} Refresh gain')
    ax = axs[1]
    w = 0.38
    for bi, bb in enumerate(BB):
        for ti, t in enumerate(TTA):
            ks = [k for k in M if k[0] == bb and t in M[k] and (t + '-noadj') in M[k]]
            if not ks:
                continue
            a = np.mean([rel(M[k][t][0], M[k]['frozen'][0]) for k in ks])
            b = np.mean([rel(M[k][t + '-noadj'][0], M[k]['frozen'][0]) for k in ks])
            xp = bi * 3.6 + ti * 1.1
            ax.bar(xp - w / 2, a, w, color=C_TTA[t], lw=0, label=t if bi == 0 else None)
            ax.bar(xp + w / 2, b, w, color='white', edgecolor=C_TTA[t], hatch='/////', lw=0.6)
    ax.axhline(0, color='black', lw=0.6)
    ax.set_xticks([1.1, 4.7, 8.3])
    ax.set_xticklabels(['iTrans.', 'PatchTST', 'DLinear'])
    ax.tick_params(axis='x', length=0)
    ax.set_ylabel(r'Rel.\ MSE change (\%)')
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.45 * (hi - lo))
    ax.legend(loc='upper center', fontsize=6, handlelength=0.9, ncol=3, columnspacing=0.6, handletextpad=0.3)
    ax.set_title(r'\textbf{(b)} With / without revision')
    fig.savefig(PAPER + 'figs/fig_fresh.pdf')
    plt.close(fig)
    return r2


def fig_budget(M, EVF):
    panels = [('iTransformer_ETTh1_96', 'iTransformer, ETTh1'), ('DLinear_weather_96', 'DLinear, Weather'),
              ('bolt-base_ETTm2_96', 'Chronos-Bolt-B, ETTm2'),
              ('iTransformer_azure_functions_96', 'iTransformer, Azure'),
              ('bolt-small_azure_functions_96', 'Chronos-Bolt-S, Azure')]
    MA = merged(['azure_functions'])
    allev = {**{'%s_%s_%s' % k: M[k]['ev'] for k in M}, **EVF, **{'%s_%s_%s' % k: MA[k]['ev'] for k in MA},
             **{k: v for k, v in load_eval('res_fm.jsonl').items() if 'azure' in k}}
    panels = [(k, t) for k, t in panels if k in allev and 'service' in allev[k]]
    stats = {}
    if not panels:
        return stats
    fig, axs = plt.subplots(1, len(panels), figsize=(7.0, 1.6), layout='constrained')
    axs = np.atleast_1d(axs)
    for ax, (key, title) in zip(axs, panels):
        sv = allev[key]['service']
        rs = sorted(int(r) for r in sv)
        calls = [100.0 / r for r in rs]
        hold = [sv[str(r)][0] for r in rs]
        upd = [sv[str(r)][3] if len(sv[str(r)]) > 3 else sv[str(r)][1] for r in rs]
        ax.plot(calls, hold, '-o', color=C_GREY, ms=2.6, lw=1.0, mfc='white', label='hold')
        ax.plot(calls, upd, '-o', color=C_OURS, ms=2.6, lw=1.3, label=r'\textsc{Infuse}')
        ax.axhline(sv['1'][0], color='#4D4D4D', lw=0.5, ls=(0, (3, 2)))
        ax.set_xscale('log')
        ax.set_xticks([100, 25, 8.3, 2.1])
        ax.set_xticklabels(['100', '25', '8.3', '2.1'])
        ax.minorticks_off()
        ax.invert_xaxis()
        ax.set_title(title, fontsize=7.5)
        ax.set_xlabel('Model calls per 100 steps')
        stats[key] = {r: (sv[str(r)][0], sv[str(r)][1]) for r in rs}
    axs[0].set_ylabel('MSE of served forecast')
    axs[0].legend(loc='upper left', fontsize=6.5)
    fig.savefig(PAPER + 'figs/fig_budget.pdf')
    plt.close(fig)
    return stats


NUM = {}


def num(name, value, fmt_='%.1f'):
    """Register a number quoted in the text (written to numbers.tex as \\newcommand)."""
    if isinstance(value, (int, np.integer)):
        NUM[name] = '%d' % value
    else:
        s = fmt_ % value
        NUM[name] = s.replace('-', '$-$') if s.startswith('-') else s
        NUM[name + 'Abs'] = fmt_ % abs(value)


def write_numbers():
    with open(PAPER + 'numbers.tex', 'w') as f:
        for k, v in sorted(NUM.items()):
            f.write('\\newcommand{\\%s}{%s}\n' % (k, v))


def budget_factor(sv, tol=0.01):
    """Largest call interval r at which the served MSE stays within tol of calling every step."""
    base = sv['1'][0]
    best = 1
    for r in sorted(int(x) for x in sv):
        v = sv[str(r)][3] if len(sv[str(r)]) > 3 else sv[str(r)][1]
        if v <= base * (1 + tol):
            best = r
    return best


def fig_teaser3(M, path='dumps/fm/bolt-small_azure_functions_96.npz', i=3010, ch=32, k=24, x0=-30):
    """Figure 1: (a) one real in-flight forecast on Azure Functions (Chronos-Bolt-S) revised by refresh and INFUSE-U;
    (b) adaptation vs. revision on the TTA benchmark; (c) accuracy vs. per-batch cost."""
    import inflight as I
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D
    rows = list(M.values())

    def red(c):                                    # mean MSE reduction (%) over the benchmark settings
        v = [-rel(r[c][0], r['frozen'][0]) for r in rows if c in r]
        return float(np.mean(v)) if v else np.nan
    fig = plt.figure(figsize=(3.33, 3.05), layout='constrained')
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.08], width_ratios=[1.0, 1.0])
    # ---------------------------------------------------------------- (a)
    ax = fig.add_subplot(gs[0, :])
    y, pred, gt, E, L, H = I.load(path)
    C = pred.shape[2]
    cov = I.ErrorCov(H, C)
    for j in range(0, i - H + 1):
        cov.add(E[j])
    G = cov.gain(k, cov.sigma())
    upd = pred[i, k:, ch] + G @ E[i, :k, ch]
    ref = pred[i + k, :H - k, ch]
    obs = y[i + L + x0:i + L + H, ch]
    ax.axvspan(0, k, color='#EEEEEE', lw=0, zorder=0)
    l_obs, = ax.plot(np.arange(x0, H), obs, color='black', lw=0.8, zorder=3)
    l_iss, = ax.plot(np.arange(H), pred[i, :, ch], color=C_GREY, lw=1.1, ls=(0, (3, 1.5)), zorder=4)
    l_ref, = ax.plot(np.arange(k, H), ref, color=C_REF, lw=1.1, zorder=4)
    l_upd, = ax.plot(np.arange(k, H), upd, color=C_OURS, lw=1.4, zorder=5)
    ax.axvline(k, color=C_OURS, lw=0.6, ls=':')
    lo_, hi_ = ax.get_ylim()
    ax.set_ylim(lo_, hi_ + 0.12 * (hi_ - lo_))
    ax.text(k / 2, hi_ + 0.06 * (hi_ - lo_), 'revealed', ha='center', va='center', fontsize=6, color='#4D4D4D')
    ax.text(k + 1.2, hi_ + 0.06 * (hi_ - lo_), r'now ($k=%d$): %d leads pending' % (k, H - k), ha='left',
            va='center', fontsize=6, color=C_OURS)
    xa = 10
    y_obs, y_iss = float(np.mean(obs[xa - x0 - 3:xa - x0 + 4])), float(pred[i, xa, ch])
    ax.annotate('', xy=(xa, y_obs), xytext=(xa, y_iss),
                arrowprops=dict(arrowstyle='<->', color='#4D4D4D', lw=0.6, shrinkA=0, shrinkB=0), zorder=6)
    ax.text(-2.5, 0.5 * (y_obs + y_iss) + 0.3, 'load jump\nmissed', fontsize=6, color='#4D4D4D', va='center',
            ha='right', linespacing=0.95)
    ax.set_xlim(x0, H - 1)
    ax.set_xlabel('Minutes after the forecast was issued', labelpad=1)
    ax.set_ylabel('Invocations (std.)', labelpad=1)
    ax.set_title(r'\textbf{(a)} An in-flight forecast of a serverless application (Azure Functions)', fontsize=7,
                 pad=13)
    ax.legend([l_obs, l_iss, l_ref, l_upd], ['observed', 'issued forecast', 'refresh (model call)',
                                            r'\textsc{Infuse-U} (no call)'],
              loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=4, fontsize=5.8, handlelength=1.4,
              columnspacing=0.8, handletextpad=0.4, borderaxespad=0.15, frameon=False)
    # ---------------------------------------------------------------- (b)
    ax = fig.add_subplot(gs[1, 0])
    x, w = 0.0, 0.36
    ticks, labels = [], []
    for t in TTA:
        a, b = red(t), red(t + '-noadj')
        ax.bar(x - w / 2, a, w, color=C_TTA[t], lw=0)
        ax.bar(x + w / 2, b, w, color='white', edgecolor=C_TTA[t], hatch='//////', lw=0.6)
        ticks.append(x); labels.append(t)
        x += 1.0
    if not np.isnan(red('DADAPTER')):
        v = red('DADAPTER')
        ax.bar(x, v, 0.52, color=C_TTA['DADAPTER'], lw=0)
        ticks.append(x); labels.append(r'$\delta$-Ad.')
        x += 1.0
    for c, lab, col in (('refresh', 'Refresh', C_REF), ('fuse', r'\textsc{Infuse}', C_OURS)):
        v = red(c)
        ax.bar(x, v, 0.6, color=col, lw=0)
        ax.text(x, v + 0.3, '%.1f' % v, ha='center', va='bottom', fontsize=5.6)
        ticks.append(x); labels.append(lab)
        x += 1.0
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=5.8, rotation_mode='anchor')
    ax.tick_params(axis='x', length=0, pad=1.5)
    ax.set_ylabel(r'MSE reduction (\%)', labelpad=1)
    ax.set_ylim(0, max(red('fuse'), 1) * 1.22)
    ax.legend(handles=[Patch(facecolor='#9E9E9E', lw=0, label='as published'),
                       Patch(facecolor='white', edgecolor='#9E9E9E', hatch='//////', lw=0.6,
                             label='revision disabled')],
              loc='upper left', fontsize=5.6, handlelength=1.2, borderaxespad=0.2, handletextpad=0.4)
    ax.set_title(r'\textbf{(b)} Adaptation vs.\ revision', fontsize=7)
    # ---------------------------------------------------------------- (c)
    ax = fig.add_subplot(gs[1, 1])
    T = json.load(open('timing.json'))['rows'] if os.path.exists('timing.json') else []
    tm = defaultdict(list)
    for r_ in T:
        tm[r_['method']].append(r_['ms'])
    pts = [('TAFAS', 'TAFAS', C_TTA['TAFAS'], 'o', (-4, 5), 'right'), ('PETSA', 'PETSA', C_TTA['PETSA'], 'o', (3, 6), 'left'),
           ('COSA', 'COSA', C_TTA['COSA'], 'o', (4, -5), 'left'),
           ('update', 'INFUSE-U', C_OURS, '*', (5, -5), 'left'), ('fuse', 'INFUSE', C_OURS, '*', (5, 0), 'left')]
    for c, key, col, mk, off, ha in pts:
        if key not in tm or np.isnan(red(c)):
            continue
        xx, yy = float(np.exp(np.mean(np.log(tm[key])))), red(c)
        ax.scatter(xx, yy, s=46 if mk == '*' else 22, marker=mk, color=col, zorder=3, lw=0)
        lab = {'update': r'\textsc{Infuse-U}', 'fuse': r'\textsc{Infuse}'}.get(c, c)
        ax.annotate(lab, (xx, yy), xytext=off, textcoords='offset points', fontsize=5.8, va='center', ha=ha)
    ax.set_xscale('log')
    ax.set_xlim(0.8, 200)
    ax.set_ylim(0, max(red('fuse'), 1) * 1.22)
    ax.set_xlabel('Time per batch (ms, log)', labelpad=1)
    ax.set_ylabel(r'MSE reduction (\%)', labelpad=1)
    ax.grid(axis='x', which='major')
    ax.annotate('', xy=(1.1, max(red('fuse'), 1) * 1.15), xytext=(6, max(red('fuse'), 1) * 0.72),
                arrowprops=dict(arrowstyle='->', color='#9E9E9E', lw=0.6))
    ax.text(1.25, max(red('fuse'), 1) * 0.86, 'better', fontsize=5.6, color='#7F7F7F', rotation=-38)
    ax.text(0.97, 0.97, 'adapters: GPU\n\\textsc{Infuse}: 1 CPU thread', transform=ax.transAxes, ha='right',
            va='top', fontsize=5.2, color='#4D4D4D', linespacing=0.95)
    ax.set_title(r'\textbf{(c)} Accuracy vs.\ cost', fontsize=7)
    fig.savefig(PAPER + 'figs/fig_teaser.pdf')
    plt.close(fig)


def fig_example(path='dumps/fm/bolt-small_azure_functions_96.npz', i=3010, ch=32, k=24, x0=-40):
    """Figure 1a: one real in-flight forecast on Azure Functions (Chronos-Bolt-S), with refresh and INFUSE-U after
    k revealed targets; the covariance is estimated causally from the forecasts fully revealed before origin i."""
    import inflight as I
    if not os.path.exists(path):
        return
    y, pred, gt, E, L, H = I.load(path)
    C = pred.shape[2]
    cov = I.ErrorCov(H, C)
    for j in range(0, i - H + 1):
        cov.add(E[j])
    G = cov.gain(k, cov.sigma())
    upd = pred[i, k:, ch] + G @ E[i, :k, ch]
    ref = pred[i + k, :H - k, ch]
    obs = y[i + L + x0:i + L + H, ch]
    fig, ax = plt.subplots(figsize=(3.33, 1.62), layout='constrained')
    ax.axvspan(0, k, color='#EDEDED', lw=0, zorder=0)
    ax.plot(np.arange(x0, H), obs, color='black', lw=0.9, label='Observed load', zorder=3)
    ax.plot(np.arange(H), pred[i, :, ch], color=C_GREY, lw=1.2, ls=(0, (3, 1.5)), label='Issued forecast', zorder=4)
    ax.plot(np.arange(k, H), ref, color=C_REF, lw=1.2, label='Refresh (new model call)', zorder=4)
    ax.plot(np.arange(k, H), upd, color=C_OURS, lw=1.5, label=r'\textsc{Infuse-U} (no model call)', zorder=5)
    ax.axvline(0, color='#4D4D4D', lw=0.6, ls=':')
    ax.axvline(k, color=C_OURS, lw=0.6, ls=':')
    lo, hi = ax.get_ylim()
    top = hi + 0.42 * (hi - lo)
    ax.set_ylim(lo, top)
    ax.text(0.8, top - 0.04 * (top - lo), 'issued', ha='left', va='top', fontsize=6.3, color='#4D4D4D')
    ax.text(k + 0.8, top - 0.04 * (top - lo), r'now: %d targets revealed, %d pending' % (k, H - k), ha='left',
            va='top', fontsize=6.3, color=C_OURS)
    xa = 12
    y_obs, y_iss = float(np.mean(obs[xa - x0 - 3:xa - x0 + 4])), float(pred[i, xa, ch])
    ax.annotate('', xy=(xa, y_obs), xytext=(xa, y_iss),
                arrowprops=dict(arrowstyle='<->', color='#4D4D4D', lw=0.6, shrinkA=0, shrinkB=0), zorder=6)
    ax.text(-4, 0.5 * (y_obs + y_iss) + 0.35, 'missed\nload jump', fontsize=6, color='#4D4D4D', va='center', ha='right')
    ax.set_xlim(x0, H - 1)
    ax.set_xlabel('Minutes after the forecast was issued')
    ax.set_ylabel('Invocations (std.)')
    ax.legend(loc='upper left', bbox_to_anchor=(0.0, 0.9), fontsize=6, ncol=2, handlelength=1.5, columnspacing=0.8,
              borderaxespad=0.2, frameon=True, facecolor='white', edgecolor='none', framealpha=0.92)
    ax.set_title(r'\textbf{(a)} An in-flight forecast of a serverless application')
    fig.savefig(PAPER + 'figs/fig_example.pdf')
    plt.close(fig)
    print('example: issued %.3f refresh %.3f update %.3f (pending-lead MSE)' % (
        float((E[i, k:, ch] ** 2).mean()), float(((gt[i, k:, ch] - ref) ** 2).mean()),
        float(((gt[i, k:, ch] - upd) ** 2).mean())))


def fig_framework_thumbs(path='dumps/grid/iTransformer_ETTh1_96.npz', k=24, n_cov=1500):
    """Small plots embedded in the framework figure (figs/framework.tex), all from one real setting: fully revealed
    error trajectories, the pooled error correlation, one conditional update with its 1-sigma band, and the fusion
    weights w_j."""
    import inflight as I
    if not os.path.exists(path):
        return
    y, pred, gt, E, L, H = I.load(path)
    n, _, C = pred.shape
    cov = I.ErrorCov(H, C)
    for j in range(n_cov):
        cov.add(E[j])
    Sig = cov.sigma()
    s = np.sqrt(cov.ss / cov.wc) + 1e-8
    G = cov.gain(k, Sig)
    lead = np.arange(1, H + 1)

    def thumb(w=1.32, h=0.72):
        fig, ax = plt.subplots(figsize=(w, h), layout='constrained')
        ax.tick_params(labelsize=5, length=1.5, pad=1)
        ax.grid(False)
        return fig, ax
    # (1) fully revealed error trajectories (normalized), a few channels and origins
    fig, ax = thumb()
    rng = np.random.default_rng(3)
    for j, c in zip(rng.choice(n_cov, 6, replace=False), rng.choice(C, 6)):
        ax.plot(lead, E[j, :, c] / s[c], lw=0.6, color=C_BLUE, alpha=0.75)
    ax.axhline(0, color='black', lw=0.4)
    ax.set_xticks([1, 96]); ax.set_yticks([])
    ax.set_xlabel('lead', fontsize=5, labelpad=0)
    fig.savefig(PAPER + 'figs/thumb_errors.pdf'); plt.close(fig)
    # (2) pooled correlation of the error trajectory
    fig, ax = thumb(0.86, 0.86)
    d = np.sqrt(np.diag(Sig))
    ax.imshow(Sig / d[:, None] / d[None, :], cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.4)
    fig.savefig(PAPER + 'figs/thumb_cov.pdf'); plt.close(fig)
    # (3) conditional update of one in-flight forecast: revealed errors, predicted pending errors with 1-sigma band
    best, bi, bc = -1.0, None, None
    for i in range(n_cov + H, n - k, 3):
        for c in range(C):
            g = (E[i, k:, c] ** 2).mean() - ((E[i, k:, c] - G @ E[i, :k, c]) ** 2).mean()
            if g > best:
                best, bi, bc = g, i, c
    e = E[bi, :, bc] / s[bc]
    cm = G @ e[:k]
    cv = np.maximum(np.diag(Sig)[k:] - np.einsum('ij,ji->i', G, Sig[:k, k:]), 0)
    fig, ax = thumb()
    ax.plot(lead[k:], e[k:], lw=0.5, color='#BDBDBD')
    ax.fill_between(lead[k:], cm - np.sqrt(cv), cm + np.sqrt(cv), color=C_OURS, alpha=0.18, lw=0)
    ax.plot(lead[k:], cm, lw=1.0, color=C_OURS)
    ax.plot(lead[:k], e[:k], 'o', ms=1.4, color=C_BLUE, mew=0)
    ax.axhline(0, color='black', lw=0.4)
    ax.axvline(k + 0.5, color='#4D4D4D', lw=0.4, ls=':')
    ax.set_xticks([1, k, 96]); ax.set_yticks([])
    ax.set_xlabel('lead', fontsize=5, labelpad=0)
    fig.savefig(PAPER + 'figs/thumb_update.pdf'); plt.close(fig)
    # (4) fusion on the same forecast: updated forecast, refreshed forecast, and their combination
    v_upd = np.maximum(np.diag(Sig)[k:] - np.einsum('ij,ji->i', G, Sig[:k, k:]), 1e-12)
    w = v_upd / (v_upd + np.diag(Sig)[:H - k])
    upd = pred[bi, k:, bc] + (G @ E[bi, :k, bc])
    ref = pred[bi + k, :H - k, bc]
    fig, ax = thumb()
    ax.plot(lead[k:], gt[bi, k:, bc], lw=0.6, color='#9E9E9E')
    ax.plot(lead[k:], upd, lw=0.8, color=C_OURS, ls=(0, (2.5, 1.2)))
    ax.plot(lead[k:], ref, lw=0.8, color=C_REF)
    ax.plot(lead[k:], (1 - w) * upd + w * ref, lw=1.1, color='#A63D00')
    ax.set_xlim(1, H)
    ax.set_xticks([1, k, 96]); ax.set_yticks([])
    ax.set_xlabel('lead', fontsize=5, labelpad=0)
    fig.savefig(PAPER + 'figs/thumb_fusion.pdf'); plt.close(fig)
    print('framework thumbs: window %d channel %d, pending MSE issued %.3f update %.3f refresh %.3f fused %.3f' % (
        bi, bc, float((E[bi, k:, bc] ** 2).mean()), float(((gt[bi, k:, bc] - upd) ** 2).mean()),
        float(((gt[bi, k:, bc] - ref) ** 2).mean()), float(((gt[bi, k:, bc] - (1 - w) * upd - w * ref) ** 2).mean())))


def fig_case(path='dumps/grid/iTransformer_ETTh1_96.npz', ch=-1, ks=(12, 36)):
    """One real in-flight forecast revised by INFUSE-U as its targets arrive (covariance estimated causally)."""
    import inflight as I
    if not os.path.exists(path):
        return
    y, pred, gt, E, L, H = I.load(path)
    n, _, C = pred.shape
    # choose, among windows late enough for a causal covariance, the one where the revision helps most on `ch`
    cov = I.ErrorCov(H, C)
    best, best_i, best_cov = -1.0, None, None
    snap = {}
    start = 600
    for i in range(n):
        if i - H >= 0:
            cov.add(E[i - H])
        if i >= start and i % 7 == 0:
            Sig = cov.sigma()
            k = ks[0]
            e_upd = E[i, k:, ch] - cov.gain(k, Sig) @ E[i, :k, ch]
            gain = (E[i, k:, ch] ** 2).mean() - (e_upd ** 2).mean()
            if gain > best:
                best, best_i, best_cov = gain, i, Sig.copy()
    i, Sig = best_i, best_cov
    t = np.arange(1, H + 1)
    fig, ax = plt.subplots(figsize=(3.33, 1.55), layout='constrained')
    ax.plot(t, gt[i, :, ch], color='black', lw=1.0, label='Observed')
    ax.plot(t, pred[i, :, ch], color=C_GREY, lw=1.1, ls=(0, (3, 1.5)), label='Issued forecast')
    cols = ['#E69F00', C_OURS]
    for k, col in zip(ks, cols):
        G = np.linalg.solve(Sig[:k, :k] + 1e-6 * np.trace(Sig[:k, :k]) / k * np.eye(k), Sig[:k, k:]).T
        rev = pred[i, k:, ch] + G @ E[i, :k, ch]
        ax.plot(t[k:], rev, color=col, lw=1.3, label=r'Revised after %d steps' % k)
        ax.axvline(k + 0.5, color=col, lw=0.5, ls=':')
    ax.set_xlim(1, H)
    ax.set_xlabel('Lead (hours)')
    ax.set_ylabel('Value (std.)')
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.42 * (hi - lo))
    ax.legend(loc='upper center', fontsize=6, ncol=2, handlelength=1.4, columnspacing=0.8, borderaxespad=0.1)
    fig.savefig(PAPER + 'figs/fig_case.pdf')
    plt.close(fig)
    print('case study window', i, 'gain', round(best, 4))


def fig_azure_time(bb='iTransformer', H=96, win=180):
    """Rolling MSE over the Azure Functions test period (Friday morning to Sunday night) for the frozen forecaster,
    TAFAS (emitted forecasts), refresh, and INFUSE; weekend shaded."""
    import inflight as I
    path = 'dumps/grid/%s_azure_functions_%d.npz' % (bb, H)
    tpath = 'dumps/grid/tta/TAFAS-noadj_%s_azure_functions_%d.npz' % (bb, H)
    if not (os.path.exists(path) and os.path.exists('dumps/series_azure_functions.npz')):
        return
    y, pred, gt, E, L, H_ = I.load(path)
    B = I.tafas_batches(y, pred.shape[0], L)
    curves = {'frozen': (E ** 2).mean((1, 2))}
    if os.path.exists(tpath):
        pt = np.load(tpath)['pred'].astype(np.float32)
        curves['TAFAS'] = ((gt - pt) ** 2).mean((1, 2))
    for m in ('refresh', 'fuse'):
        curves[m] = I.tafas(y, pred, gt, E, L, H_, mode=m, batches=B, return_window=True)
    S = np.load('dumps/series_azure_functions.npz')
    ts, st = int(S['test_start']), S['stamp']
    n = len(curves['frozen'])
    stamp = st[ts + L:ts + L + n]                                   # [month, day, weekday, hour] of each origin
    hours = np.arange(n) / 60.0
    kern = np.ones(win) / win
    fig, ax = plt.subplots(figsize=(3.33, 1.45), layout='constrained')
    wk = stamp[:, 2] >= 5
    edges = np.flatnonzero(np.diff(np.r_[0, wk.astype(int), 0]))
    for a, b in zip(edges[::2], edges[1::2]):
        ax.axvspan(hours[a], hours[min(b, n - 1)], color='#F2F2F2', lw=0, zorder=0)
    sty = {'TAFAS': ('TAFAS', C_TTA['TAFAS'], '-', 1.1), 'refresh': ('Refresh', C_REF, '-', 1.1),
           'fuse': (r'\textsc{Infuse}', C_OURS, '-', 1.5)}
    hx = hours[win // 2:n - win // 2 + 1]
    base = np.convolve(curves['frozen'], kern, mode='valid')[:len(hx)]
    for m in ('TAFAS', 'refresh', 'fuse'):
        if m not in curves:
            continue
        lab, col, ls, lw = sty[m]
        ax.plot(hx, 100 * (np.convolve(curves[m], kern, mode='valid')[:len(hx)] / base - 1), color=col, ls=ls, lw=lw,
                label=lab)
    ax.axhline(0, color='black', lw=0.6)
    days = np.flatnonzero(np.diff(stamp[:, 1]) != 0) + 1
    ax.set_xticks(hours[days])
    ax.set_xticklabels(['%s 0:00' % ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][int(stamp[d, 2])] for d in days])
    ax.set_xlim(0, hours[-1])
    ax.set_ylabel(r'MSE vs.\ frozen (\%)', labelpad=1)
    ax.set_xlabel('Azure Functions test period (3-hour windows, weekend shaded)', labelpad=1)
    lo_, hi_ = ax.get_ylim()
    ax.set_ylim(lo_, max(hi_, 0) + 0.32 * (max(hi_, 0) - lo_))
    ax.legend(loc='upper left', ncol=3, fontsize=5.8, handlelength=1.3, columnspacing=0.8, borderaxespad=0.2)
    fig.savefig(PAPER + 'figs/fig_azure_time.pdf')
    plt.close(fig)
    rel_we = {m: float(rel(curves[m][wk].mean(), curves['frozen'][wk].mean())) for m in curves if m != 'frozen'}
    rel_wd = {m: float(rel(curves[m][~wk].mean(), curves['frozen'][~wk].mean())) for m in curves if m != 'frozen'}
    num('AzWeekendInfuse', -rel_we['fuse'], '%.0f'); num('AzWeekdayInfuse', -rel_wd['fuse'], '%.0f')
    if 'TAFAS' in rel_we:
        num('AzWeekendTafas', -rel_we['TAFAS'], '%.1f'); num('AzWeekdayTafas', -rel_wd['TAFAS'], '%.1f')
    print('Azure over time: weekend', {k: round(v, 1) for k, v in rel_we.items()}, 'weekday',
          {k: round(v, 1) for k, v in rel_wd.items()})


def explained_share():
    """Share of the pending error variance explained by the first 24 revealed errors (pooled covariance, H=96)."""
    import inflight as I
    expl = defaultdict(list)
    for path in [p for p in [*('dumps/grid/%s_%s_96.npz' % (bb, ds) for bb in BB for ds in DS + ['azure_functions']),
                             *('dumps/fm/%s_%s_96.npz' % (fm, ds) for fm in FMS for ds in DS + ['azure_functions'])]
                 if os.path.exists(p)]:
        E = I.load(path)[3]
        Z = E / np.sqrt((E ** 2).mean((0, 1), keepdims=True))
        S = np.einsum('nhc,ngc->hg', Z, Z) / (Z.shape[0] * Z.shape[2])
        k = 24
        frac = np.trace(S[:k, k:].T @ np.linalg.solve(S[:k, :k], S[:k, k:])) / np.trace(S[k:, k:])
        expl['Azure' if 'azure' in path else ('FM' if '/fm/' in path else 'Trained')].append(100 * frac)
    for g, v in expl.items():
        num('Expl' + g, float(np.mean(v)), '%.0f')


def fig_analysis2(M, EVF):
    """(a) pooled error correlation across leads (iTransformer, ETTh1); (b) INFUSE-U gain predicted by the Gaussian
    model from the error covariance vs. measured gain, all settings; (c) gain on the pending leads vs. the number k of
    revealed targets (H=96); (d) gain per lead (H=96)."""
    import inflight as I
    AX = json.load(open('analysis_extra.json')) if os.path.exists('analysis_extra.json') else {}
    MA = merged(['azure_functions'])
    EVA = {k: v for k, v in load_eval('res_fm.jsonl').items() if 'azure' in k}
    groups = {'trained': {'%s_%s_%s' % k: M[k]['ev'] for k in M},
              'foundation': EVF,
              'Azure': {**{'%s_%s_%s' % k: MA[k]['ev'] for k in MA}, **EVA}}
    gcol = {'trained': C_BLUE, 'foundation': C_OURS, 'Azure': '#009E73'}
    gmk = {'trained': 'o', 'foundation': '^', 'Azure': 's'}
    fig, axs = plt.subplots(1, 4, figsize=(7.0, 1.78), layout='constrained',
                            gridspec_kw={'width_ratios': [0.95, 1.1, 1.1, 1.1]})
    # (a)
    ax = axs[0]
    y, pred, gt, E, L, H = I.load('dumps/grid/iTransformer_ETTh1_96.npz')
    Z = E / np.sqrt((E ** 2).mean((0, 1), keepdims=True))
    S = np.einsum('nhc,ngc->hg', Z, Z) / (Z.shape[0] * Z.shape[2])
    d = np.sqrt(np.diag(S))
    im = ax.imshow(S / d[:, None] / d[None, :], cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest')
    ax.set_xticks([0, 47, 95]); ax.set_xticklabels(['1', '48', '96'])
    ax.set_yticks([0, 47, 95]); ax.set_yticklabels(['1', '48', '96'])
    ax.set_xlabel('Lead'); ax.set_ylabel('Lead', labelpad=0); ax.grid(False)
    ax.set_title(r'\textbf{(a)} Error correlation', fontsize=7.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, ticks=[-1, 0, 1])
    cb.ax.tick_params(labelsize=5.5, length=1.5); cb.outline.set_linewidth(0.4)
    # (b)
    ax = axs[1]
    xs, ys = [], []
    CA = load_eval('res_causal.jsonl')
    fz = []
    for g, ev in groups.items():
        px, py = [], []
        for name, r in ev.items():
            c = CA.get(name)
            if c is None or c.get('skipped'):
                continue
            # predicted from the calibration-period covariance, measured (online INFUSE-U) on the rest of the stream
            px.append(c['pred_upd']); py.append(c['act_upd'])
            fz.append((c['act_upd'], c['act_upd_frozen']))
        if px:
            ax.scatter(px, py, s=7, marker=gmk[g], color=gcol[g], alpha=0.75, lw=0, label=g)
            xs += px; ys += py
    if xs:
        xs, ys = np.array(xs), np.array(ys)
        slope = float((xs * ys).sum() / (xs * xs).sum())
        r2 = float(np.corrcoef(xs, ys)[0, 1] ** 2)
        lim = [min(xs.min(), ys.min()) - 2, 1]
        ax.plot(lim, lim, color='#4D4D4D', lw=0.6, ls=(0, (3, 2)))
        ax.plot(lim, [slope * v for v in lim], color='#9E9E9E', lw=0.6)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.text(0.97, 0.05, r'$r^2=%.2f$' % r2, transform=ax.transAxes, ha='right', va='bottom', fontsize=6.3)
        num('RTwoUpd', r2, '%.2f'); num('SlopeUpd', 100 * slope, '%.0f')
        print('update gain: predicted vs measured r2=%.3f slope=%.2f n=%d' % (r2, slope, len(xs)))
    if fz:
        fz = np.array(fz)
        num('FrozenCovOnline', float(fz[:, 0].mean())); num('FrozenCovFrozen', float(fz[:, 1].mean()))
        num('FrozenCovMaxDiff', float(np.abs(fz[:, 0] - fz[:, 1]).max()))
        print('INFUSE-U on the evaluation part: online cov %.2f%%, cov frozen after calibration %.2f%% (n=%d, max |diff| '
              '%.2f)' % (fz[:, 0].mean(), fz[:, 1].mean(), len(fz), np.abs(fz[:, 0] - fz[:, 1]).max()))
    ax.set_xlabel(r'Predicted by Prop.~5.1 (\%)'); ax.set_ylabel(r'Measured (\%)', labelpad=0)
    ax.grid(axis='x')
    ax.legend(loc='upper left', fontsize=5.6, handletextpad=0.1, borderaxespad=0.15, markerscale=1.3)
    ax.set_title(r'\textbf{(b)} \textsc{Infuse-U} gain', fontsize=7.5)
    # (c)
    ax = axs[2]
    for g, ev in groups.items():
        curves_c = [AX[n].get('gain_k_cone', AX[n]['gain_k']) for n in ev if n.endswith('_96') and 'gain_k' in AX.get(n, {})]
        curves_s = [AX[n]['gain_k'] for n in ev if n.endswith('_96') and 'gain_k' in AX.get(n, {})]
        if not curves_c:
            continue
        ks = sorted(int(k) for k in curves_c[0])
        Cv = np.array([[c[str(k)] for k in ks] for c in curves_c])
        med = np.median(Cv, 0)
        ax.fill_between(ks, np.percentile(Cv, 25, 0), np.percentile(Cv, 75, 0), color=gcol[g], alpha=0.14, lw=0)
        ax.plot(ks, med, color=gcol[g], lw=1.25, marker=gmk[g], ms=2.4, label=g)
        if any('gain_k_cone' in AX.get(n, {}) for n in ev if n.endswith('_96')):
            Cv_s = np.array([[c[str(k)] for k in ks] for c in curves_s])
            ax.plot(ks, np.median(Cv_s, 0), color=gcol[g], lw=0.85, ls=':', alpha=0.85)
        for kk in (24,):
            if kk in ks:
                num('GainK%s%s' % ({'trained': 'Tr', 'foundation': 'FM', 'Azure': 'Az'}[g], 'TwentyFour'),
                    float(-med[ks.index(kk)]), '%.0f')
    ax.axhline(0, color='black', lw=0.5)
    ax.set_xlabel('Revealed targets $k$'); ax.set_ylabel(r'Pending-lead MSE (\%)', labelpad=0)
    ax.set_xlim(0, 65); ax.set_xticks([1, 16, 32, 48, 64])
    ax.legend(loc='lower left', fontsize=5.5, handlelength=1.2, borderaxespad=0.15)
    ax.set_title(r'\textbf{(c)} Gain vs.\ revealed steps $k$', fontsize=7.5)
    # (d) Gain across forecast horizons H in {96, 192, 336, 720}
    ax = axs[3]
    series_d = [
        ('DADAPTER', r'$\delta$-Ad.', C_TTA['DADAPTER'], ':', 's'),
        ('refresh', 'Refresh', C_REF, '-', 'o'),
        ('single_u', r'w/o cone', '#7F7F7F', ':', '^'),
        ('update', r'\textsc{Infuse-U}', C_OURS, '--', 'D'),
        ('fuse', r'\textsc{Infuse}', C_OURS, '-', '*'),
    ]
    for m, lab, col, ls, mk in series_d:
        ys_h = []
        for h in HS:
            ks_h = [k for k in M if k[2] == h]
            if m == 'single_u':
                vals_h = [rel(M[k]['ev']['ablate']['single_u'], M[k]['frozen'][0]) for k in ks_h
                          if 'single_u' in M[k]['ev'].get('ablate', {})]
            else:
                vals_h = [rel(M[k][m][0], M[k]['frozen'][0]) for k in ks_h if m in M[k]]
            ys_h.append(float(np.mean(vals_h)) if vals_h else np.nan)
        ax.plot(range(len(HS)), ys_h, ls=ls, color=col, lw=1.25 if m in ('update', 'fuse') else 1.0,
                marker=mk, ms=4.2 if mk == '*' else 2.8, label=lab)
    ax.axhline(0, color='black', lw=0.5)
    ax.set_xticks(range(len(HS))); ax.set_xticklabels(HS)
    ax.set_xlabel('Forecast horizon $H$'); ax.set_ylabel(r'MSE change (\%)', labelpad=0)
    ax.set_ylim(-23.0, 0.5)
    ax.legend(loc='lower center', ncol=2, fontsize=5.0, handlelength=1.3, columnspacing=0.6,
              handletextpad=0.3, borderaxespad=0.12, labelspacing=0.2)
    ax.set_title(r'\textbf{(d)} Gain vs.\ horizon $H$', fontsize=7.5)
    fig.savefig(PAPER + 'figs/fig_analysis.pdf')
    plt.close(fig)


def fig_analysis(M, EVF):
    """(a) pooled error correlation across leads for a trained backbone and a foundation model;
    (b) lead-wise relative MSE change of refresh / INFUSE-U / INFUSE averaged over the H=96 settings."""
    import inflight as I
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 1.72), layout='constrained', gridspec_kw={'width_ratios': [1, 1.18, 2.4]})
    for ax, (path, title) in zip(axs[:2], [('dumps/grid/iTransformer_ETTh1_96.npz', 'iTransformer, ETTh1'),
                                         ('dumps/fm/bolt-small_ETTh1_96.npz', 'Chronos-Bolt-S, ETTh1')]):
        if not os.path.exists(path):
            continue
        y, pred, gt, E, L, H = I.load(path)
        Z = E / np.sqrt((E ** 2).mean((0, 1), keepdims=True))
        S = np.einsum('nhc,ngc->hg', Z, Z) / (Z.shape[0] * Z.shape[2])
        d = np.sqrt(np.diag(S))
        R = S / d[:, None] / d[None, :]
        im = ax.imshow(R, cmap='RdBu_r', vmin=-1, vmax=1, origin='upper', interpolation='nearest')
        ax.set_xticks([0, 47, 95]); ax.set_xticklabels(['1', '48', '96'])
        ax.set_yticks([0, 47, 95]); ax.set_yticklabels(['1', '48', '96'])
        ax.set_xlabel('Lead'); ax.grid(False)
        ax.set_title(title, fontsize=7.5)
    axs[0].set_ylabel('Lead')
    cb = fig.colorbar(im, ax=axs[1], fraction=0.05, pad=0.03)
    cb.ax.tick_params(labelsize=6); cb.outline.set_linewidth(0.4)
    # share of the pending error variance explained by the first 24 revealed errors (pooled covariance, H=96)
    expl = defaultdict(list)
    for path in [p for p in [*('dumps/grid/%s_%s_96.npz' % (bb, ds) for bb in BB for ds in DS + ['azure_functions']),
                             *('dumps/fm/%s_%s_96.npz' % (fm, ds) for fm in FMS for ds in DS + ['azure_functions'])]
                 if os.path.exists(p)]:
        E = I.load(path)[3]
        Z = E / np.sqrt((E ** 2).mean((0, 1), keepdims=True))
        S = np.einsum('nhc,ngc->hg', Z, Z) / (Z.shape[0] * Z.shape[2])
        k = 24
        frac = np.trace(S[:k, k:].T @ np.linalg.solve(S[:k, :k], S[:k, k:])) / np.trace(S[k:, k:])
        expl['Azure' if 'azure' in path else ('FM' if '/fm/' in path else 'Trained')].append(100 * frac)
    for g, v in expl.items():
        num('Expl' + g, float(np.mean(v)), '%.0f')
    ax = axs[2]
    recs = [M[k]['ev'] for k in M if k[2] == '96'] + [v for k, v in EVF.items() if k.endswith('_96')]
    if recs:
        for m, lab, col, ls in (('refresh', 'Refresh', C_REF, '-'), ('update', r'\textsc{Infuse-U}', C_OURS, '--'),
                                ('fuse', r'\textsc{Infuse}', C_OURS, '-')):
            curves = [100 * (np.array(r[m + '_lead']) - np.array(r['frozen_lead'])) / np.array(r['frozen_lead'])
                      for r in recs if len(r[m + '_lead']) == 96]
            ax.plot(np.arange(1, 97), np.mean(curves, 0), ls, color=col, lw=1.3, label=lab)
        ax.axhline(0, color='black', lw=0.5)
        ax.set_xlabel('Lead'); ax.set_ylabel(r'Rel.\ MSE change (\%)')
        ax.set_xlim(1, 96); ax.set_xticks([1, 24, 48, 72, 96])
        ax.legend(loc='lower left', fontsize=6.5)
        ax.set_title(r'Per lead, all $H=96$ settings', fontsize=7.5)
    fig.savefig(PAPER + 'figs/fig_analysis.pdf')
    plt.close(fig)


def main():
    M = merged()
    EVF = load_eval('res_fm.jsonl')
    print('benchmark settings with all results: %d   FM evals: %d' % (len(M), len(EVF)))
    if M:
        dev = [abs(M[k]['frozen'][0] - M[k]['frozen_harness'][0]) for k in M if M[k].get('frozen_harness')]
        print('frozen: dump vs harness max |diff| = %.5f' % (max(dev) if dev else -1))
        wins, relsum = table_main(M)
        table_full(M)
        num('NSettings', len(M))
        for c, nm in (('TAFAS', 'RelTAFAS'), ('PETSA', 'RelPETSA'), ('COSA', 'RelCOSA'), ('DADAPTER', 'RelDAD'),
                      ('refresh', 'RelRefresh'), ('update', 'RelInfuseU'), ('fuse', 'RelInfuse'),
                      ('cosa_fuse', 'RelCosaInfuse')):
            if relsum.get(c):
                num(nm, float(np.mean(relsum[c])))
        num('NDAD', sum(1 for k in M if 'DADAPTER' in M[k]))
        num('WinsInfuse', wins['fuse'])
        print('wins:', dict(wins))
        print('mean rel change:', {c: round(float(np.mean(v)), 2) for c, v in relsum.items()})
        for meth, nm in (('fuse', 'BetterInfuse'), ('update', 'BetterInfuseU'), ('refresh', 'BetterRefresh')):
            n_better = sum(1 for k in M if M[k][meth][0] < best_adapter(M[k]))
            num(nm, n_better)
            print('%s better than best adapter: %d / %d' % (meth, n_better, len(M)))
        significance(M)
        # refresh uses the same revealed information as INFUSE: direct comparison
        num('InfuseBeatsRefresh', int(sum(M[k]['fuse'][0] < M[k]['refresh'][0] for k in M)))
        num('InfuseVsRefresh', float(np.mean([rel(M[k]['fuse'][0], M[k]['refresh'][0]) for k in M])))
        fm_ = [r for n, r in EVF.items() if any(n.startswith('%s_%s_' % (f, d)) for f in FMS for d in DS)]
        if fm_:
            num('FMInfuseBeatsRefresh', int(sum(r['fuse'][0] < r['refresh'][0] for r in fm_)))
        for t in ADP:
            ks = [k for k in M if t.lower() + '_refresh' in M[k]]
            if ks:
                nm = {'DADAPTER': 'Dad'}.get(t, t.capitalize())
                num('%sRefreshVs%s' % (nm, nm), float(np.mean([rel(M[k][t.lower() + '_refresh'][0], M[k][t][0])
                                                               for k in ks])))
        # INFUSE applied to each adapter's emitted forecasts (adapter's own revision disabled)
        gains = []
        for t in ADP:
            c, nm = t.lower() + '_fuse', {'DADAPTER': 'Dad'}.get(t, t.capitalize())
            ks = [k for k in M if c in M[k] and t in M[k]]
            if not ks:
                continue
            st = [rel(M[k][c][0], M[k][t][0]) for k in ks]
            gains.append(-np.mean(st))
            num('%sInfuseVs%s' % (nm, nm), float(np.mean(st)))
            num('%sInfuseBetter' % nm, int(sum(s < 0 for s in st)))
            num('%sInfuseBetterBest' % nm, int(sum(M[k][c][0] < best_adapter(M[k]) for k in ks)))
            num('Rel%sInfuse' % nm, float(np.mean([rel(M[k][c][0], M[k]['frozen'][0]) for k in ks])))
            num('N%sInfuse' % nm, len(ks))
            print('%s+INFUSE vs %s: mean %+.2f%% (n=%d), better in %d, better than best adapter in %s' % (
                t, t, np.mean(st), len(st), sum(s < 0 for s in st), NUM['%sInfuseBetterBest' % nm]))
        if gains:
            NUM['StackRange'] = '%.1f--%.1f' % (min(gains), max(gains))
        full_stacks = [c for c in STACK if all(c in M[k] for k in M)]
        if full_stacks:
            bs = min(full_stacks, key=lambda c: np.mean([rel(M[k][c][0], M[k]['frozen'][0]) for k in M]))
            NUM['BestStack'] = {'tafas_fuse': r'TAFAS+\Infuse{}', 'petsa_fuse': r'PETSA+\Infuse{}',
                                'cosa_fuse': r'COSA+\Infuse{}', 'dadapter_fuse': r'$\delta$-Adapter+\Infuse{}'}[bs]
            num('BestStackRel', float(np.mean([rel(M[k][bs][0], M[k]['frozen'][0]) for k in M])))
            num('BestStackBetterBest', int(sum(M[k][bs][0] < best_adapter(M[k]) for k in M)))
        # settings in which INFUSE is not more accurate than the best adapter
        lose = [k for k in M if M[k]['fuse'][0] >= best_adapter(M[k])]
        num('NLoseInfuse', len(lose))
        num('NLoseInfuseLong', sum(1 for k in lose if k[2] == '720'))
        num('NLoseStackRecovers', sum(1 for k in lose if min(M[k][c][0] for c in STACK if c in M[k])
                                      < best_adapter(M[k])))
        print('INFUSE not better than best adapter:', sorted(lose))
        dsm = {'ETTh1': 'ETThOne', 'ETTh2': 'ETThTwo', 'ETTm1': 'ETTmOne', 'ETTm2': 'ETTmTwo',
               'exchange_rate': 'Exchange', 'weather': 'Weather'}
        for ds in DS:
            num('RelInfuseDs' + dsm[ds], float(np.mean([rel(M[k]['fuse'][0], M[k]['frozen'][0]) for k in M if k[1] == ds])))
        for H, w in (('96', 'NinetySix'), ('720', 'SevenTwenty')):
            num('RelInfuseH' + w, float(np.mean([rel(M[k]['fuse'][0], M[k]['frozen'][0]) for k in M if k[2] == H])))
        for t in TTA:
            d = [rel(M[k][t + '-noadj'][0], M[k][t][0]) for k in M if t + '-noadj' in M[k]]
            if d:
                num('Noadj' + t, float(np.mean(d)), '%.2f')
                num('NoadjMaxAbs' + t, float(np.max(np.abs(d))), '%.2f')
                print('%s noadj vs official: mean %+.2f%%, max|.| %.2f%% (n=%d)' % (t, np.mean(d), np.max(np.abs(d)), len(d)))
                for bb in BB:
                    db = [rel(M[k][t + '-noadj'][0], M[k][t][0]) for k in M if k[0] == bb and t + '-noadj' in M[k]]
                    num('Noadj%s%s' % (t, bb), float(np.mean(db)), '%.1f')
                    num('NoadjMax%s%s' % (t, bb), float(np.max(db)), '%.1f')
        import make_topconf_figs as TCF
        TCF.make_teaser(M)
        TCF.make_framework()
    table_azure(merged(['azure_functions']), EVF)
    if M and EVF:
        table_family(M, EVF)
    table_naive()
    fig_autoscale()
    if EVF:
        table_fm_full(EVF)
        big = {k: v for k, v in EVF.items() if k.endswith('electricity_96') or k.endswith('traffic_96')}
        for k, v in sorted(big.items()):
            tag = ('ECL' if 'electricity' in k else 'Traffic') + ('B' if 'bolt-base' in k else '')
            for m, nm in (('refresh', 'Refresh'), ('update', 'Update'), ('fuse', 'Infuse')):
                num('Big%s%s' % (tag, nm), rel(v[m][0], v['frozen'][0]))
            num('Big%sSec' % tag, float(v['seconds']), '%.0f')
        EVF = {k: v for k, v in EVF.items() if any(k.startswith('%s_%s_' % (fm, ds)) for fm in FMS for ds in DS)}
        agg, w = table_fm(EVF)
        for fm, tag in (('bolt-small', 'S'), ('bolt-base', 'B')):
            for m, nm in (('refresh', 'Refresh'), ('update', 'Update'), ('fuse', 'Infuse')):
                if agg.get((fm, m)):
                    num('FM%s%s' % (tag, nm), float(np.mean(agg[(fm, m)])))
        num('FMWins', int(sum(v for (fm, m), v in w.items() if m == 'fuse')))
        num('FMN', len(EVF))
        print('FM rel:', {k: round(float(np.mean(v)), 2) for k, v in agg.items()}, 'wins:', dict(w))
    if M or EVF:
        ab = table_abl(M, EVF)
        for part, d in ab.items():
            print('ablation', part, {k: round(float(v), 2) for k, v in d.items()})
        mac = {'full': 'Full', 'single_u': 'SingleU', 'no_shrink': 'NoShrink', 'shrink_0.3': 'FixedShrink',
               'forget_0.99': 'Forget', 'per_channel': 'PerChannel', 'refresh': 'Refresh', 'fuse': 'Fuse',
               'single_f': 'SingleF', 'fuse_eq': 'FuseEq', 'fuse_bg': 'FuseBG', 'rule_last': 'RuleLast',
               'rule_mean': 'RuleMean', 'rule_ar1': 'RuleAR'}
        for k, v in ab['all'].items():
            num('AblAll' + mac[k], float(v), '%.1f')
        for k, v in ab['h96'].items():
            num('AblHn' + mac[k], float(v), '%.1f')
        r2 = fig_fresh(M, EVF)
        num('RTwo', r2, '%.2f')
        print('refresh-gain R2: %.3f' % r2)
        facs = []
        allev = {**{'%s_%s_%s' % kk: M[kk]['ev'] for kk in M}, **EVF}
        MA_ = merged(['azure_functions'])
        allev_b = {**allev, **{'%s_%s_%s' % kk: MA_[kk]['ev'] for kk in MA_},
                   **{k: v for k, v in load_eval('res_fm.jsonl').items() if 'azure' in k}}
        for k, v in fig_budget(M, EVF).items():
            facs.append(budget_factor(allev_b[k]['service']))
            print('budget', k, {r: tuple(round(x, 4) for x in xy) for r, xy in v.items()}, 'factor', facs[-1])
        if facs:
            num('BudgetMin', int(min(facs))); num('BudgetMax', int(max(facs)))
        # serving-cost summary over every H=96 setting with service results (trained backbones + foundation models,
        # benchmark datasets and Azure Functions)
        MA = merged(['azure_functions'])
        allev_s = {**allev, **{'%s_%s_%s' % kk: MA[kk]['ev'] for kk in MA},
                   **{k: v for k, v in load_eval('res_fm.jsonl').items() if 'azure' in k}}
        for r in (12, 24):
            inc_h, inc_f = [], []
            for k, ev in allev_s.items():
                sv = ev.get('service')
                if not k.endswith('_96') or not sv or str(r) not in sv:
                    continue
                base = sv['1'][0]
                inc_h.append(rel(sv[str(r)][0], base))
                inc_f.append(rel(sv[str(r)][3] if len(sv[str(r)]) > 3 else sv[str(r)][1], base))
            if inc_h:
                word = {12: 'Twelve', 24: 'TwentyFour'}[r]
                num('BudgetHold' + word, float(np.mean(inc_h)))
                num('BudgetInfuse' + word, float(np.mean(inc_f)))
                num('BudgetN', len(inc_h))
                print('serving r=%d over %d settings: hold %+.1f%%, INFUSE %+.1f%%' % (r, len(inc_h), np.mean(inc_h),
                                                                                       np.mean(inc_f)))
    fig_analysis2(M, EVF)
    explained_share()
    fig_azure_time()
    sd = table_seeds()
    if sd:
        print('seeds:', {k: tuple(round(x, 2) for x in v) for k, v in sd.items()})
    table_cost()
    write_numbers()


def seed_results():
    """MSE per (backbone, dataset, H) -> seed -> method for every setting trained with several seeds."""
    import glob as _glob
    R = defaultdict(lambda: defaultdict(dict))
    files = ['grid_itdl.csv', 'grid_patch.csv', 'grid_patch_small.csv', 'grid_patch_long.csv', 'grid_patch192.csv',
             'grid_seeds.csv', 'grid_seeds_2.csv'] + sorted(os.path.basename(p) for p in _glob.glob(GRID + 'grid_dad*.csv')) + \
            sorted(os.path.basename(p) for p in _glob.glob(GRID + 'grid_seeds_dad*.csv'))
    for fn in files:
        if os.path.exists(GRID + fn):
            for r in csv.reader(open(GRID + fn)):
                if len(r) >= 7 and r[5] and r[1] in DS:
                    R[(r[0], r[1], r[2])][r[3]][r[4]] = float(r[5])
    EV = load_eval('res_grid.jsonl')
    S = {}
    for (bb, ds, H), by_seed in R.items():
        rows = {}
        for seed, row in by_seed.items():
            ev = EV.get('%s_%s_%s' % (bb, ds, H) + ('' if seed == '0' else '_s%s' % seed))
            if ev is None or not all(t in row for t in ADP):
                continue
            rows[seed] = dict({t: row[t] for t in ADP}, frozen=ev['frozen'][0],
                              **{m: ev[m][0] for m in ('refresh', 'update', 'fuse')})
        if len(rows) == 3:
            S[(bb, ds, H)] = rows
    return S


def table_seeds():
    """Three seeds (iTransformer and DLinear on all settings, PatchTST at H=96): mean +- std over seeds of the mean
    relative change (main appendix table) and per-setting MSE mean +- std (full table)."""
    S = seed_results()
    if not S:
        return None
    meths = ['frozen'] + ADP + ['refresh', 'update', 'fuse']
    mn = {'frozen': 'Frozen', 'TAFAS': 'TAFAS', 'PETSA': 'PETSA', 'COSA': 'COSA', 'DADAPTER': r'$\delta$-Adapter',
          'refresh': 'Refresh', 'update': r'\InfuseU{}', 'fuse': r'\Infuse{}'}
    groups = [('itdl', lambda k: k[0] in ('iTransformer', 'DLinear')), ('patch', lambda k: k[0] == 'PatchTST')]
    out, cols = {}, []
    for g, f in groups:
        ks = [k for k in S if f(k)]
        if not ks:
            continue
        cols.append((g, len(ks)))
        for m in meths[1:]:
            v = [np.mean([rel(S[k][s][m], S[k][s]['frozen']) for k in ks]) for s in ('0', '1', '2')]
            out[(g, m)] = (np.mean(v), np.std(v))
        # in how many settings INFUSE beats the best adapter for every seed
        num('SeedBetterAll' + g.capitalize(), int(sum(all(S[k][s]['fuse'] < min(S[k][s][t] for t in ADP)
                                                          for s in ('0', '1', '2')) for k in ks)))
        num('SeedN' + g.capitalize(), len(ks))
    head = {'itdl': r'\makecell{iTrans., DLinear\\ all $H$}', 'patch': r'\makecell{PatchTST\\ $H=96$}'}
    lines = [r'\begin{table}[H]', r'\centering',
             r'\caption{Three seeds: mean relative MSE change with respect to the frozen forecaster, mean $\pm$ '
             r'standard deviation over seeds (number of settings in parentheses).}',
             r'\label{tab:seeds}', r'\small',
             r'\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}l %s}' % ('c' * len(cols)), r'\toprule',
             r'Method & ' + ' & '.join('%s (%d)' % (head[g], n) for g, n in cols) + r' \\', r'\midrule']
    for m in meths[1:]:
        lines.append(r'%s & %s \\' % (mn[m], ' & '.join(r'$%+.1f \pm %.1f$' % out[(g, m)] for g, _ in cols)))
    lines += [r'\bottomrule', r'\end{tabular*}', r'\end{table}']
    open(PAPER + 'tab_seeds.tex', 'w').write('\n'.join(lines) + '\n')
    num('NSeeds', 3)
    num('SeedStdMax', float(max(v[1] for v in out.values())), '%.1f')
    # per dataset: MSE averaged over the horizons, mean +- std over seeds, for the best adapter and INFUSE
    blocks = [('iTransformer', HS), ('DLinear', HS), ('PatchTST', ['96'])]
    blocks = [(bb, hs) for bb, hs in blocks if any((bb, ds, H) in S for ds in DS for H in hs)]
    fl = [r'\begin{table}[H]', r'\centering',
          r'\caption{MSE over three seeds (mean$_{\pm\text{std}}$), averaged over the horizons (PatchTST: $H=96$), '
          r'for the best adapter per setting (B.A.) and \Infuse{}, with the better in \textbf{bold}.}',
          r'\label{tab:seeds_full}', r'\scriptsize', r'\setlength{\tabcolsep}{1.5pt}',
          r'\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}l %s}' % ' '.join(['cc'] * len(blocks)), r'\toprule',
          r'& ' + ' & '.join(r'\multicolumn{2}{c}{%s}' % {'iTransformer': 'iTrans.'}.get(bb, bb) for bb, hs in blocks)
          + r' \\',
          ''.join(r'\cmidrule(lr){%d-%d}' % (2 + 2 * i, 3 + 2 * i) for i in range(len(blocks))),
          r'Data & ' + ' & '.join([r'B.A. & \Infuse{}'] * len(blocks)) + r' \\', r'\midrule']
    for ds in DS:
        cells = []
        for bb, hs in blocks:
            ks = [(bb, ds, H) for H in hs if (bb, ds, H) in S]
            if len(ks) != len(hs):
                cells += ['--', '--']
                continue
            best_t = {k: min(ADP, key=lambda t: np.mean([S[k][s][t] for s in ('0', '1', '2')])) for k in ks}
            ba = [np.mean([S[k][s][best_t[k]] for k in ks]) for s in ('0', '1', '2')]
            inf = [np.mean([S[k][s]['fuse'] for k in ks]) for s in ('0', '1', '2')]
            b_inf = np.mean(inf) < np.mean(ba)
            cells += [(r'$\mathbf{%s}_{\pm%s}$' if not b_inf else r'$%s_{\pm%s}$') % (fmt(np.mean(ba)), fmt(np.std(ba))),
                      (r'$\mathbf{%s}_{\pm%s}$' if b_inf else r'$%s_{\pm%s}$') % (fmt(np.mean(inf)), fmt(np.std(inf)))]
        fl.append('%s & %s \\\\' % (DSN[ds], ' & '.join(cells)))
    fl += [r'\bottomrule', r'\end{tabular*}', r'\end{table}']
    open(PAPER + 'tab_seeds_full.tex', 'w').write('\n'.join(fl) + '\n')
    return {k: v for k, v in out.items()}


def table_cost():
    """Per-batch time (timing.json from time_all.py); placeholders if not measured yet."""
    if not os.path.exists('timing.json'):
        for k in ('TimeInfuseU', 'TimeInfuse', 'TimeTAFAS', 'TimeCOSA', 'TimePETSA'):
            NUM[k] = '--'
        return
    T = json.load(open('timing.json'))
    rows = T['rows']          # list of {setting, method, ms}
    settings = list(dict.fromkeys(r['setting'] for r in rows))
    meths = ['TAFAS', 'PETSA', 'COSA', 'INFUSE-U', 'INFUSE']
    mname = {'TAFAS': 'TAFAS', 'PETSA': 'PETSA', 'COSA': 'COSA', 'INFUSE-U': r'\InfuseU{}', 'INFUSE': r'\Infuse{}'}
    val = {(r['setting'], r['method']): r['ms'] for r in rows}
    # INFUSE: direct per-batch measurement of its own operations (time_infuse2.py -> timing_h.json)
    TH = json.load(open('timing_h.json')) if os.path.exists('timing_h.json') else {}
    smap = {'iTrans. ETTh1': 'iTransformer_ETTh1_96', 'iTrans. Weather': 'iTransformer_weather_96',
            'PatchTST ETTm1': 'PatchTST_ETTm1_96'}
    for s in settings:
        t = TH.get(smap.get(s, ''))
        if t:
            val[(s, 'INFUSE-U')], val[(s, 'INFUSE')] = t['update_ms'], t['fuse_ms']
    lines = [r'\begin{table}[H]', r'\centering',
             r'\caption{Per-batch time under the TTA protocol (ms, mean over all batches of the test stream, $H=96$). '
             r'TAFAS, PETSA, and COSA run on the GPU, \InfuseU{} and \Infuse{} on one CPU thread. Bottom: \Infuse{} '
             r'versus the horizon (iTransformer).}',
             r'\label{tab:cost}', r'\small', r'\setlength{\tabcolsep}{3pt}',
             r'\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}l %s}' % ('r' * len(settings)), r'\toprule',
             r'Method & ' + ' & '.join(s.replace('_', ' ') for s in settings) + r' \\', r'\midrule']
    for m in meths:
        lines.append('%s & %s \\\\' % (mname[m], ' & '.join('%.1f' % val[(s, m)] if (s, m) in val else '--'
                                                            for s in settings)))
    hs = [h for h in HS if 'iTransformer_ETTh1_%s' % h in TH]
    if hs:
        lines += [r'\midrule', r'\multicolumn{%d}{l}{\Infuse{} (ms/batch), ETTh1 at $H=$ %s, and Weather at $H=720$} \\'
                  % (len(settings) + 1, ', '.join(hs))]
        cells = ['%.1f' % TH['iTransformer_ETTh1_%s' % h]['fuse_ms'] for h in hs]
        if 'iTransformer_weather_720' in TH:
            cells.append('%.1f' % TH['iTransformer_weather_720']['fuse_ms'])
        lines.append(r'\multicolumn{%d}{l}{\quad %s} \\' % (len(settings) + 1, ' / '.join(cells)))
        num('TimeInfuseLong', TH['iTransformer_ETTh1_720']['fuse_ms'] if 'iTransformer_ETTh1_720' in TH else 0.0)
    lines += [r'\bottomrule', r'\end{tabular*}', r'\end{table}']
    open(PAPER + 'tab_cost.tex', 'w').write('\n'.join(lines) + '\n')
    for m, k in (('INFUSE-U', 'TimeInfuseU'), ('INFUSE', 'TimeInfuse'), ('TAFAS', 'TimeTAFAS'), ('COSA', 'TimeCOSA'),
                 ('PETSA', 'TimePETSA')):
        vs = [val[(s, m)] for s in settings if (s, m) in val]
        NUM[k] = ('%.1f--%.1f' % (min(vs), max(vs))) if len(vs) > 1 else ('%.1f' % vs[0] if vs else '--')


if __name__ == '__main__':
    main()
