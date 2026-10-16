import re
import numpy as np
from collections import defaultdict

data = defaultdict(dict)
pat = re.compile(r'MS\|(\w+)\|(\w+)\|s(\d+)\|Test MSE: ([\d.]+)')
for line in open('multiseed.log', encoding='utf-8', errors='ignore'):
    m = pat.match(line)
    if m:
        data[(m.group(1), m.group(2))][int(m.group(3))] = float(m.group(4))

s2024 = {
    ('outaff', 'ETTh2'): .2852, ('outaff', 'ETTm2'): .1787, ('outaff', 'electricity'): .2481,
    ('ridge', 'ETTh2'): .2775, ('ridge', 'ETTm2'): .1751, ('ridge', 'electricity'): .2289,
    ('comb', 'ETTh2'): .2740, ('comb', 'ETTm2'): .1733, ('comb', 'electricity'): .2245,
    ('pool', 'ETTh2'): .2665, ('pool', 'ETTm2'): .1692, ('pool', 'electricity'): .2227,
    ('rpool', 'ETTh2'): .2668, ('rpool', 'ETTm2'): .1695, ('rpool', 'electricity'): .2225,
    ('eta5', 'ETTh2'): .2656, ('eta5', 'ETTm2'): .1693, ('eta5', 'electricity'): .2231,
    ('eta500', 'ETTh2'): .2718, ('eta500', 'ETTm2'): .1704, ('eta500', 'electricity'): .2222,
    ('hist480', 'ETTh2'): .2761, ('hist480', 'electricity'): .2238,
    ('hist2880', 'ETTh2'): .2723, ('hist2880', 'electricity'): .2230,
}
out = {}
for key in sorted(set(list(data.keys()) + list(s2024.keys()))):
    tag, ds = key
    vals = dict(data.get(key, {}))
    if s2024.get(key):
        vals[2024] = s2024[key]
    if len(vals) >= 5:
        v = np.array(list(vals.values()))
        out[key] = (v.mean(), v.std())
        print(f'{tag:8s} {ds:12s} n={len(v)} mean={v.mean():.4f} std={v.std():.4f}')
np.save('multiseed_agg.npy', {f'{t}|{d}': v for (t, d), v in out.items()}, allow_pickle=True)
print('saved', len(out), 'entries')
