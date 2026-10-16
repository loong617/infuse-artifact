"""Aggregate Azure Functions 2019 per-minute invocation counts to applications and print statistics."""
import numpy as np
import pandas as pd

MIN = [str(i) for i in range(1, 1441)]
days = []
for d in range(1, 15):
    df = pd.read_csv('invocations_per_function_md.anon.d%02d.csv' % d)
    df['app'] = df['HashOwner'] + '/' + df['HashApp']
    g = df.groupby('app')[MIN].sum()
    trig = df.groupby('app')['Trigger'].agg(lambda s: '+'.join(sorted(set(s))))
    days.append((g, trig))
    print('day', d, 'functions', len(df), 'apps', len(g), flush=True)
apps = sorted(set.intersection(*[set(g.index) for g, _ in days]))
print('apps present on all 14 days:', len(apps))
X = np.concatenate([g.loc[apps].values for g, _ in days], axis=1).T.astype(np.float64)   # [20160, n_apps]
trig = days[0][1].loc[apps]
tot = X.sum(0)
order = np.argsort(-tot)
np.save('apps_all14.npy', X[:, order[:2000]])
pd.Series(np.array(apps)[order[:2000]]).to_csv('apps_all14_ids.csv', index=False)
trig.iloc[order[:2000]].to_csv('apps_all14_trig.csv')
for rank in list(range(0, 30)) + [49, 99, 199, 499, 999]:
    j = order[rank]
    x = X[:, j]
    print('%4d mean/min %10.1f  zero-frac %.3f  cv %.2f  lag1-diff-cv %.3f  trig %s' % (
        rank, x.mean(), (x == 0).mean(), x.std() / x.mean(), np.abs(np.diff(x)).mean() / x.mean(), trig.iloc[rank]))
