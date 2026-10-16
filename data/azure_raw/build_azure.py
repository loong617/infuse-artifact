"""Build data/azure_functions/azure_functions.csv from the Azure Functions 2019 trace.

Source: https://github.com/Azure/AzurePublicDataset/blob/master/AzureFunctionsDataset2019.md
(invocations_per_function_md.anon.d01..d14.csv: invocations per function and minute, 14 days).

Per-minute invocation counts are summed over the functions of each application (HashOwner, HashApp), since
applications are the unit that is scaled. We keep the 50 applications with the most invocations among those
invoked on all 14 days, ordered by total invocations. Values are raw counts; the loader standardizes them
with training statistics like every other dataset. Timestamps start on a Monday, which matches the weekly
pattern of the trace (days 6, 7, 13, 14 are the low-activity days).
"""
import os

import numpy as np
import pandas as pd

N_APPS = 50
MIN = [str(i) for i in range(1, 1441)]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'azure_functions')

per_day = []
for d in range(1, 15):
    df = pd.read_csv('invocations_per_function_md.anon.d%02d.csv' % d)
    df['app'] = df['HashOwner'] + '/' + df['HashApp']
    per_day.append(df.groupby('app')[MIN].sum())
apps = sorted(set.intersection(*[set(g.index) for g in per_day]))
X = np.concatenate([g.loc[apps].values for g in per_day], axis=1).T.astype(np.float64)   # [20160, n_apps]
order = np.argsort(-X.sum(0), kind='stable')[:N_APPS]
X = X[:, order]
dates = pd.date_range('2019-07-15 00:00', periods=len(X), freq='min')
out = pd.DataFrame(X, columns=['app%02d' % i for i in range(N_APPS)])
out.insert(0, 'date', dates.strftime('%Y-%m-%d %H:%M:%S'))
os.makedirs(OUT, exist_ok=True)
out.to_csv(os.path.join(OUT, 'azure_functions.csv'), index=False)
pd.Series(np.array(apps)[order], name='owner/app').to_csv(os.path.join(OUT, 'apps.csv'), index_label='column')
print('apps invoked on all 14 days: %d; kept %d; steps %d; share of all invocations of those apps: %.1f%%' % (
    len(apps), N_APPS, len(X), 100 * X.sum() / np.concatenate([g.loc[apps].values for g in per_day], axis=1).sum()))
