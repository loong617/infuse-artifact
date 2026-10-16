import numpy as np

X = np.load('apps_all14.npy')[:, :50]
T = len(X)
tr = int(T * 0.7)
for name, Y in (('raw', X), ('log1p', np.log1p(X))):
    mu, sd = Y[:tr].mean(0), Y[:tr].std(0)
    sd[sd == 0] = 1
    Z = (Y - mu) / sd
    te = Z[T - int(T * 0.2):]
    mx = np.abs(te).max(0)
    print('%-6s test |z| max: median %.1f  90%% %.1f  max %.1f   test var mean %.2f  max %.2f   const-in-train %d' % (
        name, np.median(mx), np.percentile(mx, 90), mx.max(), te.var(0).mean(), te.var(0).max(), int((Y[:tr].std(0) == 0).sum())))
    print('       channels with test var > 4:', np.where(te.var(0) > 4)[0].tolist())
# periodicity of the aggregate (daily cycle visible?) and correlation of channels
agg = X.sum(1)
print('daily profile corr between consecutive days (aggregate):',
      np.round([np.corrcoef(agg[d * 1440:(d + 1) * 1440], agg[(d + 1) * 1440:(d + 2) * 1440])[0, 1] for d in range(13)], 2))
