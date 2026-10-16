"""Compare fusion-weight parametrizations under the TAFAS protocol (causal estimates from revealed history)."""
import sys

import numpy as np

import inflight as I


def run(path):
    y, pred, gt, E, L, H = I.load(path)
    n, _, C = pred.shape
    B = I.tafas_batches(y, n, L)
    P = max(p for _, _, p in B)
    K = min(P, H - 1) + 1
    cov = I.ErrorCov(H, C)
    # stats indexed [k, l] (l = refresh lead = j - k): a=E u^2, b=E v^2, c=E uv, cnt
    S = np.zeros((4, K, H))
    pend, nxt = [], 0
    variants = ['refresh', 'update', 'w_j', 'w_l', 'w_kl', 'w_kl_shrunk', 'half', 'w_j_rho90', 'w_j_rho97',
                'w_j_sh50']
    se = {v: 0.0 for v in variants}
    Sj = np.zeros((3, H))
    Sr = {0.9: np.zeros((3, H)), 0.97: np.zeros((3, H))}
    for i0, b_end, p in B:
        T = i0 + p
        while nxt < n and nxt + H <= T:
            cov.add(E[nxt]); nxt += 1
        keep = []
        for rr in Sr:
            Sr[rr] *= rr
        for (w, k, eu, er) in pend:
            if w + H <= T:
                for rr in Sr:
                    Sr[rr][0, k:] += (eu ** 2).mean(1); Sr[rr][1, k:] += (er ** 2).mean(1)
                    Sr[rr][2, k:] += (eu * er).mean(1)
                S[0, k, :H - k] += (eu ** 2).mean(1); S[1, k, :H - k] += (er ** 2).mean(1)
                S[2, k, :H - k] += (eu * er).mean(1); S[3, k, :H - k] += 1
                Sj[0, k:] += (eu ** 2).mean(1); Sj[1, k:] += (er ** 2).mean(1); Sj[2, k:] += (eu * er).mean(1)
            else:
                keep.append((w, k, eu, er))
        pend = keep
        Sig = cov.sigma() if cov.ready(2 * H) else None
        al, bl, cl = S[0].sum(0), S[1].sum(0), S[2].sum(0)          # pooled over k, per l
        for i in range(i0, b_end):
            k = min(T - i, H)
            e = E[i]
            if not (0 < k < H and i + k < n):
                for v in variants:
                    se[v] += float((e ** 2).sum())
                continue
            er = gt[i, k:] - pred[i + k, :H - k]
            eu = e[k:] - cov.gain(k, Sig) @ e[:k] if Sig is not None else e[k:].copy()
            base = float((e[:k] ** 2).sum())

            def wt(a, b, c):
                den = a + b - 2 * c
                return np.clip(np.where(den > 1e-12, (a - c) / np.maximum(den, 1e-12), 0.5), 0, 1)

            ws = {
                'refresh': np.ones(H - k), 'update': np.zeros(H - k), 'half': np.full(H - k, 0.5),
                'w_j': wt(Sj[0, k:], Sj[1, k:], Sj[2, k:]),
                'w_l': wt(al[:H - k], bl[:H - k], cl[:H - k]),
                'w_kl': wt(S[0, k, :H - k], S[1, k, :H - k], S[2, k, :H - k]),
            }
            lam = 10.0                     # pseudo-batches of the pooled-over-k statistics
            nk = S[3, k, :H - k]
            Nl = np.maximum(S[3].sum(0)[:H - k], 1)
            ws['w_j_rho90'] = wt(Sr[0.9][0, k:], Sr[0.9][1, k:], Sr[0.9][2, k:])
            ws['w_j_rho97'] = wt(Sr[0.97][0, k:], Sr[0.97][1, k:], Sr[0.97][2, k:])
            ws['w_j_sh50'] = 0.5 * ws['w_j'] + 0.25
            ws['w_kl_shrunk'] = wt(S[0, k, :H - k] + lam * al[:H - k] / Nl, S[1, k, :H - k] + lam * bl[:H - k] / Nl,
                                   S[2, k, :H - k] + lam * cl[:H - k] / Nl)
            for v in variants:
                w = ws[v][:, None]
                se[v] += base + float((((1 - w) * eu + w * er) ** 2).sum())
            pend.append((i, k, eu, er))
    fr = float((E ** 2).mean())
    print('== %s frozen %.4f' % (path, fr))
    print('   ' + '  '.join('%s %+.2f%%' % (v, 100 * (se[v] / E.size - fr) / fr) for v in variants))


if __name__ == '__main__':
    for p_ in sys.argv[1:]:
        run(p_)
