"""나우캐스트: 기준시각 h0까지의 실측(실시간 주차 API로 얻을 수 있는 정보)으로 h0+1..h0+3 재차대수 예측.
비교: 프로파일(사전 평균), 지속(현재값 유지 ≈ 기관 서비스의 고정 표시), GBM 나우캐스트. LODO 31-fold."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from train import OCC, ARR, DEP, X, D, H, dtype, ev_day, days, CAP

K = (1, 2, 3)
ORIG = range(8, 21)

def rows(d, P):
    out = []
    for h0 in ORIG:
        for k in K:
            t = h0 + k
            if t >= H: continue
            base = P[dtype[d]]
            f = [k, h0, t, OCC[d, h0], OCC[d, h0] - OCC[d, h0 - 1], OCC[d, h0] - OCC[d, h0 - 2],
                 ARR[d, h0], DEP[d, h0], base[h0], base[t], OCC[d, h0] - base[h0], *X[d, t, 1:]]
            out.append((f, OCC[d, t], h0, t, k))
    return out

def run():
    res = {m: [] for m in ('profile', 'persist', 'gbm', 'gbm_q80')}
    meta = []
    for d in range(D):
        tr = [i for i in range(D) if i != d]
        P = {t: OCC[[i for i in tr if dtype[i] == t]].mean(0) for t in (0, 2)}
        Rtr = [r for i in tr for r in rows(i, P)]
        m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=15,
                                          min_samples_leaf=15, l2_regularization=1.0, random_state=0)
        Xt = np.array([r[0] for r in Rtr]); yt = np.array([r[1] for r in Rtr])
        m.fit(Xt, yt - Xt[:, 3])  # 현재값 대비 변화량 학습
        mq = HistGradientBoostingRegressor(loss='quantile', quantile=0.8, max_iter=400, learning_rate=0.05,
                                           max_leaf_nodes=15, min_samples_leaf=15, l2_regularization=1.0, random_state=0)
        mq.fit(Xt, yt - Xt[:, 3])
        Rte = rows(d, P)
        Xe = np.array([r[0] for r in Rte])
        pg = Xe[:, 3] + m.predict(Xe)
        pq = Xe[:, 3] + mq.predict(Xe)
        for (f, y, h0, t, k), g, q in zip(Rte, pg, pq):
            res['gbm_q80'].append(max(q, 0))
            res['profile'].append(P[dtype[d]][t]); res['persist'].append(f[3]); res['gbm'].append(max(g, 0))
            meta.append((d, h0, t, k, y))
    return {k: np.array(v) for k, v in res.items()}, np.array(meta)

if __name__ == '__main__':
    res, meta = run()
    y = meta[:, 4]; d = meta[:, 0].astype(int); t = meta[:, 2]; k = meta[:, 3]
    evm = ev_day[d] & (t >= 15) & (t <= 22)
    full_true = y >= 0.9 * CAP
    print(f'n={len(y)}  event-evening n={evm.sum()}  만차(>=90%) 실측 n={full_true.sum()}')
    for name, p in res.items():
        e = np.abs(p - y)
        line = f'{name:8s} MAE all {e.mean():6.1f} | event-evening {e[evm].mean():6.1f} | '
        line += ' '.join(f'k{kk}:{e[(k == kk) & evm].mean():6.1f}' for kk in K)
        tp = ((p >= 0.9 * CAP) & full_true).sum(); fp = ((p >= 0.9 * CAP) & ~full_true).sum()
        line += f' | 만차 탐지 {tp}/{full_true.sum()} 오탐 {fp}'
        line += f' | MAPE(ev) {100 * np.mean(e[evm] / np.maximum(y[evm], 1)):5.1f}%'
        print(line)
    np.savez(__import__('pathlib').Path(__file__).parent / 'nowcast_eval.npz', meta=meta, **res)
