"""검증 강화: ① 일 단위 부트스트랩 95% 신뢰구간(개선폭) ② 변수 제거 실험(ablation) ③ 추가 기준모델(전주 같은 요일·시각)."""
import json, numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
import nowcast as N
from train import OCC, D, ev_day, days

P = Path(__file__).parent
z = np.load(P / 'nowcast_eval.npz'); meta = z['meta']
d = meta[:, 0].astype(int); t = meta[:, 2].astype(int); y = meta[:, 4]
evm = ev_day[d] & (t >= 15) & (t <= 22)

# ③ 전주 같은 요일·시각(7월 1~7일은 이전 주 없음 → 제외 비교)
sn = np.array([OCC[di - 7, ti] if di >= 7 else np.nan for di, ti in zip(d, t)])
ok = ~np.isnan(sn)

def mae(p, m): return float(np.abs(p[m] - y[m]).mean())
res = {'seasonal_naive_evening_d>=8': round(mae(sn, evm & ok), 1), 'gbm_evening_d>=8': round(mae(z['gbm'], evm & ok), 1),
       'profile_evening_d>=8': round(mae(z['profile'], evm & ok), 1)}

# ① 부트스트랩: 날짜 재표본(공연일만), 개선폭 = 기준 MAE - 나우캐스트 MAE
rng = np.random.default_rng(0); evd = np.unique(d[evm]); B = 5000
def boot(base):
    out = []
    for _ in range(B):
        s = rng.choice(evd, len(evd), replace=True)
        m = np.concatenate([np.where(evm & (d == k))[0] for k in s])
        out.append(np.abs(z[base][m] - y[m]).mean() - np.abs(z['gbm'][m] - y[m]).mean())
    return [round(float(np.percentile(out, q)), 1) for q in (2.5, 50, 97.5)]
res['improve_vs_profile_CI95'] = boot('profile'); res['improve_vs_persist_CI95'] = boot('persist')

# ② ablation: 피처 그룹 제거 후 LODO 재평가(공연일 저녁 MAE)
names = ['k', 'h0', 't', 'occ', 'd1', 'd2', 'arr', 'dep', 'base_h0', 'base_t', 'dev', 'weekend', 'fri', 'sin', 'cos', 'temp', 'rain', 'pre', 'dur', 'post', 'dome']
groups = {'기상(기온·강수)': ['temp', 'rain'], '공연 일정': ['pre', 'dur', 'post', 'dome'], '실시간 추세(변화량·입출차)': ['d1', 'd2', 'arr', 'dep']}
def run(drop):
    keep = [i for i, n in enumerate(names) if n not in drop]
    errs = []
    for dd in range(D):
        tr = [i for i in range(D) if i != dd]
        Pp = {tt: OCC[[i for i in tr if N.dtype[i] == tt]].mean(0) for tt in (0, 2)}
        R = [r for i in tr for r in N.rows(i, Pp)]
        X = np.array([r[0] for r in R]); Y = np.array([r[1] for r in R])
        m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=15, l2_regularization=1.0, random_state=0)
        m.fit(X[:, keep], Y - X[:, 3])
        Re = N.rows(dd, Pp); Xe = np.array([r[0] for r in Re])
        p = np.maximum(Xe[:, 3] + m.predict(Xe[:, keep]), 0)
        for (f, yy, h0, tt, k), pp in zip(Re, p):
            if ev_day[dd] and 15 <= tt <= 22: errs.append(abs(pp - yy))
    return round(float(np.mean(errs)), 1)
res['ablation_event_evening_MAE'] = {'전체 변수': run([])} | {f'-{g}': run(v) for g, v in groups.items()}
print(json.dumps(res, ensure_ascii=False, indent=1))
json.dump(res, open(P / 'robust.json', 'w'), ensure_ascii=False, indent=1)
