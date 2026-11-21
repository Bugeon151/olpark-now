"""사전(전날) 예측 개선: 공연장별 '시작 시각 기준 경과 시간'을 변수로 쓰는 분위 회귀(10·50·90%).
기존 시나리오 방식(평소 곡선 + KSPO DOME 초과 분포 × 좌석 비율)과 같은 홀드아웃(1~7월 학습 → 10~12월 공연일)으로 비교."""
import json, numpy as np, model2025 as M
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
CAP = M.CAP
VEN = ['KSPO DOME', '핸드볼경기장', '올림픽홀', '우리금융아트홀', '야외']
V = [d for d in range(M.D) if M.VALID[d]]
GB = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=1.0, random_state=0)

def feats(date, sessions, h):
    import datetime as dt
    d = dt.date.fromisoformat(date) if isinstance(date, str) else date
    dtype = 2 if (d.weekday() >= 5 or d.isoformat() in M.HOL) else (1 if d.weekday() == 4 else 0)
    f = [h, dtype, np.sin(2 * np.pi * d.month / 12), np.cos(2 * np.pi * d.month / 12)]
    for v in VEN:
        ss = [s for vv, s in sessions if vv == v]
        f.append(h - min(ss) if ss else -99)  # 첫 회차 시작 기준 경과 시간(없으면 -99)
        f.append(len(ss))
    f.append(sum(M.SEATS.get(v, 0) for v, _ in sessions if v != '야외') / 1000)
    return f

def X_of(days):
    X, y = [], []
    for d in days:
        ses = M.by_day.get(M.days[d].isoformat(), [])
        for h in range(24): X.append(feats(M.days[d], ses, h)); y.append(M.OCC[d, h])
    return np.array(X), np.array(y)

def fit(days):
    X, y = X_of(days)
    return {q: HistGradientBoostingRegressor(loss='quantile', quantile=q / 100, **GB).fit(X, y) for q in (10, 50, 90)}

def predict(models, date, sessions):
    X = np.array([feats(date, sessions, h) for h in range(24)])
    P = {q: np.maximum(m.predict(X), 0) for q, m in models.items()}
    lo = np.minimum.reduce(list(P.values())); hi = np.maximum.reduce(list(P.values()))
    return lo, P[50], hi

first = lambda a, thr: next((h for h in range(6, 24) if a[h] >= thr), None)

def evaluate(fc, TE):
    cov, pk = [], []; res = {nm: {'actual_days': 0, 'warned_days': 0, 'hit': 0, 'false_alarm': 0} for nm in ('busy70', 'full90', 'over')}
    for d in TE:
        lo, mid, hi = fc(d); y = M.OCC[d]
        cov += [lo[h] <= y[h] <= hi[h] for h in range(9, 23)]; pk.append(abs(mid.max() - y.max()))
        for nm, thr in (('busy70', .7 * CAP), ('full90', .9 * CAP), ('over', CAP)):
            a, f = first(y, thr), first(hi, thr); r = res[nm]
            r['actual_days'] += a is not None; r['warned_days'] += f is not None; r['hit'] += (a is not None and f is not None); r['false_alarm'] += (a is None and f is not None)
    return {'band_coverage_9_22': round(float(np.mean(cov)), 3), 'peak_mae_mid': round(float(np.mean(pk)), 1), **res}

if __name__ == '__main__':
    import scen_eval as S
    TR = [d for d in V if M.month[d] <= 7]; TE = [d for d in V if M.month[d] >= 10 and M.ev_day[d]]
    mod = fit(TR)
    q = evaluate(lambda d: predict(mod, M.days[d], M.by_day.get(M.days[d].isoformat(), [])), TE)
    s = evaluate(S.forecast, TE)
    out = {'test_event_days': len(TE), 'quantile_gbm': q, 'scenario': s}
    print(json.dumps(out, ensure_ascii=False))
    json.dump(out, open(Path(__file__).parent / 'dayahead_eval.json', 'w'), ensure_ascii=False, indent=1)
