"""올팍 나우 — 시간대별 재차대수·출차량 예측 (2024-07, 올림픽공원 동2·남2·북2문).
검증: Leave-One-Day-Out (31 fold). 베이스라인: 요일유형×시간 평균 프로파일."""
import json, datetime as dt, numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

R = Path(__file__).resolve().parents[1] / 'data/raw'
CAP = 2200

def load_pipe(name, n_parts):
    rows = [l.strip().split('|') for l in open(R / name) if l[0].isdigit()]
    return [np.array([list(map(float, r[i].split())) for r in rows]) for i in range(1, n_parts + 1)]

ARR, DEP, OCC = load_pipe('inout_2024_07_hourly.txt', 3)
TEMP, RAIN = load_pipe('weather_2024_07_hourly.txt', 2)
EV = json.load(open(R / 'events.json'))
SEATS = {k: v['seats'] for k, v in EV['venues'].items()}
D, H = OCC.shape
days = [dt.date(2024, 7, i + 1) for i in range(D)]

def event_feats(sessions, date, h):
    """공연 회차별 좌석 가중 커널: 도착(시작 3h 전~시작), 관람 중(시작~+3h), 종료 직후(+2~+4h)."""
    pre = dur = post = dome = 0.0
    for d_, v, s, _ in sessions:
        if d_ != date:
            continue
        k = SEATS[v] / 1000
        if s - 3 <= h < s: pre += k * (h - s + 4) / 3
        if s <= h <= s + 3: dur += k
        if s + 2 <= h <= s + 4: post += k
        if v == 'KSPO DOME' and s - 3 <= h <= s + 3: dome += 1
    return pre, dur, post, dome

def features(date, h, temp, rain, sessions):
    wd = date.weekday()
    pre, dur, post, dome = event_feats(sessions, date.isoformat(), h)
    return [h, wd >= 5, wd == 4, np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
            temp, rain, pre, dur, post, dome]

S24 = EV['sessions_2024_07']
X = np.array([[features(days[d], h, TEMP[d, h], RAIN[d, h], S24) for h in range(H)] for d in range(D)], float)
dtype = np.array([2 if days[d].weekday() >= 5 else 0 for d in range(D)])

def profile(train_idx, Y):
    P = {}
    for t in (0, 2):
        idx = [i for i in train_idx if dtype[i] == t]
        P[t] = Y[idx].mean(0)
    return P

def lodo(Y, kind):
    pred = np.zeros_like(Y)
    for d in range(D):
        tr = [i for i in range(D) if i != d]
        P = profile(tr, Y)
        base_tr = np.array([P[dtype[i]] for i in tr]).reshape(-1)
        Xtr = X[tr].reshape(-1, X.shape[2]); ytr = Y[tr].reshape(-1)
        if kind == 'profile':
            pred[d] = P[dtype[d]]
        elif kind == 'gbm':
            m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                              min_samples_leaf=10, l2_regularization=1.0, random_state=0)
            m.fit(Xtr, ytr); pred[d] = m.predict(X[d])
        elif kind == 'hybrid':  # 프로파일 + 공연·날씨 잔차를 리지로 보정
            ev = lambda A: A[:, [5, 6, 7, 8, 9, 10]]
            m = Ridge(alpha=1.0).fit(ev(Xtr), ytr - base_tr)
            pred[d] = P[dtype[d]] + m.predict(ev(X[d]))
    return np.clip(pred, 0, None)

ev_day = np.array([any(s[0] == days[d].isoformat() for s in S24) for d in range(D)])
mask_ev = np.zeros((D, H), bool); mask_ev[ev_day, 15:23] = True

def report(Y, name):
    out = {}
    for k in ('profile', 'gbm', 'hybrid'):
        p = lodo(Y, k)
        e = np.abs(p - Y)
        out[k] = dict(MAE=round(e.mean(), 1), MAE_event_evening=round(e[mask_ev].mean(), 1),
                      MAE_nonevent=round(e[~mask_ev].mean(), 1),
                      peak_err_event_days=[round(float(p[d].max() - Y[d].max())) for d in np.where(ev_day)[0]])
        out[k]['_pred'] = p
    print(f'== {name}')
    for k, v in out.items():
        print(k, {a: b for a, b in v.items() if a != '_pred'})
    return out

if __name__ == '__main__':
    print('event days', [days[d].day for d in np.where(ev_day)[0]])
    r_occ = report(OCC, 'occupancy (cars at hh:30)')
    r_dep = report(DEP, 'departures per hour')
    np.save(Path(__file__).parent / 'pred_occ.npy', np.stack([r_occ[k]['_pred'] for k in ('profile', 'gbm', 'hybrid')]))
    np.save(Path(__file__).parent / 'pred_dep.npy', np.stack([r_dep[k]['_pred'] for k in ('profile', 'gbm', 'hybrid')]))
