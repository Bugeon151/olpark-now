"""2025년 12개월(입출차 원시기록 집계) 학습·검증.
입력: data/raw/y2025/hourly.csv(시간대별 입차·출차·재차), weather.csv, ../sessions_2025.txt
검증: ① 월 단위 교차검증(12 fold) ② 시간순 홀드아웃(1~7월 학습 → 10~12월 평가; 8·9월은 원천 표본 수록이라 제외)
비교: 평균 프로파일(월·요일유형·시간), 현재값 유지, 전주 같은 시각, 나우캐스트(GBM), 사전 예측(GBM)."""
import json, datetime as dt, numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor

R = Path(__file__).resolve().parents[1] / 'data/raw'
CAP = 2200
D0 = dt.date(2025, 1, 1)
HOL = {'2025-01-01', '2025-01-28', '2025-01-29', '2025-01-30', '2025-03-01', '2025-03-03', '2025-05-05', '2025-05-06', '2025-06-03', '2025-06-06',
       '2025-08-15', '2025-10-03', '2025-10-05', '2025-10-06', '2025-10-07', '2025-10-08', '2025-10-09', '2025-12-25'}
SEATS = {'KSPO DOME': 15000, '핸드볼경기장': 5003, '올림픽홀': 3152, '우리금융아트홀': 1184, '야외': 10000}

def venue(v):
    if 'KSPO' in v and '잔디' in v: return '야외'
    if 'KSPO' in v: return 'KSPO DOME'
    if '핸드볼' in v or '티켓링크' in v: return '핸드볼경기장'
    if '올림픽홀' in v: return '올림픽홀'
    if '우리금융' in v: return '우리금융아트홀'
    return '야외'  # 88잔디마당·올림픽공원 등 야외 행사(수용 인원 미공개 → 1만 가정, 별도 변수)

def load():
    import os
    H = np.loadtxt(R / os.environ.get('Y25_HOURLY', 'y2025/hourly.csv'), delimiter=',', skiprows=1)[:365 * 24]
    n = len(H) // 24 * 24  # 그룹별 열(arr0,dep0,occ0,...)이면 0번 그룹 = 동2·남2·북2문(수용 2,200대)
    arr, dep, occ = (H[:n, i].reshape(-1, 24) for i in (1, 2, 3))
    W = np.loadtxt(R / 'y2025/weather.csv', delimiter=',', skiprows=1)[:365 * 24]
    temp, rain = W[:n, 1].reshape(-1, 24), W[:n, 2].reshape(-1, 24)
    ses = []
    for s in open(R / 'sessions_2025.txt').read().strip().split('\n')[1].split(';'):
        d, v, h, src = s.split(',')[:4]
        ses.append((d, venue(v), int(h)))
    return arr, dep, occ, temp, rain, ses

ARR, DEP, OCC, TEMP, RAIN, SES = load()
D = OCC.shape[0]
days = [D0 + dt.timedelta(i) for i in range(D)]
dtype = np.array([2 if (d.weekday() >= 5 or d.isoformat() in HOL) else (1 if d.weekday() == 4 else 0) for d in days])
month = np.array([d.month for d in days])
by_day = {}
for d, v, h in SES: by_day.setdefault(d, []).append((v, h))
ev_day = np.array([d.isoformat() in by_day for d in days])
# 원천 데이터 수록 범위(2025 입출차 2,411,225건 전수 집계로 확인): 매월 말일 0건, 8·9월은 일평균 약 2.2천 건(다른 달 약 9.4천 건의 23%),
# 11·12월은 1~7일만 수록(11/7은 515건). → 수록이 완전한 날만 학습·평가에 사용한다.
_last = np.array([(d + dt.timedelta(1)).month != d.month for d in days])
VALID = ~_last & ~np.isin(month, (8, 9)) & ~((month >= 11) & (np.array([d.day for d in days]) >= 7))

def ev_feats(di, h):
    pre = dur = post = dome = outdoor = 0.0
    for v, s in by_day.get(days[di].isoformat(), []):
        k = SEATS[v] / 1000
        if v == '야외':
            if s - 2 <= h <= s + 6: outdoor = 1
            continue
        if s - 3 <= h < s: pre += k * (h - s + 4) / 3
        if s <= h <= s + 3: dur += k
        if s + 2 <= h <= s + 4: post += k
        if v == 'KSPO DOME' and s - 3 <= h <= s + 3: dome = 1
    return [pre, dur, post, dome, outdoor]

import pandas as pd
_L = pd.read_csv(R / 'y2025/linked_2025.csv'); _L['dt'] = pd.to_datetime(_L['dt'])
_E = _L[_L['공연명'].notna()]
ATT = _E.groupby([_E.dt.dt.date, '공연명'])['공연관람객수'].max().groupby(level=0).sum()  # 일자별 공연 관람객 합(연계 데이터)
att_day = np.array([float(ATT.get(d, 0.0)) / 1000 for d in days])
def att_kernel(di, h):  # 관람객 규모 × 주 공연 시작 시각 주변 구간
    if att_day[di] == 0: return [0.0, 0.0]
    ss = [s for v, s in by_day.get(days[di].isoformat(), []) if v != '야외'] or [18]
    s = min(ss)
    return [att_day[di] * (1.0 if s - 3 <= h <= s + 3 else 0.0), att_day[di]]
EVF = np.array([[ev_feats(d, h) + att_kernel(d, h) for h in range(24)] for d in range(D)])  # D,24,7

def profile(tr):
    P = np.zeros((13, 3, 24))
    for t in range(3):
        for m in range(1, 13):
            idx = [i for i in tr if dtype[i] == t and month[i] == m and not ev_day[i]]
            if len(idx) < 2: idx = [i for i in tr if dtype[i] == t and not ev_day[i]]
            if len(idx) < 2: idx = [i for i in tr if dtype[i] == t]
            P[m, t] = OCC[idx].mean(0)
    return P

ORIG, K = range(8, 21), (1, 2, 3)
def nrows(d, P):
    base = P[month[d], dtype[d]]; out = []
    for h0 in ORIG:
        for k in K:
            t = h0 + k
            f = [k, h0, t, OCC[d, h0], OCC[d, h0] - OCC[d, h0 - 1], OCC[d, h0] - OCC[d, h0 - 2], ARR[d, h0], DEP[d, h0],
                 base[h0], base[t], OCC[d, h0] - base[h0], dtype[d], month[d], TEMP[d, t], RAIN[d, t], *EVF[d, t]]
            out.append((f, OCC[d, t], d, t, k))
    return out

def prow(d, h, P):  # 사전 예측(전날까지 아는 정보: 달력·공연 일정·기상예보 가정)
    return [h, dtype[d], month[d], days[d].weekday(), P[month[d], dtype[d], h], TEMP[d, h], RAIN[d, h], *EVF[d, h],
            sum(SEATS[v] for v, _ in by_day.get(days[d].isoformat(), []) if v != '야외') / 1000]

GB = dict(max_iter=500, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=1.0, random_state=0)

def evaluate(test_days, train_days):
    P = profile(train_days)
    Rtr = [r for i in train_days for r in nrows(i, P)]
    Xt = np.array([r[0] for r in Rtr]); yt = np.array([r[1] for r in Rtr])
    m = HistGradientBoostingRegressor(**GB).fit(Xt, yt - Xt[:, 3])
    mq = HistGradientBoostingRegressor(loss='quantile', quantile=0.8, **GB).fit(Xt, yt - Xt[:, 3])
    Xp = np.array([prow(i, h, P) for i in train_days for h in range(24)]); yp = OCC[train_days].reshape(-1)
    mp = HistGradientBoostingRegressor(**GB).fit(Xp, yp)
    out = []
    for d in test_days:
        Re = nrows(d, P); Xe = np.array([r[0] for r in Re])
        g = np.maximum(Xe[:, 3] + m.predict(Xe), 0); q = np.maximum(Xe[:, 3] + mq.predict(Xe), 0)
        pre = mp.predict(np.array([prow(d, h, P) for h in range(24)]))
        for (f, y, dd, t, k), gg, qq in zip(Re, g, q):
            sn = OCC[d - 7, t] if d >= 7 and VALID[d - 7] else np.nan
            out.append((dd, t, k, y, P[month[d], dtype[d], t], f[3], sn, gg, qq, pre[t]))
    return np.array(out)

def summarize(res, name):
    d, t, k, y = res[:, 0].astype(int), res[:, 2 - 1].astype(int), res[:, 2], res[:, 3]
    evm = ev_day[d] & (t >= 15) & (t <= 22)
    cols = {'profile': 4, 'persist': 5, 'seasonal_naive': 6, 'nowcast': 7, 'nowcast_q80': 8, 'pre_event_gbm': 9}
    o = {'n': int(len(y)), 'n_event_evening': int(evm.sum()), 'mae_all': {}, 'mae_event_evening': {}, 'mape_event_evening': {}}
    for c, i in cols.items():
        p = res[:, i]; ok = ~np.isnan(p)
        o['mae_all'][c] = round(float(np.abs(p[ok] - y[ok]).mean()), 1)
        o['mae_event_evening'][c] = round(float(np.abs(p[evm & ok] - y[evm & ok]).mean()), 1)
        o['mape_event_evening'][c] = round(float(100 * np.mean(np.abs(p[evm & ok] - y[evm & ok]) / np.maximum(y[evm & ok], 1))), 1)
    for thr, nm in ((0.7, 'busy70'), (0.9, 'full90')):
        T = y >= thr * CAP
        for c in ('persist', 'nowcast', 'nowcast_q80'):
            p = res[:, cols[c]] >= thr * CAP
            tp, fp = int((p & T).sum()), int((p & ~T).sum())
            o[f'{nm}_{c}'] = {'actual': int(T.sum()), 'recall': round(tp / max(T.sum(), 1), 3), 'precision': round(tp / max(tp + fp, 1), 3)}
    # 부트스트랩(공연일 단위) 개선폭
    rng = np.random.default_rng(0); evd = np.unique(d[evm]); B = []
    for _ in range(2000):
        s = rng.choice(evd, len(evd)); m = np.concatenate([np.where(evm & (d == x))[0] for x in s])
        B.append([np.abs(res[m, 4] - y[m]).mean() - np.abs(res[m, 7] - y[m]).mean(), np.abs(res[m, 5] - y[m]).mean() - np.abs(res[m, 7] - y[m]).mean()])
    B = np.array(B)
    o['improve_ci95_vs_profile'] = [round(float(x), 1) for x in np.percentile(B[:, 0], [2.5, 50, 97.5])]
    o['improve_ci95_vs_persist'] = [round(float(x), 1) for x in np.percentile(B[:, 1], [2.5, 50, 97.5])]
    print(name, json.dumps(o, ensure_ascii=False))
    return o

if __name__ == '__main__':
    valid = [d for d in range(7, D) if VALID[d]]
    V = np.array(valid); pk = OCC[V].max(1)
    info = {'days_valid': len(valid), 'event_days_valid': int(ev_day[V].sum()), 'peak_occ': int(pk.max()), 'peak_day': days[int(V[pk.argmax()])].isoformat(),
            'days_ge90': int((pk >= 0.9 * CAP).sum()), 'days_ge70': int((pk >= 0.7 * CAP).sum()), 'arrivals_total_all': int(ARR.sum()),
            'coverage_note': 'month-end days, Aug-Sep (23% sampled), Nov/Dec 7+ excluded'}
    print('INFO', info)
    cv = np.concatenate([evaluate([d for d in valid if month[d] == m], [d for d in valid if month[d] != m]) for m in sorted(set(month[V]))])
    o1 = summarize(cv, 'MONTH-CV')
    ho = evaluate([d for d in valid if month[d] >= 10], [d for d in valid if month[d] <= 7])
    o2 = summarize(ho, 'HOLDOUT Oct-Dec')
    np.save(Path(__file__).parent / 'cv2025.npy', cv)
    np.save(Path(__file__).parent / 'ho2025.npy', ho)
    json.dump({'info': info, 'month_cv': o1, 'holdout_q4': o2}, open(Path(__file__).parent / 'metrics2025.json', 'w'), ensure_ascii=False, indent=1)
