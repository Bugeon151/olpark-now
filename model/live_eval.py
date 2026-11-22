"""실시간 주차 대수(한국체육산업개발 공개 API, parking_sum)로 공연일 당일 나우캐스트를 현장 검증.
실시간 API는 주차 대수만 주므로 입·출차 대수와 관람객 수 없이 동작하는 '실시간용' 나우캐스트를 쓴다.
1) 2025 데이터로 실시간용 모델의 홀드아웃 성능(1~7월 학습 → 10~12월)
2) 현장 검증: 수집한 parking_sum을 hh:30 시계열로 만들고, 새벽 기준선 차이(장기주차 등 집계 정의 차이)를 보정해 비교
사용: python3 live_eval.py <live.csv ...> [--date 2026-11-22]"""
import sys, json, datetime as dt, numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
import model2025 as M

KEEP = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19]  # 입·출차, 기상, 관람객 수 제외
VALID = [d for d in range(7, M.D) if M.VALID[d]]
P_ALL = M.profile(VALID)

def rows(days, P):
    R = [r for d in days for r in M.nrows(d, P)]
    return np.array([[r[0][i] for i in KEEP] for r in R]), np.array([r[1] for r in R]), R

def fit(days, P):
    X, y, _ = rows(days, P)
    return HistGradientBoostingRegressor(**M.GB).fit(X, y - X[:, 3])

def holdout():
    tr = [d for d in VALID if M.month[d] <= 7]; te = [d for d in VALID if M.month[d] >= 10]
    P = M.profile(tr); m = fit(tr, P)
    X, y, R = rows(te, P)
    p = np.maximum(X[:, 3] + m.predict(X), 0)
    d = np.array([r[2] for r in R]); t = np.array([r[3] for r in R])
    ev = M.ev_day[d] & (t >= 15) & (t <= 22)
    mae = lambda a: round(float(np.abs(a[ev] - y[ev]).mean()), 1)
    return {'n_event_evening': int(ev.sum()), 'nowcast_live': mae(p), 'persist': mae(X[:, 3]),
            'mape_nowcast_live': round(float(100 * np.mean(np.abs(p[ev] - y[ev]) / np.maximum(y[ev], 1))), 1)}

def load_live(paths, date):
    pts = []
    for p in paths:
        for line in open(p):
            s = line.strip().split(',')
            if len(s) >= 2 and s[1]:
                pts.append((dt.datetime.fromisoformat(s[0][:19]), int(s[1])))
    pts.sort(); day = dt.datetime.fromisoformat(date)
    occ = np.full(24, np.nan)
    for h in range(24):
        tgt = day + dt.timedelta(hours=h, minutes=30)
        near = [(abs((t - tgt).total_seconds()), v) for t, v in pts if abs((t - tgt).total_seconds()) <= 900]
        if near: occ[h] = min(near)[1]
    ok = ~np.isnan(occ)
    if ok.sum() >= 2: occ[~ok] = np.interp(np.where(~ok)[0], np.where(ok)[0], occ[ok])
    return occ, pts

def ev_rows(sessions, h):
    pre = dur = post = dome = outdoor = 0.0
    for v, s in sessions:
        k = M.SEATS[v] / 1000
        if v == '야외':
            if s - 2 <= h <= s + 6: outdoor = 1
            continue
        if s - 3 <= h < s: pre += k * (h - s + 4) / 3
        if s <= h <= s + 3: dur += k
        if s + 2 <= h <= s + 4: post += k
        if v == 'KSPO DOME' and s - 3 <= h <= s + 3: dome = 1
    return [pre, dur, post, dome, outdoor]

def field(paths, date, sessions):
    occ_raw, pts = load_live(paths, date)
    d = dt.date.fromisoformat(date); month = d.month; dtype = 2 if d.weekday() >= 5 else (1 if d.weekday() == 4 else 0)
    base = P_ALL[month, dtype]
    night = [3, 4, 5]
    offset = float(np.nanmean(occ_raw[night]) - base[night].mean())  # 집계 정의 차이(새벽 기준선)
    occ = occ_raw - offset
    m = fit(VALID, P_ALL)
    J = json.load(open(Path(__file__).parents[1] / 'demo/model.json'))
    sc = J['dome_excess_scenarios_rel_start_-6_to_+5']; r0 = J['scen_from']
    pm = np.array(J['profile_m'][str(month)][['weekday', 'friday', 'weekend'][dtype]], float)
    PR = {}
    for q in ('low', 'mid', 'high'):  # 데모와 같은 전날 시나리오(공연장별 분포)
        arr = pm.copy(); seen = set()
        for v, s in sorted(sessions, key=lambda x: x[1]):
            if v in seen: continue
            seen.add(v)
            VS = J.get('venue_excess_scenarios', {}).get(v)
            src = J['outdoor_excess_scenarios'] if v == '야외' else sc if v == 'KSPO DOME' else VS or sc
            w = 1 if (v in ('야외', 'KSPO DOME') or VS) else J['venues'].get(v, {'seats': 0})['seats'] / 15000
            for i, x in enumerate(src[q]):
                h = s + r0 + i
                if 0 <= h < 24: arr[h] += x * w
        PR[q] = arr
    pre = PR['mid']; lo = np.minimum.reduce(list(PR.values())); hi = np.maximum.reduce(list(PR.values()))
    out, E = {}, {'nowcast': [], 'persist': [], 'pre': []}
    for h0 in M.ORIG:
        if np.isnan(occ[h0]): continue
        for k in M.K:
            t = h0 + k
            f = [k, h0, t, occ[h0], occ[h0] - occ[h0 - 1], occ[h0] - occ[h0 - 2], base[h0], base[t], occ[h0] - base[h0], dtype, month, *ev_rows(sessions, t)]
            p = max(0.0, occ[h0] + float(m.predict(np.array([f]))[0]))
            out.setdefault(h0, {})[t] = round(p + offset)
            if 15 <= t <= 22:
                E['nowcast'].append(abs(p - occ[t])); E['persist'].append(abs(occ[h0] - occ[t])); E['pre'].append(abs(pre[t] - occ[t]))
    res = {'date': date, 'samples': len(pts), 'offset_night': round(offset), 'actual_raw': [None if np.isnan(x) else int(x) for x in occ_raw],
           'actual_adj': [None if np.isnan(x) else round(float(x)) for x in occ], 'pre_mid_adj': [round(float(x)) for x in pre],
           'nowcast_raw_by_h0': {int(a): {int(b): c for b, c in v.items()} for a, v in out.items()},
           'mae_evening': {k: round(float(np.mean(v)), 1) for k, v in E.items() if v}, 'n_evening': len(E['nowcast']),
           'pre_band_adj': [[round(float(a)), round(float(b))] for a, b in zip(lo, hi)],
           'pre_band_coverage_9_22': round(float(np.mean([lo[h] <= occ[h] <= hi[h] for h in range(9, 23)])), 3),
           'peak_actual_adj': [int(np.nanargmax(occ)), round(float(np.nanmax(occ)))], 'peak_pre_mid': [int(np.argmax(pre)), round(float(pre.max()))],
           'first_cross': {nm: {'actual': next((h for h in range(6, 24) if occ[h] >= f * M.CAP), None), 'pre_high': next((h for h in range(6, 24) if hi[h] >= f * M.CAP), None)}
                           for nm, f in (('busy70', .7), ('full90', .9), ('over', 1.0))}}
    return res

if __name__ == '__main__':
    av = sys.argv[1:]; date = '2026-11-22'
    if '--date' in av: i = av.index('--date'); date = av[i + 1]; av = av[:i] + av[i + 2:]
    args = av
    out = {'holdout_live_model': holdout()}
    print('HOLDOUT', out['holdout_live_model'])
    if args:
        out['field'] = field(args, date, [('KSPO DOME', 17)])
        print('FIELD', json.dumps({k: out['field'][k] for k in ('samples', 'offset_night', 'mae_evening', 'n_evening')}, ensure_ascii=False))
    json.dump(out, open(Path(__file__).parent / 'live_eval.json', 'w'), ensure_ascii=False, indent=1)
