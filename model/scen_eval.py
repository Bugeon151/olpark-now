"""사전(전날) 시나리오 예측 검증: 1~7월 자료로만 만든 기준 곡선 + KSPO DOME 초과 주차 분위 시나리오를
10~12월 공연일에 적용해, 실측이 10~90% 범위 안에 든 비율과 위험 시각(70%·90%·수용 초과) 예측을 평가."""
import json, numpy as np, model2025 as M
from pathlib import Path
CAP, R0 = M.CAP, -10
V = [d for d in range(M.D) if M.VALID[d]]
TR = [d for d in V if M.month[d] <= 7]; TE = [d for d in V if M.month[d] >= 10 and M.ev_day[d]]

def pooled(days, m, t, need=5):
    for w in range(7):
        ms = {(m - 1 + k) % 12 + 1 for k in range(-w, w + 1)}
        idx = [i for i in days if M.dtype[i] == t and not M.ev_day[i] and M.month[i] in ms]
        if len(idx) >= need: return M.OCC[idx].mean(0)
    return M.OCC[[i for i in days if M.dtype[i] == t and not M.ev_day[i]]].mean(0)

def scenarios(days, venue='KSPO DOME'):
    ex = []
    for d in days:
        ss = [s for v, s in M.by_day.get(M.days[d].isoformat(), []) if v == venue]
        if not ss: continue
        s = min(ss); b = pooled(days, M.month[d], M.dtype[d])
        ex.append([M.OCC[d, s + r] - b[s + r] if 0 <= s + r < 24 else 0.0 for r in range(R0, 6)])
    E = np.array(ex)
    return {q: np.maximum(np.percentile(E, q, axis=0), 0) for q in (10, 50, 90)}, len(ex)

SC, n_dome = scenarios(TR)
SO, n_out = scenarios(TR, '야외')  # 야외 행사(88잔디마당 등)는 좌석 수가 없어 별도 분포

def venue_scen(days, venue, need=5):  # KSPO DOME·야외가 없는 날의 해당 공연장 초과 주차 분포(경험값)
    ex = []
    for d in days:
        ses = M.by_day.get(M.days[d].isoformat(), []); vs = {v for v, _ in ses}
        if venue not in vs or 'KSPO DOME' in vs or '야외' in vs: continue
        s = min(x for v, x in ses if v == venue); b = pooled(days, M.month[d], M.dtype[d])
        ex.append([M.OCC[d, s + r] - b[s + r] if 0 <= s + r < 24 else 0.0 for r in range(R0, 6)])
    if len(ex) < need: return None, len(ex)
    E = np.array(ex)
    return {q: np.maximum(np.percentile(E, q, axis=0), 0) for q in (10, 50, 90)}, len(ex)
EMP = {v: venue_scen(TR, v) for v in ('올림픽홀', '핸드볼경기장', '우리금융아트홀')}
def forecast(d):
    b = pooled(TR, M.month[d], M.dtype[d]); add = {q: np.zeros(24) for q in SC}
    ses = M.by_day.get(M.days[d].isoformat(), []); seen = set()
    for v, s in ses:
        if v in seen: continue  # 같은 공연장 여러 회차는 첫 회차 기준 한 번만
        seen.add(v)
        if v == '야외': src, w = SO, 1.0
        elif v == 'KSPO DOME': src, w = SC, 1.0
        elif EMP.get(v, (None, 0))[0] is not None: src, w = EMP[v][0], 1.0
        else: src, w = SC, M.SEATS[v] / 15000
        for q in SC:
            for i, x in enumerate(src[q]):
                h = s + R0 + i
                if 0 <= h < 24: add[q][h] += x * w
    F = {q: b + add[q] for q in SC}
    lo = np.minimum.reduce([F[q] for q in SC]); hi = np.maximum.reduce([F[q] for q in SC])
    return lo, F[50], hi

if __name__ == '__main__':
    first = lambda a, thr: next((h for h in range(6, 24) if a[h] >= thr), None)
    cov, pk_err, res = [], [], {}
    for nm, thr in (('busy70', .7 * CAP), ('full90', .9 * CAP), ('over', CAP)):
        res[nm] = {'actual_days': 0, 'warned_days': 0, 'hit': 0, 'false_alarm': 0, 'lead_h': []}
    for d in TE:
        lo, mid, hi = forecast(d); y = M.OCC[d]
        hs = range(9, 23); cov += [lo[h] <= y[h] <= hi[h] for h in hs]
        pk_err.append(abs(mid.max() - y.max()))
        for nm, thr in (('busy70', .7 * CAP), ('full90', .9 * CAP), ('over', CAP)):
            a, f = first(y, thr), first(hi, thr); r = res[nm]
            r['actual_days'] += a is not None; r['warned_days'] += f is not None
            if a is not None and f is not None: r['hit'] += 1; r['lead_h'].append(a - f)
            if a is None and f is not None: r['false_alarm'] += 1
    out = {'test_event_days': len(TE), 'train_dome_days': n_dome, 'train_outdoor_days': n_out, 'train_venue_days': {v: n for v, (_, n) in EMP.items()}, 'band_coverage_9_22': round(float(np.mean(cov)), 3),
           'peak_mae_mid': round(float(np.mean(pk_err)), 1)}
    for nm, r in res.items():
        L = r.pop('lead_h'); r['median_lead_h'] = float(np.median(L)) if L else None; r['lead_ge0_share'] = round(float(np.mean(np.array(L) >= 0)), 3) if L else None
        out[nm] = r
    print(json.dumps(out, ensure_ascii=False)); json.dump(out, open(Path(__file__).parent / 'scen_eval.json', 'w'), ensure_ascii=False, indent=1)
