"""2025년 12개월 결과 → 데모 JSON(model.json 덮어쓰기, 2024-07 결과는 m2024로 보존)."""
import json, numpy as np
from pathlib import Path
import model2025 as M

P = Path(__file__).parent; ROOT = P.parent
cv = np.load(P / 'cv2025.npy'); ho = np.load(P / 'ho2025.npy')
met = json.load(open(P / 'metrics2025.json'))
OCC, DEP, D, days, CAP = M.OCC, M.DEP, M.D, M.days, M.CAP
allidx = [d for d in range(D) if M.VALID[d]]  # 원천 수록이 완전한 240일
Pm = M.profile(allidx)  # [month, dtype, 24] — 초과수요 계산용(평가 기준과 동일)

def pooled(m, t, need=5):  # 데모 기준 곡선: 같은 요일유형의 공연 없는 날을 인접 월로 넓혀 5일 이상 확보
    for w in range(7):
        ms = {(m - 1 + k) % 12 + 1 for k in range(-w, w + 1)}
        idx = [i for i in allidx if M.dtype[i] == t and not M.ev_day[i] and M.month[i] in ms]
        if len(idx) >= need: return OCC[idx].mean(0), len(idx), w
    idx = [i for i in allidx if M.dtype[i] == t and not M.ev_day[i]]
    return OCC[idx].mean(0), len(idx), 6
PP = {m: {k: pooled(m, t) for t, k in enumerate(('weekday', 'friday', 'weekend'))} for m in range(1, 13)}

# KSPO DOME 공연일 초과수요(주 시작시각 정렬 -10..+5)
dome = {}
for d in allidx:
    ss = [s for v, s in M.by_day.get(days[d].isoformat(), []) if v == 'KSPO DOME']
    if ss: dome[d] = min(ss)
ex = {d: [float(OCC[d, s + r] - Pm[M.month[d], M.dtype[d], s + r]) if 0 <= s + r < 24 else 0.0 for r in range(-10, 6)] for d, s in dome.items()}
pk = {d: max(v) for d, v in ex.items()}
order = sorted(pk, key=pk.get)
q = lambda f: order[min(len(order) - 1, int(round(f * (len(order) - 1))))]
EX = np.array(list(ex.values()))  # 시각별 분위수(특정 하루의 다른 행사 영향이 곡선 전체를 끌지 않도록)
scen = {k: [max(0.0, round(float(x))) for x in np.percentile(EX, f * 100, axis=0)] for k, f in (('low', .1), ('mid', .5), ('high', .9))}
dep_rel = []
for d, s in dome.items():
    y = [DEP[d, s + r] - 0 for r in range(6) if s + r < 24]
    dep_rel.append(int(np.argmax(y)))
dep_rel_counts = {str(k): dep_rel.count(k) for k in sorted(set(dep_rel))}

# 연간 운영 지표(공연일·비공연일 전체 시간)
OV = OCC[allidx]
busy_h = int((OV >= .7 * CAP).sum()); full_h = int((OV >= .9 * CAP).sum())
busy_days = int((OV.max(1) >= .7 * CAP).sum()); full_days = int((OV.max(1) >= .9 * CAP).sum())

# 10~12월 검증 기간 재현: 공연일 중 최대 재차가 큰 2일
hod = [int(x) for x in np.unique(ho[:, 0])]
cand = sorted([d for d in hod if M.ev_day[d]], key=lambda d: -OCC[d].max())[:2]
replay = {}
for d in cand:
    r = ho[ho[:, 0] == d]
    replay[days[d].isoformat()] = {
        'actual': OCC[d].round().tolist(), 'profile': Pm[M.month[d], M.dtype[d]].round().tolist(),
        'events': [[v, s] for v, s in M.by_day.get(days[d].isoformat(), [])],
        'nowcast': {int(h0): {int(t): round(float(g)) for (_, t, k, y, pr, pe, sn, g, qq, pre) in r if int(t) - int(k) == h0} for h0 in M.ORIG}}

out = json.load(open(ROOT / 'demo/model.json'))
out['m2024'] = {k: out[k] for k in ('metrics', 'replay', 'profile', 'dome_excess_scenarios_rel_start_-6_to_+5') if k in out}
out.update({
    'source_year': 2025, 'cap': CAP,
    'profile_m': {str(m): {k: PP[m][k][0].round().tolist() for k in PP[m]} for m in range(1, 13)},
    'profile_m_days_window': {str(m): {k: [PP[m][k][1], PP[m][k][2]] for k in PP[m]} for m in range(1, 13)},
    'dome_excess_scenarios_rel_start_-6_to_+5': scen, 'scen_from': -10, 'dome_days': len(dome), 'dome_peak_excess_p10_p50_p90_max': [round(pk[q(.1)]), round(pk[q(.5)]), round(pk[q(.9)]), round(max(pk.values()))],
    'dep_peak_rel_counts': dep_rel_counts, 'annual': {'valid_days': len(allidx), 'busy_hours': busy_h, 'full_hours': full_h, 'busy_days': busy_days, 'full_days': full_days},
    'metrics2025': met, 'replay': replay,
})
json.dump(out, open(ROOT / 'demo/model.json', 'w'), ensure_ascii=False, separators=(',', ':'))
print(json.dumps({k: out[k] for k in ('dome_days', 'dome_peak_excess_p10_p50_p90_max', 'dep_peak_rel_counts', 'annual')}, ensure_ascii=False), list(replay))
