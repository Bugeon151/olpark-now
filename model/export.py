"""데모용 JSON 생성: 기준 프로파일, 공연 초과수요 시나리오(저·중·고), 7/20·7/21 나우캐스트 리플레이, 검증지표."""
import json, numpy as np
from pathlib import Path
from train import OCC, DEP, ARR, D, H, days, dtype, CAP, EV
P = Path(__file__).parent
z = np.load(P / 'nowcast_eval.npz'); meta = z['meta']

wk = [i for i in range(D) if dtype[i] == 0 and i != 11]            # 평일(공연 없는 날; 7/12 금 공연일 제외)
wkend_base = OCC[[5, 6]].mean(0)                                     # 7/6·7/7: 소규모(올림픽홀) 공연만 있던 주말
prof = {'weekday': OCC[wk].mean(0), 'weekend': wkend_base}
dep_prof = {'weekday': DEP[wk].mean(0), 'weekend': DEP[[5, 6]].mean(0)}

# KSPO DOME 공연일 초과수요(시작시각 기준 정렬, -6h..+5h)
dome = {11: 19, 12: 17, 13: 16, 19: 18, 20: 17, 26: 18, 27: 16}     # 0-index day: 시작시각
ex = {}
for d, s in dome.items():
    base = prof['weekday'] if dtype[d] == 0 else prof['weekend']
    ex[d] = [float(OCC[d, s + r] - base[s + r]) if 0 <= s + r < H else 0.0 for r in range(-6, 6)]
peaks = {d: max(v) for d, v in ex.items()}
order = sorted(peaks, key=peaks.get)
scen = {'low': ex[order[0]], 'mid': ex[order[len(order) // 2]], 'high': ex[order[-1]]}
scen = {k: [max(0.0, round(x)) for x in v] for k, v in scen.items()}

dep_ex = {}
for d, s in dome.items():
    base = dep_prof['weekday'] if dtype[d] == 0 else dep_prof['weekend']
    dep_ex[d] = [float(DEP[d, s + r] - base[s + r]) if 0 <= s + r < H else 0.0 for r in range(0, 6)]
dep_peak_rel = {days[d].isoformat(): int(np.argmax(v)) for d, v in dep_ex.items()}

replay = {}
for d in (19, 20):
    rows = meta[meta[:, 0] == d]
    replay[days[d].isoformat()] = {
        'actual': OCC[d].round().tolist(),
        'profile': prof['weekend'].round().tolist(),
        'nowcast': {int(h0): {int(t): round(float(g)) for (_, hh, t, _, _), g in zip(rows, z['gbm'][meta[:, 0] == d]) if hh == h0}
                    for h0 in sorted(set(rows[:, 1].astype(int)))},
        'risk80': {int(h0): {int(t): round(float(g)) for (_, hh, t, _, _), g in zip(rows, z['gbm_q80'][meta[:, 0] == d]) if hh == h0}
                   for h0 in sorted(set(rows[:, 1].astype(int)))},
    }

y = meta[:, 4]; t = meta[:, 2]; evm = np.array([any(s[0] == days[int(d)].isoformat() for s in EV['sessions_2024_07']) for d in meta[:, 0]]) & (t >= 15) & (t <= 22)
metrics = {n: {'mae_all': round(float(np.abs(z[n] - y).mean()), 1), 'mae_event_evening': round(float(np.abs(z[n] - y)[evm].mean()), 1)}
           for n in ('profile', 'persist', 'gbm', 'gbm_q80')}
out = {
    'cap': CAP, 'profile': {k: v.round().tolist() for k, v in prof.items()},
    'dep_profile': {k: v.round().tolist() for k, v in dep_prof.items()},
    'dome_excess_scenarios_rel_start_-6_to_+5': scen,
    'dome_peaks_by_day': {days[d].isoformat(): round(v) for d, v in peaks.items()},
    'dep_peak_hours_after_start': dep_peak_rel,
    'replay': replay, 'metrics': metrics,
    'venues': EV['venues'], 'sessions_upcoming': EV['sessions_upcoming'],
}
(P.parent / 'demo').mkdir(exist_ok=True)
json.dump(out, open(P.parent / 'demo/model.json', 'w'), ensure_ascii=False, separators=(',', ':'))
print(json.dumps({k: out[k] for k in ('dome_peaks_by_day', 'dep_peak_hours_after_start', 'metrics')}, ensure_ascii=False))
print('scen', scen)
print('profiles wd', out['profile']['weekday']); print('we', out['profile']['weekend'])
