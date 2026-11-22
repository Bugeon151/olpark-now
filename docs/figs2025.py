"""기획서 그림(2025 전수 집계 기준). 실행: python3 docs/figs2025.py"""
import sys, json, datetime as dt, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'model'))
import model2025 as M
plt.rcParams.update({'font.family': 'Noto Sans CJK KR', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#8a8984', 'axes.labelcolor': '#52514e', 'xtick.color': '#52514e', 'ytick.color': '#52514e'})
B, O, MUT, WARN, CRIT = '#2a78d6', '#eb6834', '#8a8984', '#fab219', '#d03b3b'
out = Path(__file__).parent / 'fig'; out.mkdir(exist_ok=True)
J = json.load(open(ROOT / 'demo/model.json')); CAP = M.CAP
save = (lambda f, n: (f.tight_layout(), f.savefig(out / n, dpi=200), plt.close(f)))

# g1) 2025 일별 최대 재차(원천 미수록·표본 수록 구간 표시)
fig, ax = plt.subplots(figsize=(7.2, 2.7))
pk = M.OCC.max(1); x = np.arange(M.D)
for i in x:
    if M.VALID[i]: ax.bar(i, pk[i], color=O if M.ev_day[i] else B, width=1.0, lw=0)
    else: ax.axvspan(i - .5, i + .5, color='#e6e4de', lw=0)
for thr, lab, c in ((0.7, '혼잡 70%', WARN), (0.9, '만차 90%', CRIT), (1.0, '공개 수용 2,200대', '#52514e')):
    ax.axhline(thr * CAP, ls='--', lw=.9, color=c); ax.text(244, thr * CAP + {0.7: 30, 0.9: -30, 1.0: 40}[thr], lab, ha='center', va='top' if thr == 0.9 else 'baseline', fontsize=7.5, color='#52514e')
ms = [(dt.date(2025, m, 1) - dt.date(2025, 1, 1)).days for m in range(1, 13)]
ax.set_xticks(ms); ax.set_xticklabels([f'{m}월' for m in range(1, 13)]); ax.set_xlim(-1, M.D); ax.set_ylabel('일 최대 재차(대)')
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=B), plt.Rectangle((0, 0), 1, 1, color=O), plt.Rectangle((0, 0), 1, 1, color='#e6e4de')],
          labels=['공연 없는 날', '공연일', '원천 미수록·표본 수록(제외)'], frameon=False, fontsize=7.5, loc='upper right', ncol=3, bbox_to_anchor=(1, 1.13))
save(fig, 'g1_daily_peak_2025.png')

# g2) KSPO DOME 공연일 68일: 시작 시각 기준 초과 주차(분위 띠) + 출차 피크 위치
sc = J['dome_excess_scenarios_rel_start_-6_to_+5']; r = np.arange(J['scen_from'], J['scen_from'] + len(sc['mid']))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.5), gridspec_kw={'width_ratios': [1.5, 1]})
a1.fill_between(r, sc['low'], sc['high'], color=B, alpha=.2, lw=0, label='10~90% 범위'); a1.plot(r, sc['mid'], color=B, lw=2, label='중앙값')
a1.axvline(0, color='#52514e', lw=.8); a1.text(.2, max(sc['high']) * .95, '공연 시작', fontsize=8)
a1.set_xlabel('공연 시작 기준 시간(h)'); a1.set_ylabel('평소 대비 초과 재차(대)'); a1.legend(frameon=False, fontsize=7.5, loc='upper left')
cnt = J['dep_peak_rel_counts']; ks = sorted(int(k) for k in cnt)
a2.bar([str(k) for k in ks], [cnt[str(k)] for k in ks], color=[O if k in (2, 3) else MUT for k in ks], width=.6)
for k in ks: a2.text(str(k), cnt[str(k)] + .8, cnt[str(k)], ha='center', fontsize=8)
a2.set_xlabel('출차 최대 시각(시작 +h)'); a2.set_ylabel('공연일 수'); a2.set_ylim(0, 48)
save(fig, 'g2_dome_excess_exit.png')

# g3) 정확도(홀드아웃)
ho = J['metrics2025']['holdout_q4']['mae_event_evening']
names = [('profile', '평균 프로파일'), ('seasonal_naive', '전주 같은 시각'), ('pre_event_gbm', '사전 예측(공연·달력·기상)'), ('persist', '현재값 유지'), ('event_shape_persist', '현재값+공연일 평균 변화'), ('nowcast', '올팍 나우 나우캐스트')]
fig, ax = plt.subplots(figsize=(6.0, 2.7))
v = [ho[k] for k, _ in names]
ax.barh([n for _, n in names], v, color=[MUT] * 5 + [O], height=.55)
for i, val in enumerate(v): ax.text(val + 10, i, f'{val:.1f}대', va='center', fontsize=8.5)
ax.invert_yaxis(); ax.set_xlim(0, 1000); ax.set_xlabel('10~12월 홀드아웃 공연일 15~22시, 1~3시간 뒤 예측 MAE(대)')
save(fig, 'g3_mae_2025.png')

# g4) 2025-11-01 재현(15시 기준)
R = J['replay']['2025-11-01']; h0 = 15; hs = np.arange(24)
fig, ax = plt.subplots(figsize=(7.2, 2.8))
ax.plot(hs, R['profile'], ls='--', color=MUT, lw=1.3, label='평균 프로파일(같은 달·주말)')
ax.plot(hs[:h0 + 1], R['actual'][:h0 + 1], color=B, lw=2.2, label='실측 재차')
ax.plot(hs[h0:], R['actual'][h0:], color=B, lw=1.4, alpha=.35)
nc = R['nowcast'][str(h0)]; xs = [h0] + [int(k) for k in nc]; ys = [R['actual'][h0]] + list(nc.values())
ax.plot(xs, ys, color=O, lw=2.2, marker='o', ms=5, label='나우캐스트(1~3시간 뒤)')
ax.axvline(h0, color='#52514e', lw=.8); ax.text(h0 + .2, 3450, '현재 15:00', fontsize=8)
for thr, c in ((0.7, WARN), (0.9, CRIT), (1.0, '#52514e')): ax.axhline(thr * CAP, ls='--', lw=.8, color=c)
ax.set_ylim(0, 3700); ax.set_xlim(0, 23); ax.set_xlabel('시각'); ax.set_ylabel('재차(대)'); ax.legend(frameon=False, fontsize=7.5, loc='upper left')
save(fig, 'g4_replay_1101.png')

# g7) 예측 → 보정 → 결정 구조
fig, ax = plt.subplots(figsize=(7.2, 2.7)); ax.axis('off'); ax.set_xlim(0, 100); ax.set_ylim(0, 38)
def box(x, y, w, h, t, c='#eef3fa'):
    ax.add_patch(matplotlib.patches.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.6', fc=c, ec='#8a8984', lw=.8))
    ax.text(x + w / 2, y + h / 2, t, ha='center', va='center', fontsize=7.8)
def arr(x1, y1, x2, y2): ax.annotate('', (x2, y2), (x1, y1), arrowprops=dict(arrowstyle='->', color='#52514e', lw=1))
ax.text(13, 36, '① 예측(전날)', ha='center', fontsize=9, weight='bold'); ax.text(49, 36, '② 보정(당일)', ha='center', fontsize=9, weight='bold'); ax.text(85, 36, '③ 결정', ha='center', fontsize=9, weight='bold')
box(1, 24, 24, 8, '공연 일정·좌석 수\n같은 달 평소 패턴'); box(1, 13, 24, 8, 'KSPO DOME 68일\n초과 주차 분포(10~90%)'); box(1, 2, 24, 8, '구역 만차 순서\n(혼잡도 699건)')
box(37, 24, 24, 8, '실시간 주차현황 API\n입출차 주기 수집', '#fff4e0'); box(37, 13, 24, 8, '나우캐스트\n1~3시간 뒤 재차·80% 위험측', '#fff4e0'); box(37, 2, 24, 8, '범위 좁힘·경보 갱신\n(70%·90%·수용 초과)', '#fff4e0')
box(73, 24, 25, 8, '사전 안내·대중교통 권고\n구역 유도 표지 시각'); box(73, 13, 25, 8, '임시 주차장 개방 규모\n진입 통제 준비'); box(73, 2, 25, 8, '출차 유도 인력(+2~3h)\n관람객 시차 귀가 안내')
for y in (28, 17, 6): arr(26.2, y, 35.8, 17)
arr(49, 23.6, 49, 21.4); arr(49, 12.6, 49, 10.4)
for y in (28, 17, 6): arr(62.2, 6, 71.8, y)
save(fig, 'g7_flow.png')
print('ok')

# g8) 현장 검증 2026-11-22(일) KSPO DOME 17시 공연 — 실시간 API(새벽 기준선 보정) vs 전날 범위·14시 나우캐스트
LV = json.load(open(ROOT / 'model/live_eval.json'))['field']
act = np.array([np.nan if v is None else v for v in LV['actual_adj']], float); band = np.array(LV['pre_band_adj'], float); off = LV['offset_night']
fig, ax = plt.subplots(figsize=(7.2, 2.7)); hs = np.arange(24)
ax.fill_between(hs, band[:, 0], band[:, 1], color=B, alpha=.18, lw=0, label='전날 예측 범위(10~90%)')
ax.plot(hs, LV['pre_mid_adj'], color=B, lw=1.8, label='전날 예측(중간)')
ax.plot(hs, act, color=O, lw=2.2, marker='o', ms=3.5, label='실측(실시간 API, 기준선 보정)')
nc = LV['nowcast_raw_by_h0']['14']; xs = [14] + [int(k) for k in nc]; ys = [act[14]] + [v - off for v in nc.values()]
ax.plot(xs, ys, color='#52514e', lw=1.6, ls='--', marker='s', ms=4, label='14시 기준 나우캐스트')
for thr, c in ((0.7, WARN), (0.9, CRIT)): ax.axhline(thr * CAP, ls='--', lw=.8, color=c)
ax.axvline(17, color='#8a8984', lw=.8); ax.text(17.2, 2350, '공연 17:00', fontsize=8)
ax.set_xlim(0, 23); ax.set_ylim(0, 2700); ax.set_xlabel('시각'); ax.set_ylabel('주차 차량(대)'); ax.legend(frameon=False, fontsize=7.2, loc='upper left')
save(fig, 'g8_field_1122.png')
print('g8 ok')
