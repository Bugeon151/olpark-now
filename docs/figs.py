import sys, json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / 'model'))
from train import OCC, DEP, days, CAP
plt.rcParams.update({'font.family': 'Noto Sans CJK KR', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#8a8984', 'axes.labelcolor': '#52514e', 'xtick.color': '#52514e', 'ytick.color': '#52514e'})
B, O, G, MUT = '#2a78d6', '#eb6834', '#1baf7a', '#8a8984'
out = Path(__file__).parent / 'fig'; out.mkdir(exist_ok=True)
M = json.load(open(Path(__file__).parents[1] / 'demo/model.json'))

# 1) 일별 최대 재차대수(공연일 표시)
ev = {6: '10CM·차은우', 7: '10CM', 12: '더보이즈', 13: '더보이즈·DPR', 14: '더보이즈·DPR', 20: 'SG워너비·휘인', 21: 'SG워너비·휘인', 27: '백현', 28: '백현'}
fig, ax = plt.subplots(figsize=(7.2, 2.8))
pk = OCC.max(1)
ax.bar(range(1, 32), pk, color=[O if d in ev else B for d in range(1, 32)], width=0.72)
for thr, lab, c in ((0.7, '혼잡 70%', '#fab219'), (0.9, '만차 90%', '#d03b3b')):
    ax.axhline(thr * CAP, ls='--', lw=1, color=c); ax.text(31.6, thr * CAP + 25, lab, ha='right', fontsize=8, color='#52514e')
for d in (20, 21): ax.text(d, pk[d - 1] + 30, f'{int(pk[d-1]):,}', ha='center', fontsize=8)
ax.set_xlabel('2024년 7월 (일)'); ax.set_ylabel('최대 재차대수(대)'); ax.set_xticks([1, 5, 10, 15, 20, 25, 31])
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=B), plt.Rectangle((0, 0), 1, 1, color=O)], labels=['공연 없음', '공연일'], frameon=False, loc='upper left', fontsize=8)
fig.tight_layout(); fig.savefig(out / 'f1_daily_peak.png', dpi=200); plt.close(fig)

# 2) 7/20 나우캐스트 재현(기준 15시)
R = M['replay']['2024-07-20']; h0 = 15
fig, ax = plt.subplots(figsize=(7.2, 2.9))
hs = np.arange(24)
ax.plot(hs, R['profile'], ls='--', color=MUT, lw=1.3, label='평균 프로파일(사전)')
ax.plot(hs[:h0 + 1], R['actual'][:h0 + 1], color=B, lw=2.2, label='실측')
ax.plot(hs[h0:], R['actual'][h0:], color=B, lw=1.5, alpha=.35)
nc = R['nowcast'][str(h0)]; xs = [h0] + [int(k) for k in nc]; ys = [R['actual'][h0]] + list(nc.values())
ax.plot(xs, ys, color=O, lw=2.2, marker='o', ms=5, label='나우캐스트(1~3시간 뒤)')
ax.axhline(R['actual'][h0], color=MUT, lw=1, ls=':', label='현재값 유지')
ax.axvline(h0, color='#52514e', lw=.8); ax.text(h0 + .2, 2330, '현재 15:00', fontsize=8)
for thr, c in ((0.7, '#fab219'), (0.9, '#d03b3b')): ax.axhline(thr * CAP, ls='--', lw=.9, color=c)
ax.set_ylim(0, 2450); ax.set_xlim(0, 23); ax.set_xlabel('시각'); ax.set_ylabel('재차대수(대)')
ax.legend(frameon=False, fontsize=8, loc='upper left', ncol=2)
fig.tight_layout(); fig.savefig(out / 'f2_replay_0720.png', dpi=200); plt.close(fig)

# 3) 정확도 비교
mt = M['metrics']; names = ['평균 프로파일', '현재값 유지', '올팍 나우']
vals = [mt['profile']['mae_event_evening'], mt['persist']['mae_event_evening'], mt['gbm']['mae_event_evening']]
fig, ax = plt.subplots(figsize=(5.2, 2.3))
ax.barh(names[::-1], vals[::-1], color=[O, MUT, MUT], height=.55)
for i, v in enumerate(vals[::-1]): ax.text(v + 5, i, f'{v:.1f}대', va='center', fontsize=9)
ax.set_xlabel('공연일 저녁 1~3시간 뒤 예측 MAE(대, 낮을수록 좋음)'); ax.set_xlim(0, 400)
fig.tight_layout(); fig.savefig(out / 'f3_mae.png', dpi=200); plt.close(fig)

# 4) 출차 몰림: KSPO DOME 공연일 시작 기준 출차량
fig, ax = plt.subplots(figsize=(5.6, 2.5))
dome = {12: 19, 13: 17, 14: 16, 20: 18, 21: 17, 27: 18, 28: 16}
for d, s in dome.items():
    y = [DEP[d - 1, s + r] if s + r < 24 else np.nan for r in range(0, 6)]
    ax.plot(range(0, 6), y, color=O if d in (20, 21) else B, alpha=.85 if d in (20, 21) else .45, lw=1.6)
ax.axvline(3, color='#52514e', lw=.8, ls='--'); ax.text(3.05, ax.get_ylim()[1] * .92, '시작 +3시간', fontsize=8)
ax.set_xlabel('공연 시작 후 경과 시간(h)'); ax.set_ylabel('시간당 출차(대)')
fig.tight_layout(); fig.savefig(out / 'f4_exit.png', dpi=200); plt.close(fig)
print('ok', sorted(p.name for p in out.iterdir()))

# 6) 구역별 최초 만차 일수(혼잡도 정보)
Z = json.load(open(Path(__file__).parents[1] / 'data/derived/zone_stats.json'))['zones']
it = sorted(Z.items(), key=lambda kv: -kv[1]['first_full_days'])
fig, ax = plt.subplots(figsize=(6.4, 2.5))
ax.bar([k for k, _ in it], [v['first_full_days'] for _, v in it], color=[O if v['full_hour_median'] <= 13 else B for _, v in it], width=.65)
for i, (_, v) in enumerate(it): ax.text(i, v['first_full_days'] + .6, f"{v['full_hour_median']}시", ha='center', fontsize=7.5, color='#52514e')
ax.set_ylabel('최초 만차 일수(일)'); ax.set_ylim(0, 36)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=O), plt.Rectangle((0, 0), 1, 1, color=B)], labels=['만차 시각 중앙값 13시', '14~15시'], frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(out / 'f6_zones.png', dpi=200); plt.close(fig)

# 7) 서비스 구조도
fig, ax = plt.subplots(figsize=(7.2, 2.6)); ax.axis('off'); ax.set_xlim(0, 100); ax.set_ylim(0, 36)
def box(x, y, w, h, t, c='#eef3fa'):
    ax.add_patch(matplotlib.patches.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.6', fc=c, ec='#8a8984', lw=.8))
    ax.text(x + w / 2, y + h / 2, t, ha='center', va='center', fontsize=8.2)
def arr(x1, y1, x2, y2): ax.annotate('', (x2, y2), (x1, y1), arrowprops=dict(arrowstyle='->', color='#52514e', lw=1))
box(1, 25, 24, 8, '실시간 주차현황 API\n(15분~1시간 주기)'); box(1, 14, 24, 8, '공연정보·영문 공연정보\n주차장 혼잡도 정보'); box(1, 3, 24, 8, '입출차 이력·월별 주차\n기상 재분석')
box(37, 25, 24, 8, '수집·가공 배치\n(시간대별 재차대수)', '#fff4e0'); box(37, 14, 24, 8, '모델 학습·검증\n(LODO, 월 1회 재학습)', '#fff4e0'); box(37, 3, 24, 8, '나우캐스트\n(1~3시간 뒤·혼잡 경보)', '#fff4e0')
box(73, 25, 25, 8, '검증·데이터 품질 공개'); box(73, 14, 25, 8, '기관: 운영자 모드\n(경보·인력·구역 유도)'); box(73, 3, 25, 8, '관람객: 공연일 플래너\n(국문·영문, 모바일)')
for y in (29, 18, 7): arr(26.2, y, 35.8, 29)
arr(49, 24.6, 49, 22.4); arr(49, 13.6, 49, 11.4); arr(62.2, 29, 71.8, 29); arr(62.2, 7, 71.8, 18); arr(62.2, 7, 71.8, 7); arr(62.2, 18, 71.8, 29)
fig.tight_layout(); fig.savefig(out / 'f7_arch.png', dpi=200); plt.close(fig)
print('ok2')
