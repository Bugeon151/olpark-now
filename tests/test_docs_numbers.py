"""기획서(build_docx.js)에 적힌 핵심 수치가 원자료·지표 파일에서 다시 계산한 값과 같은지 확인."""
import json, numpy as np, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'model'))
DOC = (ROOT / 'docs/build_docx.js').read_text()
M25 = json.load(open(ROOT / 'model/metrics2025.json')); J = json.load(open(ROOT / 'demo/model.json'))
f1 = lambda x: f'{x:.1f}'
c = lambda n: f'{int(round(n)):,}'

def has(*xs):
    for x in xs: assert x in DOC, x

def test_holdout_and_cv_table():
    ho, cv = M25['holdout_q4'], M25['month_cv']
    for k in ('profile', 'seasonal_naive', 'persist', 'pre_event_gbm', 'nowcast'):
        has(f"'{f1(cv['mae_event_evening'][k])}', '{f1(ho['mae_event_evening'][k])}', '{f1(ho['mape_event_evening'][k])}%'")
    has(f"{ho['improve_ci95_vs_persist'][0]}~{ho['improve_ci95_vs_persist'][2]}대", f"{ho['improve_ci95_vs_profile'][0]}~{ho['improve_ci95_vs_profile'][2]}대")
    has(f"재현율 {ho['busy70_nowcast']['recall']}·정밀도 {ho['busy70_nowcast']['precision']}", f"({ho['busy70_persist']['recall']}·{ho['busy70_persist']['precision']})")
    has(f"재현율 {ho['full90_nowcast_q80']['recall']}·정밀도 {ho['full90_nowcast_q80']['precision']}")
    red = 1 - ho['mae_event_evening']['nowcast'] / ho['mae_event_evening']['persist']; has(f'보다 {round(100 * red)}%, 가장 강한 기준')
    has(f"평가 시점은 월 교차검증 {c(cv['n_event_evening'])}개, 홀드아웃 {ho['n_event_evening']}개")

def test_data_facts():
    import model2025 as M
    V = np.where(M.VALID)[0]; pk = M.OCC[V].max(1)
    full = pk >= 0.9 * M.CAP; over = M.OCC[V] > M.CAP
    has(f'만차(90%, 1,980대)에 이른 {int(full.sum())}일은 모두 공연일', f'공연일 {int((over.any(1)).sum())}일, {int(over.sum())}시간', c(pk.max()))
    assert M.ev_day[V][full].all() and M.ev_day[V][over.any(1)].all()
    p = J['dome_peak_excess_p10_p50_p90_max']; has(f'{c(p[0])}~{c(p[2])}대', f'최대 {c(p[3])}대', f"{J['dome_days']}일")
    d = J['dep_peak_rel_counts']; has(f"68일 중 {d['2'] + d['3']}일")
    has('2,411,225', '1,975,475', '0.952', '247일')

def test_replay_numbers():
    R = J['replay']['2025-11-01']; nc = R['nowcast']['15']
    has(f"{c(nc['16'])}대·{c(nc['17'])}대", f"({c(R['actual'][16])}대·{c(R['actual'][17])}대)", f"{c(nc['18'])}대로 실측 {c(R['actual'][18])}대보다 {c(R['actual'][18] - nc['18'])}대")

def test_secondary_facts():
    import model2025 as M
    V = np.where(M.VALID)[0]; wk = M.dtype[V] == 2; ev = M.ev_day[V]; P = M.OCC[V].max(1)
    has(f'평균 {c(P[wk & ev].mean())}대로 주말 비공연일({c(P[wk & ~ev].mean())}대)')
    meta = json.load(open(ROOT / 'data/raw/y2025/meta.json'))
    h = np.cumsum(meta['stay_min_hist_by_group'][0]); h = h / h[-1]
    has(f'중앙값은 {int(np.searchsorted(h, .5))}분, 90분위는 {int(np.searchsorted(h, .9))}분', f"{c(meta['noExit'][0])}건은 제외")
    off = []
    for i in V:
        ss = [s for v, s in M.by_day.get(M.days[i].isoformat(), []) if v == 'KSPO DOME']
        if ss: off.append(next((h - min(ss) for h in range(24) if M.OCC[i, h] >= 1540), None))
    r = [x for x in off if x is not None]
    has(f'68일 중 {len(r)}일이 혼잡(70%)에 이르렀고, 그 시각은 공연 시작 {int(-np.median(r))}시간 전')

def test_live_model_holdout():
    L = json.load(open(ROOT / 'model/live_eval.json'))['holdout_live_model']
    has(f"홀드아웃 MAE {f1(L['nowcast_live'])}대, MAPE {f1(L['mape_nowcast_live'])}%")
    assert L['nowcast_live'] < L['persist']

def test_strong_baseline_and_dayahead():
    ho, cv = M25['holdout_q4'], M25['month_cv']; S = json.load(open(ROOT / 'model/scen_eval.json'))
    k = 'event_shape_persist'
    has(f"'{f1(cv['mae_event_evening'][k])}', '{f1(ho['mae_event_evening'][k])}', '{f1(ho['mape_event_evening'][k])}%'")
    has(f"({f1(ho['mae_event_evening'][k])}대)보다도 {round(100 * (1 - ho['mae_event_evening']['nowcast'] / ho['mae_event_evening'][k]))}% 작았습니다")
    has(f"가장 강한 기준모델 대비 {ho['improve_ci95_vs_event_shape'][0]}~{ho['improve_ci95_vs_event_shape'][2]}대")
    assert ho['improve_ci95_vs_event_shape'][0] > 0 and cv['improve_ci95_vs_event_shape'][0] > 0
    has(f"공연일 {S['test_event_days']}일에 적용", f"({S['busy70']['actual_days']}·{S['full90']['actual_days']}·{S['over']['actual_days']}일)",
        f"(오경보 {S['busy70']['false_alarm']}·{S['full90']['false_alarm']}·{S['over']['false_alarm']}일)", f"비율은 {round(100 * S['band_coverage_9_22'])}%", f"오차는 {c(S['peak_mae_mid'])}대")
    for k2 in ('busy70', 'full90', 'over'): assert S[k2]['hit'] == S[k2]['actual_days']
