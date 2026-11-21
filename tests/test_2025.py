"""2025 입출차 전수 집계(2,411,225건) 무결성·지표 재현 테스트."""
import json, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
Y = ROOT / 'data/raw/y2025'
H = np.loadtxt(Y / 'hourly.csv', delimiter=',', skiprows=1)
META = json.load(open(Y / 'meta.json'))

def test_all_records_collected_once():
    assert META['items'] == META['total'] == 2411225 and META['dup'] == 0 and META['badParse'] == 0
    assert H[:, [1, 4, 7, 10]].sum() == 2411225
    assert [int(H[:, c].sum()) for c in (1, 4, 7, 10)] == META['arr_total_by_group']

def test_departures_plus_open_stays_balance():
    # 그룹별 입차 = 집계창 안 출차 + 집계창 끝 재차 + 출차시각 없는 기록(재차 계산에서 제외)
    for g in range(4):
        assert H[:, 1 + 3 * g].sum() == H[:, 2 + 3 * g].sum() + H[-1, 3 + 3 * g] + META['noExit'][g]

def test_source_coverage_gaps_are_handled():
    import sys; sys.path.insert(0, str(ROOT / 'model')); import model2025 as M
    day = H[:8760, [1, 4, 7, 10]].sum(1).reshape(365, 24).sum(1)
    last = [i for i, d in enumerate(M.days) if (d.month != (d + __import__('datetime').timedelta(1)).month)]
    assert (day[last] == 0).all()                      # 원천: 매월 말일 미수록
    assert not M.VALID[last].any() and not M.VALID[M.month == 8].any() and M.VALID.sum() == 247

def test_occupancy_matches_agency_linked_parking_count():
    L = pd.read_csv(Y / 'linked_2025.csv'); L['dt'] = pd.to_datetime(L['dt'])
    idx = ((L.dt - pd.Timestamp('2025-01-01')) / pd.Timedelta(hours=1)).astype(int).values
    assert np.corrcoef(H[idx, 3], L['주차대수'])[0, 1] > 0.95

def test_2025_metrics_reproducible():
    m = json.load(open(ROOT / 'model/metrics2025.json'))
    cv, ho = m['month_cv'], m['holdout_q4']
    assert cv['mae_event_evening']['nowcast'] == 162.1 and ho['mae_event_evening']['nowcast'] == 179.0
    for o in (cv, ho):
        assert o['mae_event_evening']['nowcast'] < min(o['mae_event_evening'][k] for k in ('profile', 'persist', 'seasonal_naive', 'pre_event_gbm'))
        assert o['improve_ci95_vs_persist'][0] > 0 and o['improve_ci95_vs_profile'][0] > 0

def test_demo_uses_2025_results():
    m = json.load(open(ROOT / 'demo/model.json'))
    assert m['metrics2025'] == json.load(open(ROOT / 'model/metrics2025.json'))
    assert set(m['profile_m']) == {str(i) for i in range(1, 13)} and m['scen_from'] == -10
    sc = m['dome_excess_scenarios_rel_start_-6_to_+5']
    assert all(lo <= md <= hi for lo, md, hi in zip(sc['low'], sc['mid'], sc['high']))
    assert m['dep_peak_rel_counts']['2'] + m['dep_peak_rel_counts']['3'] == 56 and m['dome_days'] == 68
