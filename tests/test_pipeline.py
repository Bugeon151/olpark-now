"""데이터 무결성·지표 재현성 테스트. 실행: python3 -m pytest -q tests"""
import json, sys, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'model'))
from train import ARR, DEP, OCC, TEMP, RAIN, EV, D, H

def test_shapes():
    for a in (ARR, DEP, OCC, TEMP, RAIN):
        assert a.shape == (31, 24)

def test_total_records_match_source():
    # 원본 215,528건 = 입차 합계. 출차는 8월로 넘어간 차량(7/31 23:30 재차 65대)만큼 7월 집계에서 빠짐
    assert ARR.sum() == 215528
    assert DEP.sum() + OCC[30, 23] == 215528

def test_stock_flow_consistency():
    # 재차대수 변화(hh:30→다음 hh:30)는 그 사이 입차-출차와 대략 일치해야 함(시간 경계 차이 허용)
    occ = OCC.reshape(-1); net = (ARR - DEP).reshape(-1)
    diff = occ[1:] - occ[:-1]
    approx = (net[:-1] + net[1:]) / 2
    assert np.corrcoef(diff, approx)[0, 1] > 0.9

def test_capacity_and_peaks():
    assert EV['lots_main_capacity'] == 2200
    assert int(OCC[19].max()) == 2013 and int(OCC[20].max()) == 2105

def test_events_inside_july_and_known_venues():
    for d, v, s, _ in EV['sessions_2024_07']:
        assert d.startswith('2024-07') and v in EV['venues'] and 0 <= s < 24

def test_metrics_reproducible():
    M = json.load(open(ROOT / 'demo/model.json'))['metrics']
    assert M['gbm']['mae_event_evening'] == 147.1
    assert M['profile']['mae_event_evening'] == 332.5
    assert M['persist']['mae_event_evening'] == 301.4
    assert M['gbm']['mae_event_evening'] < min(M['profile']['mae_event_evening'], M['persist']['mae_event_evening'])

def test_robust_ci_excludes_zero():
    R = json.load(open(ROOT / 'model/robust.json'))
    assert R['improve_vs_profile_CI95'][0] > 0 and R['improve_vs_persist_CI95'][0] > 0

def test_exit_peak_claim():
    dome = {12: 19, 13: 17, 14: 16, 20: 18, 21: 17, 27: 18, 28: 16}
    rel = [int(np.argmax([DEP[d - 1, s + r] for r in range(6) if s + r < 24])) for d, s in dome.items()]
    assert rel.count(3) == 5 and all(abs(r - 3) <= 1 for r in rel)

def test_zone_stats_quality_note():
    Z = json.load(open(ROOT / 'data/derived/zone_stats.json'))
    assert sum(z['records'] for z in Z['zones'].values()) == 699
