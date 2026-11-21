"""model.json + 구역통계 + 검증강화 결과를 합쳐 index.html 생성."""
import json
from pathlib import Path
R = Path(__file__).resolve().parents[1]
m = json.load(open(R / 'demo/model.json'))
m['zones'] = json.load(open(R / 'data/derived/zone_stats.json'))
m['robust'] = json.load(open(R / 'model/robust.json'))
t = (R / 'demo/template.html').read_text()
assert '/*DATA*/null' in t
(R / 'demo/index.html').write_text(t.replace('/*DATA*/null', json.dumps(m, ensure_ascii=False, separators=(',', ':'))))
print('ok', (R / 'demo/index.html').stat().st_size)
