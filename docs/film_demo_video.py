"""데모 시연 영상 프레임 촬영. 출력: frames/<scene>/NNN.jpg + scenes.json(장면별 [파일, 초])"""
import asyncio, json, base64, sys
from pathlib import Path
from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).parent / 'frames'; OUT.mkdir(exist_ok=True)
DEMO = (ROOT / 'index.html').as_uri()
FIG = ROOT / 'docs/fig'
SC = {}

CAP_JS = """(t) => { let d = document.getElementById('__cap'); if (!d) { d = document.createElement('div'); d.id = '__cap';
 d.style.cssText = 'position:fixed;left:0;right:0;top:0;height:78px;display:flex;align-items:center;justify-content:center;background:#1c1c1a;color:#fff;font:600 20px/1.45 \\"Noto Sans CJK KR\\",sans-serif;padding:0 40px;z-index:99999;text-align:center;letter-spacing:-.2px';
 document.body.appendChild(d); document.body.style.paddingTop = '78px'; } d.innerHTML = t; }"""

CARD = """<html><body style="margin:0;width:1280px;height:720px;display:flex;flex-direction:column;justify-content:center;align-items:center;background:#f5f4f0;font-family:'Noto Sans CJK KR';color:#141413">{}</body></html>"""

class Rec:
    def __init__(s, pg): s.pg = pg
    async def shot(s, scene, sec):
        lst = SC.setdefault(scene, []); f = OUT / scene / f'{len(lst):03d}.jpg'; f.parent.mkdir(exist_ok=True)
        await s.pg.screenshot(path=str(f), type='jpeg', quality=92); lst.append([str(f), sec])
    async def cap(s, t): await s.pg.evaluate(CAP_JS, t)
    async def to(s, sel, off=100):
        await s.pg.evaluate(f"window.scrollTo(0, document.querySelector('{sel}').getBoundingClientRect().top + scrollY - {off})"); await s.pg.wait_for_timeout(150)
    async def tab(s, t):
        await s.pg.evaluate(f"location.hash='{t}'"); await s.pg.wait_for_timeout(250); await s.pg.evaluate('window.scrollTo(0,0)')

def img_card(path, title, note):
    b64 = base64.b64encode(Path(path).read_bytes()).decode()
    return CARD.format(f'<div style="font-size:30px;font-weight:700;margin-bottom:16px">{title}</div>'
                       f'<img src="data:image/png;base64,{b64}" style="max-width:1120px;max-height:470px;background:#fff;border-radius:10px;padding:8px">'
                       f'<div style="font-size:19px;color:#52514e;margin-top:16px">{note}</div>')

async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(args=['--no-sandbox'])
        pg = await b.new_page(viewport={'width': 1280, 'height': 720}, device_scale_factor=1.5)
        r = Rec(pg)
        # 1 타이틀
        await pg.set_content(CARD.format('<div style="font-size:64px;font-weight:800">올팍 <span style="color:#2a78d6">나우</span></div>'
            '<div style="font-size:26px;margin-top:14px">실측 입출차 데이터로 현장 상황을 보정하는<br>올림픽공원 공연일 주차 운영 의사결정 지원 시스템</div>'
            '<div style="font-size:18px;color:#52514e;margin-top:28px">한국체육산업개발(주) 공공데이터 · 2025년 입출차 2,411,225건 전수 분석</div>'
            '<div style="font-size:16px;color:#8a8984;margin-top:40px">2026년도 공공데이터 활용 경진대회 · 박부건</div>'), wait_until='load')
        await r.shot('s1', 4.5)
        # 2 문제
        await pg.set_content(img_card(FIG / 'g1_daily_peak_2025.png', '문제: 만차는 공연이 만들고, 수용을 넘는 날도 있습니다',
            '2025년 만차(90%) 38일은 모두 공연일 · 28일(117시간)은 공개 수용 2,200대 초과 · 최대 3,563대'), wait_until='load')
        await r.shot('s2', 7)
        await pg.set_content(img_card(FIG / 'g2_dome_excess_exit.png', '같은 KSPO DOME 공연도 주차 수요가 크게 다릅니다',
            '초과 주차 517~1,924대(68일, 10~90%) → 고정 예측의 한계 · 출차는 공연 시작 2~3시간 뒤 몰림(68일 중 56일)'), wait_until='load')
        await r.shot('s2', 7)
        # 3 운영 대시보드
        await pg.goto(DEMO); await pg.wait_for_timeout(500); await pg.evaluate('window.scrollTo(0,0)')
        await r.cap('운영 대시보드: 날짜를 고르면 그날 공연과 함께<br>혼잡·만차 위험 시각, 출차 인력 배치 시각, 수용 초과 대수를 바로 보여 줍니다')
        await r.shot('s3', 7)
        await r.to('#c0', 150); await r.cap('예상 주차 차량 수(파란 선)와 범위(하늘색) — 2025년 KSPO DOME 공연일 68일 관측 분포로 만든 범위')
        await r.shot('s3', 6)
        await r.to('#opsTbl', 110); await r.cap('시간대별 경보와 권장 조치 — 사전 안내, 구역 유도 표지, 진입 통제 준비, 출차 유도 인력')
        await r.shot('s3', 6)
        await pg.evaluate("window.scrollBy(0, 300)"); await pg.wait_for_timeout(150); await r.cap('출차가 몰리는 공연 시작 +2~3시간에는 출차 유도 인력 배치와 시차 귀가 안내'); await r.shot('s3', 5)
        # 4 다른 날짜
        await pg.evaluate("window.scrollTo(0,0)"); await pg.select_option('#date', '2026-12-05'); await pg.wait_for_timeout(300)
        await r.cap('공연이 여러 곳에서 겹치는 날을 고르면 위험 시각과 조치가 그날에 맞게 바뀝니다'); await r.shot('s4', 6)
        # 5 나우캐스트 재현
        await r.tab('replay'); await r.cap('나우캐스트 검증: 2025년 1~7월 데이터로만 학습한 모델로 11월 1일 공연일을 예측합니다')
        await pg.fill('#h0', '10'); await pg.dispatch_event('#h0', 'input'); await pg.wait_for_timeout(200); await r.shot('s5', 5)
        await r.cap('현재 시각을 옮기면, 그때까지의 실측으로 1~3시간 뒤(주황 점)를 다시 예측합니다')
        for h in range(11, 19):
            await pg.fill('#h0', str(h)); await pg.dispatch_event('#h0', 'input'); await pg.wait_for_timeout(120); await r.shot('s5', 1.1)
        await pg.fill('#h0', '15'); await pg.dispatch_event('#h0', 'input'); await pg.wait_for_timeout(150)
        await r.cap('15시 기준: 실측으로 보정한 예측이 평소 패턴(회색 점선)보다 훨씬 정확합니다'); await r.shot('s5', 6)
        # 6 검증
        await r.tab('valid'); await r.cap('10~12월 홀드아웃: 오차 179.0대(MAPE 14.5%) — 현재값 유지(429.9대)보다 58% 작음 · 혼잡 경보 정밀도 0.905')
        await r.shot('s6', 8)
        await r.to('#qual', 140); await r.cap('241만 건을 전수 집계하며 찾은 데이터 품질 개선점도 공개합니다'); await r.shot('s6', 6)
        # 7 관람객 안내 + 영문
        await r.tab('plan'); await r.cap('같은 예측을 관람객 안내로도 내보냅니다 — 도착 권장 시각, 구역, 대중교통, 시차 귀가'); await r.shot('s7', 6)
        await pg.click('#lang'); await pg.wait_for_timeout(300); await r.cap('English mode for international visitors'); await r.shot('s7', 4)
        await pg.click('#lang'); await pg.wait_for_timeout(200)
        # 8 엔딩
        await pg.set_content(CARD.format('<div style="font-size:44px;font-weight:800">올팍 <span style="color:#2a78d6">나우</span></div>'
            '<div style="font-size:24px;margin-top:12px">예측 → 당일 보정 → 운영 결정</div>'
            '<div style="font-size:20px;margin-top:30px;line-height:1.8">서비스 bugeon151.github.io/olpark-now<br>소스 github.com/Bugeon151/olpark-now</div>'
            '<div style="font-size:16px;color:#8a8984;margin-top:34px">데이터: 한국체육산업개발(주) 공공데이터 · Open-Meteo</div>'), wait_until='load')
        await r.shot('s8', 5)
        await b.close()
    json.dump(SC, open(Path(__file__).parent / 'scenes.json', 'w'), ensure_ascii=False, indent=0)
    print({k: round(sum(x[1] for x in v), 1) for k, v in SC.items()}, round(sum(x[1] for v in SC.values() for x in v), 1))
asyncio.run(main())
