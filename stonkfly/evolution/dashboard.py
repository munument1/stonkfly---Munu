from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _finite(values):
    return [float(value) for value in values if math.isfinite(float(value))]


def _generation_summary(payload):
    ranked = payload.get("ranked") or []
    scores = _finite(row["fitness"]["score"] for row in ranked)
    returns = _finite(row["fitness"]["return_pct"] for row in ranked)
    drawdowns = _finite(row["fitness"]["max_drawdown_pct"] for row in ranked)
    seconds = _finite(row.get("evaluation_seconds", 0.0) for row in ranked)
    if not scores:
        return None
    champion = max(ranked, key=lambda row: row["fitness"]["score"])
    return {
        "generation": int(payload["generation"]),
        "best_score": max(scores),
        "mean_score": statistics.fmean(scores),
        "median_score": statistics.median(scores),
        "worst_score": min(scores),
        "best_return_pct": max(returns) if returns else 0.0,
        "mean_return_pct": statistics.fmean(returns) if returns else 0.0,
        "best_drawdown_pct": champion["fitness"]["max_drawdown_pct"],
        "mean_drawdown_pct": statistics.fmean(drawdowns) if drawdowns else 0.0,
        "evaluation_seconds": sum(seconds),
        "population": len(ranked),
        "champion": champion["fingerprint"],
        "genome": champion.get("genome", {}),
    }


def _downsample(rows, limit):
    if len(rows) <= limit:
        return rows
    indexes = {0, len(rows) - 1}
    step = (len(rows) - 1) / (limit - 1)
    indexes.update(round(index * step) for index in range(1, limit - 1))
    return [rows[index] for index in sorted(indexes)]


def _mode_directories(run: Path):
    if (run / "config.json").is_file():
        config = _read_json(run / "config.json") or {}
        return [(config.get("inheritance", "evolution"), run)]
    result = []
    for mode in ("darwinian", "lamarckian"):
        path = run / mode
        if (path / "config.json").is_file():
            result.append((mode, path))
    return result


def build_snapshot(run: Path, max_points: int = 2500):
    modes = {}
    for mode, path in _mode_directories(run):
        config = _read_json(path / "config.json") or {}
        progress = _read_json(path / "progress.json") or {}
        summaries = []
        latest = []
        for generation_path in sorted(path.glob("generation-*.json")):
            payload = _read_json(generation_path)
            if not isinstance(payload, dict):
                continue
            try:
                summary = _generation_summary(payload)
            except (KeyError, TypeError, ValueError):
                continue
            if summary is not None:
                summaries.append(summary)
                latest = payload.get("ranked", [])[:20]
        champion = _read_json(path / "champion.json")
        if not progress:
            progress = {
                "status": "completed" if champion else "waiting",
                "completed_generations": len(summaries),
                "total_generations": config.get("generations", 0),
            }
        modes[mode] = {
            "config": config,
            "progress": progress,
            "series": _downsample(summaries, max_points),
            "recorded_generations": len(summaries),
            "latest_ranked": latest,
            "champion": champion,
        }
    return {
        "run": str(run.resolve()),
        "server_time_unix": time.time(),
        "modes": modes,
    }


HTML = r'''<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stonkfly Evolution Monitor</title>
<style>
:root{color-scheme:light dark;--bg:#f5f7fb;--surface:#fff;--text:#172033;--muted:#667085;--border:#d9dfeb;--blue:#2563eb;--orange:#ea580c;--green:#0f9d71;--red:#d92d20}@media(prefers-color-scheme:dark){:root{--bg:#0d111b;--surface:#161c29;--text:#e6eaf2;--muted:#9aa4b5;--border:#2b3445;--blue:#6ea8fe;--orange:#ff9b61;--green:#55d6a8;--red:#ff7b72}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,sans-serif}.shell{max-width:1500px;margin:auto;padding:24px}.top{display:flex;justify-content:space-between;align-items:flex-start;gap:16px;flex-wrap:wrap}h1{font-size:24px;margin:0 0 4px}h2{font-size:16px;margin:0 0 12px}.muted{color:var(--muted)}.modes{display:flex;gap:8px}button{font:inherit;border:1px solid var(--border);background:var(--surface);color:var(--text);border-radius:8px;padding:8px 12px;cursor:pointer}button.active{background:var(--blue);color:#fff;border-color:var(--blue)}.grid{display:grid;grid-template-columns:repeat(4,minmax(150px,1fr));gap:12px;margin:20px 0}.card,.panel{background:var(--surface);border:1px solid var(--border);border-radius:12px}.card{padding:14px}.label{color:var(--muted);font-size:12px}.value{font-size:24px;font-weight:650;margin-top:4px;font-variant-numeric:tabular-nums}.progress{height:8px;background:var(--border);border-radius:99px;overflow:hidden;margin-top:10px}.bar{height:100%;background:var(--green);transition:width .3s}.charts{display:grid;grid-template-columns:2fr 1fr;gap:12px}.panel{padding:16px;margin-bottom:12px}canvas{display:block;width:100%;height:300px}.table-wrap{overflow:auto;max-height:420px}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{text-align:right;padding:8px;border-bottom:1px solid var(--border);white-space:nowrap}th:first-child,td:first-child{text-align:left}th{position:sticky;top:0;background:var(--surface);color:var(--muted)}.error{color:var(--red)}@media(max-width:850px){.grid{grid-template-columns:repeat(2,1fr)}.charts{grid-template-columns:1fr}}@media(max-width:480px){.shell{padding:14px}.grid{grid-template-columns:1fr}}
</style></head><body><main class="shell">
<div class="top"><div><h1>Stonkfly Evolution Monitor</h1><div id="path" class="muted"></div></div><div><div id="modes" class="modes"></div><div id="updated" class="muted" style="margin-top:6px;text-align:right"></div></div></div>
<section class="grid"><div class="card"><div class="label">진행 세대</div><div id="generation" class="value">—</div><div class="progress"><div id="bar" class="bar"></div></div></div><div class="card"><div class="label">최고 Fitness</div><div id="fitness" class="value">—</div></div><div class="card"><div class="label">최고 수익률</div><div id="returns" class="value">—</div></div><div class="card"><div class="label">상태</div><div id="status" class="value">대기</div><div id="individual" class="muted"></div></div></section>
<section class="charts"><div class="panel"><h2>세대별 Fitness</h2><canvas id="fitnessChart" aria-label="세대별 최고, 평균, 최저 fitness"></canvas></div><div class="panel"><h2>수익률과 낙폭</h2><canvas id="riskChart" aria-label="세대별 최고 수익률과 챔피언 낙폭"></canvas></div></section>
<section class="panel"><h2>최근 완료 세대 상위 개체</h2><div class="table-wrap"><table><thead><tr><th>개체</th><th>Fitness</th><th>수익률</th><th>최대 낙폭</th><th>거래</th><th>평가 시간</th></tr></thead><tbody id="rows"></tbody></table></div></section>
<div id="error" class="error" role="alert"></div></main>
<script>
let snapshot=null,selected=null;const $=id=>document.getElementById(id),fmt=(v,n=3)=>Number.isFinite(+v)?(+v).toFixed(n):'—';function color(n){return getComputedStyle(document.documentElement).getPropertyValue(n).trim()}
function draw(canvas,series){const ratio=devicePixelRatio||1,w=canvas.clientWidth,h=canvas.clientHeight;canvas.width=w*ratio;canvas.height=h*ratio;const c=canvas.getContext('2d');c.scale(ratio,ratio);c.clearRect(0,0,w,h);if(!series.length||!series[0].values.length){c.fillStyle=color('--muted');c.fillText('완료된 세대를 기다리는 중입니다.',16,28);return}const pad={l:58,r:18,t:16,b:34},pw=w-pad.l-pad.r,ph=h-pad.t-pad.b,values=series.flatMap(s=>s.values).filter(Number.isFinite);let min=Math.min(...values),max=Math.max(...values);if(min===max){min-=1;max+=1}const x=i=>pad.l+(series[0].values.length===1?0:pw*i/(series[0].values.length-1)),y=v=>pad.t+ph-(v-min)/(max-min)*ph;c.strokeStyle=color('--border');c.lineWidth=1;c.strokeRect(pad.l,pad.t,pw,ph);c.fillStyle=color('--muted');c.font='12px system-ui';c.textAlign='right';for(let i=0;i<5;i++){const v=min+(max-min)*i/4,yy=y(v);c.fillText(fmt(v,2),pad.l-8,yy+4);c.beginPath();c.moveTo(pad.l,yy);c.lineTo(w-pad.r,yy);c.globalAlpha=.35;c.stroke();c.globalAlpha=1}series.forEach(s=>{c.strokeStyle=s.color;c.lineWidth=s.width||2;c.beginPath();s.values.forEach((v,i)=>{if(Number.isFinite(v))(i?c.lineTo(x(i),y(v)):c.moveTo(x(i),y(v)))});c.stroke()});c.textAlign='left';c.fillStyle=color('--muted');c.fillText('세대 '+series[0].gens[0],pad.l,h-10);c.textAlign='right';c.fillText('세대 '+series[0].gens.at(-1),w-pad.r,h-10)}
function render(){const mode=snapshot?.modes?.[selected];if(!mode)return;const p=mode.progress||{},s=mode.series||[],last=s.at(-1),total=p.total_generations||mode.config.generations||0,done=p.completed_generations??mode.recorded_generations;$('path').textContent=snapshot.run;$('generation').textContent=done.toLocaleString()+' / '+total.toLocaleString();$('bar').style.width=total?Math.min(100,100*done/total)+'%':'0%';$('fitness').textContent=last?fmt(Math.max(...s.map(x=>x.best_score)),4):'—';$('returns').textContent=last?fmt(Math.max(...s.map(x=>x.best_return_pct)),3)+'%':'—';$('status').textContent=({running:'학습 중',completed:'완료',waiting:'대기'})[p.status]||p.status||'대기';$('individual').textContent=p.current_individual==null?'':'현재 개체 '+(p.current_individual+1)+' / '+p.total_individuals;draw($('fitnessChart'),[{gens:s.map(x=>x.generation),values:s.map(x=>x.best_score),color:color('--blue')},{gens:s.map(x=>x.generation),values:s.map(x=>x.mean_score),color:color('--green')},{gens:s.map(x=>x.generation),values:s.map(x=>x.worst_score),color:color('--orange'),width:1}]);draw($('riskChart'),[{gens:s.map(x=>x.generation),values:s.map(x=>x.best_return_pct),color:color('--green')},{gens:s.map(x=>x.generation),values:s.map(x=>-x.best_drawdown_pct),color:color('--red')}]);$('rows').innerHTML=mode.latest_ranked.map(r=>'<tr><td>'+r.fingerprint+'</td><td>'+fmt(r.fitness.score,4)+'</td><td>'+fmt(r.fitness.return_pct,3)+'%</td><td>'+fmt(r.fitness.max_drawdown_pct,3)+'%</td><td>'+r.fitness.trade_count+'</td><td>'+fmt(r.evaluation_seconds,1)+'초</td></tr>').join('')}
async function refresh(){try{const r=await fetch('/api/snapshot',{cache:'no-store'});if(!r.ok)throw Error(r.statusText);snapshot=await r.json();const names=Object.keys(snapshot.modes);if(!selected||!snapshot.modes[selected])selected=names[0];$('modes').innerHTML=names.map(n=>'<button class="'+(n===selected?'active':'')+'" data-mode="'+n+'">'+n+'</button>').join('');document.querySelectorAll('[data-mode]').forEach(b=>b.onclick=()=>{selected=b.dataset.mode;render();refreshButtons()});$('updated').textContent='자동 갱신 · '+new Date().toLocaleTimeString();$('error').textContent='';render()}catch(e){$('error').textContent='데이터를 읽지 못했습니다: '+e.message}}
function refreshButtons(){document.querySelectorAll('[data-mode]').forEach(b=>b.classList.toggle('active',b.dataset.mode===selected))}refresh();setInterval(refresh,2500);addEventListener('resize',()=>snapshot&&render());matchMedia('(prefers-color-scheme: dark)').addEventListener('change',()=>snapshot&&render());
</script></body></html>'''


def _handler(run: Path, max_points: int):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                body = HTML.encode()
                content_type = "text/html; charset=utf-8"
            elif path == "/api/snapshot":
                body = json.dumps(build_snapshot(run, max_points)).encode()
                content_type = "application/json; charset=utf-8"
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            return

    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description="Live monitor for Stonkfly evolution runs")
    parser.add_argument("--run", type=Path, default=Path("runs/evolution"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--max-points", type=int, default=2500)
    parser.add_argument("--open", action="store_true", help="Open the monitor in a browser")
    args = parser.parse_args(argv)
    if not 100 <= args.max_points <= 10000:
        parser.error("--max-points must be between 100 and 10000")
    server = ThreadingHTTPServer((args.host, args.port), _handler(args.run, args.max_points))
    url = f"http://{args.host}:{server.server_port}/"
    print(f"Monitoring {args.run.resolve()} at {url}", flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
