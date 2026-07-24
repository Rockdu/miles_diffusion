"""Render the JSONL from miles.dashboard.hooks into a self-contained HTML dashboard.

Usage: python -m miles.dashboard.viewer --workspace <path> --out dash.html
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

from miles.dashboard.events import SPAN_KINDS


def _read_jsonl(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return rows


def _fold_trajectory(events):
    """Pair start/end TrajectoryEvents back into closed {stage, t0, t1} segments."""
    by_key = defaultdict(list)
    for e in events:
        by_key[(e.get("rollout_id", 0), e.get("sample_index", 0))].append(e)
    segments = []
    for (rid, sidx), evs in by_key.items():
        evs.sort(key=lambda e: e.get("ts", 0))
        open_t0 = {}
        for e in evs:
            fold = SPAN_KINDS.get(e.get("kind"))
            if fold is None:
                continue
            base, is_start = fold
            if is_start:
                open_t0[base] = e.get("ts")
            else:
                t0 = open_t0.pop(base, None)
                if t0 is not None:
                    segments.append(
                        {"rollout_id": rid, "sample_index": sidx, "stage": base, "t0": t0, "t1": e.get("ts")}
                    )
    return segments


def load_streams(workspace: str):
    phases = _read_jsonl(sorted(glob.glob(os.path.join(workspace, "phases", "*.jsonl"))))
    gpu = _read_jsonl(sorted(glob.glob(os.path.join(workspace, "gpu_util", "*.jsonl"))))
    traj = _read_jsonl(sorted(glob.glob(os.path.join(workspace, "trajectories", "*.jsonl"))))
    return phases, gpu, _fold_trajectory(traj)


def _compute_data(phases, gpu, life=None) -> dict:
    life = life or []
    # normalize all timestamps to the earliest event across streams
    ts_candidates = [p["t0"] for p in phases] + [g["ts"] for g in gpu] + [x["t0"] for x in life]
    t0 = min(ts_candidates) if ts_candidates else 0.0
    ph = [
        {"name": p.get("name", "?"), "role": p.get("role", "?"), "s": round(p["t0"] - t0, 3), "e": round(p["t1"] - t0, 3)}
        for p in phases
        if p.get("t1", 0) >= p.get("t0", 0)
    ]
    ph.sort(key=lambda r: r["s"])
    gp = [
        {
            "g": f"{g.get('host', '')}:{g.get('gpu', 0)}",
            "t": round(g["ts"] - t0, 3),
            "u": g.get("util", 0),
            "m": g.get("mem_mb", 0),
            "p": g.get("power_w", 0),
        }
        for g in gpu
    ]
    gp.sort(key=lambda r: r["t"])
    lf = [
        {
            "k": f"{x.get('rollout_id', 0)}:{x.get('sample_index', 0)}",
            "stage": x.get("stage", "?"),
            "s": round(x["t0"] - t0, 3),
            "e": round(x["t1"] - t0, 3),
        }
        for x in life
        if x.get("t1", 0) >= x.get("t0", 0)
    ]
    span = max([r["e"] for r in ph] + [r["t"] for r in gp] + [r["e"] for r in lf] + [1.0])
    return {"phases": ph, "gpu": gp, "life": lf, "span": span}


def build_html(phases, gpu, life=None, title="miles-D rollout dashboard") -> str:
    data = json.dumps(_compute_data(phases, gpu, life), separators=(",", ":"))
    return _TEMPLATE.replace("__TITLE__", title).replace("__SERVE__", "false").replace("__DATA__", data)


_TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>__TITLE__</title>
<style>
@import url("https://fonts.googleapis.com/css2?family=Hanken+Grotesk:wght@600;700&family=Inter:wght@400;500;600&display=swap");
:root{
  --bg:#faf7f4;--panel:#fefcfa;--border:#e8e1d8;--text:#231f1c;--muted:#7a7168;
  --accent:#d55816;--accent-light:#e8722a;--topbar-bg:#0c0b0a;--topbar-border:#242220;
  --topbar-text:#efe9e1;--topbar-muted:#a99e91;
  --font-heading:"Hanken Grotesk",-apple-system,"Segoe UI",sans-serif;
  --font-body:"Inter",-apple-system,"Segoe UI",sans-serif;
  font-size:14px;
}
*{box-sizing:border-box;}
body{margin:0;background:var(--bg);color:var(--text);font-family:var(--font-body);}
#topbar{display:flex;align-items:center;gap:22px;padding:0 18px;background:var(--topbar-bg);border-bottom:1px solid var(--topbar-border);position:sticky;top:0;z-index:10;min-height:46px;}
#topbar .brand{font-family:var(--font-heading);font-weight:700;font-size:16px;color:var(--topbar-text);}
#topbar .brand b{color:var(--accent-light);}
#topbar .run{font-family:var(--font-heading);font-weight:600;font-size:15px;color:var(--topbar-text);}
#topbar #runinfo{margin-left:auto;color:var(--topbar-muted);font-size:12px;font-family:ui-monospace,monospace;}
#view{padding:16px 18px 60px;max-width:1400px;margin:0 auto;}
.controls{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:12px;}
button{background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:4px;padding:4px 10px;cursor:pointer;font:inherit;font-size:13px;}
button:hover{border-color:var(--accent);}
.hint{color:var(--muted);font-size:12px;}
.panel{background:var(--panel);border:1px solid var(--border);border-radius:6px;padding:12px;margin-bottom:14px;}
.panel h3{margin:0 0 8px;font-family:var(--font-heading);font-size:13px;font-weight:600;color:var(--muted);}
.legend{display:flex;align-items:center;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--muted);margin-bottom:8px;}
.legend .k{display:inline-flex;gap:5px;align-items:center;}
.legend .sw{width:12px;height:12px;border-radius:3px;display:inline-block;}
svg{width:100%;display:block;}
.axlab{fill:var(--muted);font:11px ui-monospace,monospace;}
.grid{stroke:var(--border);stroke-width:1;}
.rowlab{fill:var(--text);font:11px ui-monospace,monospace;}
.roundline{stroke:var(--accent);stroke-width:1;stroke-dasharray:3 3;opacity:.5;}
#tooltip{position:fixed;pointer-events:none;background:var(--topbar-bg);color:var(--topbar-text);border:1px solid var(--topbar-border);border-radius:4px;padding:6px 9px;font-size:12px;z-index:100;opacity:0;font-family:ui-monospace,monospace;white-space:pre;line-height:1.5;}
</style></head><body>
<header id="topbar">
  <span class="brand">miles<b>·</b>D</span>
  <span class="run">rollout dashboard</span>
  <span id="runinfo"></span>
</header>
<main id="view">
  <div class="controls">
    <button id="reset">Reset zoom</button>
    <span class="hint">wheel = zoom · drag = pan · hover = details</span>
    <span id="rounds"></span>
  </div>
  <div class="panel"><h3>GPU utilization %</h3><svg id="gpu" role="img" aria-label="GPU utilization over time"></svg></div>
  <div class="panel"><h3>Phase timeline (cross-round)</h3><svg id="gantt" role="img" aria-label="Phase timeline"></svg></div>
  <div class="panel"><h3>Per-sample lifecycle</h3><div class="legend" id="lifelegend"></div><svg id="samples" role="img" aria-label="Per-sample lifecycle"></svg></div>
</main>
<div id="tooltip"></div>
<script>
const SERVE=__SERVE__;
const PAL=["#2a78d6","#d55816","#1baf7a","#e87ba4","#4a3aa7","#eda100","#e34948","#008300","#8a6d3b","#5c6570"];
const W=1200, GPU_H=190, PADL=140, PADR=16, ROW_H=16, GAP=3;
const tip=document.getElementById("tooltip");
let D=__DATA__;
let phaseNames=[], colorOf={}, gpuIds=[], lifeStages=[], stageColor={}, sampKeys=[], rounds=[], boundaryName="";
let x0=0, x1=1, userZoomed=false;

// (re)derive everything from the current D; called on load and on each live refresh
function rebuild(){
  phaseNames=[...new Set(D.phases.map(p=>p.name))];
  colorOf={}; phaseNames.forEach((n,i)=>colorOf[n]=PAL[i%PAL.length]);
  gpuIds=[...new Set(D.gpu.map(g=>g.g))].sort();
  lifeStages=[...new Set(D.life.map(x=>x.stage))];
  stageColor={}; lifeStages.forEach((s,i)=>stageColor[s]=PAL[i%PAL.length]);
  sampKeys=[...new Set(D.life.map(x=>x.k))];
  boundaryName = phaseNames.includes("rollout_offload") ? "rollout_offload"
    : phaseNames.includes("actor_train") ? "actor_train" : (phaseNames[0]||"");
  rounds = D.phases.filter(p=>p.name===boundaryName).map(p=>p.s).sort((a,b)=>a-b);
  if(!userZoomed){ x0=0; x1=D.span; } else if(x1>D.span){ x1=D.span; }
  const lg=document.getElementById("lifelegend");
  if(D.life.length){ lg.innerHTML=lifeStages.map(s=>`<span class="k"><span class="sw" style="background:${stageColor[s]}"></span>${s}</span>`).join(""); }
  else{ document.getElementById("samples").closest(".panel").style.display="none"; }
  document.getElementById("runinfo").textContent =
    `span ${D.span.toFixed(1)}s · ${D.phases.length} phases · ${gpuIds.length} GPUs · ${D.gpu.length} gpu samples · ${sampKeys.length} reqs`;
  const rb=document.getElementById("rounds"); rb.innerHTML="";
  if(rounds.length){
    rb.insertAdjacentHTML("beforeend",`<span class="hint">rounds (${boundaryName}): </span>`);
    rounds.forEach((r,i)=>{const b=document.createElement("button");b.textContent="R"+i;
      b.onclick=()=>{userZoomed=true;const nxt=rounds[i+1]??D.span;const pad=(nxt-r)*0.05;x0=Math.max(0,r-pad);x1=Math.min(D.span,nxt+pad);draw();};
      rb.appendChild(b);});
  }
}
function sx(t,w){return PADL + (t-x0)/(x1-x0)*(w-PADL-PADR);}
function invx(px,w){return x0 + (px-PADL)/(w-PADL-PADR)*(x1-x0);}

function ticks(){const n=8,step=niceStep((x1-x0)/n),out=[];for(let t=Math.ceil(x0/step)*step;t<=x1;t+=step)out.push(+t.toFixed(6));return out;}
function niceStep(r){const p=Math.pow(10,Math.floor(Math.log10(r)));const f=r/p;return (f<1.5?1:f<3?2:f<7?5:10)*p;}

function draw(){
  drawGpu(); drawGantt(); if(D.life.length)drawSamples();
}
function drawSamples(){
  const svg=document.getElementById("samples"),w=W;
  // per-step rows: within each rollout, row = sample_index - that rollout's min index;
  // different rollouts reuse the same rows (they never overlap in time, so no collision).
  const rollMin={};
  D.life.forEach(seg=>{const p=seg.k.split(":"),si=+p[1];if(rollMin[p[0]]===undefined||si<rollMin[p[0]])rollMin[p[0]]=si;});
  let nRows=1;
  D.life.forEach(seg=>{const p=seg.k.split(":"),r=+p[1]-rollMin[p[0]];if(r+1>nRows)nRows=r+1;});
  const SROW=3, SGAP=1, bandTop=18, h=bandTop+nRows*(SROW+SGAP)+30;
  svg.setAttribute("viewBox",`0 0 ${w} ${h}`);
  let s=axis(svg,w,h);
  s+=`<clipPath id="cs"><rect x="${PADL}" y="${bandTop}" width="${w-PADL-PADR}" height="${h-24-bandTop}"/></clipPath><g clip-path="url(#cs)">`;
  D.life.forEach(seg=>{
    const p=seg.k.split(":"),row=+p[1]-rollMin[p[0]];
    let xa=sx(seg.s,w),xb=sx(seg.e,w);
    if(xb<PADL||xa>w-PADR)return;
    xa=Math.max(xa,PADL);xb=Math.min(xb,w-PADR);
    s+=`<rect class="lf" x="${xa.toFixed(1)}" y="${bandTop+row*(SROW+SGAP)}" width="${Math.max(xb-xa,0.6).toFixed(1)}" height="${SROW}" fill="${stageColor[seg.stage]}" data-st="${seg.stage}" data-k="${seg.k}" data-s="${seg.s}" data-e="${seg.e}"/>`;
  });
  s+=`</g><text class="axlab" x="${PADL-8}" y="${bandTop+8}" text-anchor="end">${nRows}/step</text>`;
  svg.innerHTML=s;
  svg.querySelectorAll(".lf").forEach(el=>{
    el.addEventListener("mousemove",e=>{const a=+el.dataset.s,b=+el.dataset.e;
      tip.innerHTML=`${el.dataset.k} · ${el.dataset.st}<br>${(b-a).toFixed(2)}s (${a.toFixed(1)}→${b.toFixed(1)}s)`;
      tip.style.opacity=1;tip.style.left=(e.clientX+12)+"px";tip.style.top=(e.clientY+12)+"px";});
    el.addEventListener("mouseleave",()=>tip.style.opacity=0);
  });
}
function axis(svg,w,h){
  let s=`<line class="grid" x1="${PADL}" y1="${h-24}" x2="${w-PADR}" y2="${h-24}"/>`;
  for(const t of ticks()){const px=sx(t,w);s+=`<line class="grid" x1="${px}" y1="8" x2="${px}" y2="${h-24}"/>`+
    `<text class="axlab" x="${px}" y="${h-10}" text-anchor="middle">${t}s</text>`;}
  for(const r of rounds){const px=sx(r,w);if(px>=PADL&&px<=w-PADR)s+=`<line class="roundline" x1="${px}" y1="8" x2="${px}" y2="${h-24}"/>`;}
  return s;
}
function drawGpu(){
  const svg=document.getElementById("gpu"),w=W,h=GPU_H;svg.setAttribute("viewBox",`0 0 ${w} ${h}`);
  const plotH=h-24-8;
  let s=axis(svg,w,h);
  for(const yv of [0,50,100]){const py=8+plotH-yv/100*plotH;s+=`<text class="axlab" x="${PADL-6}" y="${py+3}" text-anchor="end">${yv}</text>`;}
  s+=`<clipPath id="cg"><rect x="${PADL}" y="0" width="${w-PADL-PADR}" height="${h}"/></clipPath><g clip-path="url(#cg)">`;
  gpuIds.forEach((gid,i)=>{
    const pts=D.gpu.filter(g=>g.g===gid).map(g=>`${sx(g.t,w).toFixed(1)},${(8+plotH-g.u/100*plotH).toFixed(1)}`);
    if(pts.length)s+=`<polyline fill="none" stroke="${PAL[i%PAL.length]}" stroke-width="1.4" points="${pts.join(" ")}" opacity="0.9"/>`;
  });
  s+=`</g><text class="axlab" x="${PADL}" y="6" >GPU util %</text>`;
  svg.innerHTML=s;
}
function drawGantt(){
  const GANTT_H=Math.max(80, phaseNames.length*(ROW_H+GAP)+40);
  const svg=document.getElementById("gantt"),w=W,h=GANTT_H;svg.setAttribute("viewBox",`0 0 ${w} ${h}`);
  let s=axis(svg,w,h);
  phaseNames.forEach((nm,i)=>{
    const y=8+i*(ROW_H+GAP);
    s+=`<text class="rowlab" x="${PADL-8}" y="${y+ROW_H-4}" text-anchor="end">${nm}</text>`;
    D.phases.filter(p=>p.name===nm).forEach((p,j)=>{
      let xa=sx(p.s,w),xb=sx(p.e,w);
      if(xb<PADL||xa>w-PADR)return;
      xa=Math.max(xa,PADL);xb=Math.min(xb,w-PADR);
      const ww=Math.max(xb-xa,1.0);
      s+=`<rect class="ph" x="${xa.toFixed(1)}" y="${y}" width="${ww.toFixed(1)}" height="${ROW_H}" rx="2" fill="${colorOf[nm]}" data-n="${nm}" data-s="${p.s}" data-e="${p.e}" data-r="${p.role}"/>`;
    });
  });
  svg.innerHTML=s;
  svg.querySelectorAll(".ph").forEach(el=>{
    el.addEventListener("mousemove",e=>{const s=+el.dataset.s,en=+el.dataset.e;
      tip.innerHTML=`${el.dataset.n} · ${el.dataset.r}<br>${(en-s).toFixed(2)}s  (${s.toFixed(1)}→${en.toFixed(1)}s)`;
      tip.style.opacity=1;tip.style.left=(e.clientX+12)+"px";tip.style.top=(e.clientY+12)+"px";});
    el.addEventListener("mouseleave",()=>tip.style.opacity=0);
  });
}
// zoom/pan (shared x); any zoom/pan sets userZoomed so live refresh keeps the view
function attachZoom(svg){
  svg.addEventListener("wheel",e=>{e.preventDefault();userZoomed=true;const w=W;const tc=invx(e.offsetX/svg.clientWidth*w,w);
    const k=e.deltaY<0?0.82:1.22;x0=tc-(tc-x0)*k;x1=tc+(x1-tc)*k;x0=Math.max(0,x0);x1=Math.min(D.span,x1);if(x1-x0<0.05)x1=x0+0.05;draw();},{passive:false});
  let drag=null;
  svg.addEventListener("mousedown",e=>drag={px:e.clientX});
  window.addEventListener("mouseup",()=>drag=null);
  window.addEventListener("mousemove",e=>{if(!drag)return;userZoomed=true;const w=W;const dt=(e.clientX-drag.px)/svg.clientWidth*w/(w-PADL-PADR)*(x1-x0);
    x0-=dt;x1-=dt;if(x0<0){x1-=x0;x0=0;}if(x1>D.span){x0-=x1-D.span;x1=D.span;}drag.px=e.clientX;draw();});
}
["gpu","gantt","samples"].forEach(id=>attachZoom(document.getElementById(id)));
document.getElementById("reset").onclick=()=>{userZoomed=false;x0=0;x1=D.span;draw();};
async function refresh(){
  if(SERVE){ try{ D=await (await fetch("data")).json(); }catch(e){ return; } }
  rebuild(); draw();
}
refresh();
if(SERVE) setInterval(refresh, 3000);
</script></body></html>"""


def serve(workspace: str, host: str, port: int, title: str) -> None:
    """Live server: serves the page shell once and a fresh /data JSON each poll,
    so the page auto-updates a running run's data without losing the zoom view."""
    import http.server

    shell = _TEMPLATE.replace("__TITLE__", title).replace("__SERVE__", "true").replace("__DATA__", "null").encode()

    class Handler(http.server.BaseHTTPRequestHandler):
        def _send(self, body: bytes, ctype: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/data"):
                body = json.dumps(_compute_data(*load_streams(workspace))).encode()
                self._send(body, "application/json")
            else:
                self._send(shell, "text/html; charset=utf-8")

        def log_message(self, *a):
            pass

    print(f"serving dashboard at http://{host}:{port}  (workspace={workspace})")
    http.server.HTTPServer((host, port), Handler).serve_forever()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workspace", required=True, help="miles dashboard workspace")
    ap.add_argument("--out", default="dashboard.html")
    ap.add_argument("--title", default="miles-D rollout dashboard")
    ap.add_argument("--serve", action="store_true", help="run a live auto-updating server instead of writing a static file")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    if args.serve:
        serve(args.workspace, args.host, args.port, args.title)
        return
    phases, gpu, life = load_streams(args.workspace)
    html = build_html(phases, gpu, life, title=args.title)
    with open(args.out, "w") as f:
        f.write(html)
    print(f"wrote {args.out}: {len(phases)} phase spans, {len(gpu)} gpu samples, {len(life)} lifecycle segs")


if __name__ == "__main__":
    main()
