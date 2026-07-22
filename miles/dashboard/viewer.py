"""Render the JSONL from miles.dashboard.hooks into a self-contained HTML dashboard.

Usage: python -m miles.dashboard.viewer --dump-dir <run>/dashboard --out dash.html
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


def load_streams(dump_dir: str):
    phases = _read_jsonl(sorted(glob.glob(os.path.join(dump_dir, "phases", "*.jsonl"))))
    gpu = _read_jsonl(sorted(glob.glob(os.path.join(dump_dir, "gpu_util", "*.jsonl"))))
    traj = _read_jsonl(sorted(glob.glob(os.path.join(dump_dir, "trajectories", "*.jsonl"))))
    return phases, gpu, _fold_trajectory(traj)


def build_html(phases, gpu, life=None, title="miles-D rollout dashboard") -> str:
    life = life or []
    # normalize all timestamps to the earliest event across streams
    ts_candidates = [p["t0"] for p in phases] + [g["ts"] for g in gpu] + [x["t0"] for x in life]
    t0 = min(ts_candidates) if ts_candidates else 0.0

    ph = [
        {
            "name": p.get("name", "?"),
            "role": p.get("role", "?"),
            "s": round(p["t0"] - t0, 3),
            "e": round(p["t1"] - t0, 3),
        }
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

    data = json.dumps({"phases": ph, "gpu": gp, "life": lf, "span": span}, separators=(",", ":"))
    return _TEMPLATE.replace("__TITLE__", title).replace("__DATA__", data)


_TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>__TITLE__</title>
<style>
:root{--bg:#fcfcfb;--fg:#0b0b0b;--mut:#8a8984;--grid:#e8e7e3;--panel:#f4f3f0;}
@media(prefers-color-scheme:dark){:root{--bg:#1a1a19;--fg:#fff;--mut:#8f8e86;--grid:#333330;--panel:#232320;}}
body{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;background:var(--bg);color:var(--fg);}
header{padding:14px 18px 6px;}h1{font-size:17px;margin:0 0 2px;}.sub{color:var(--mut);font-size:13px;}
.ctl{display:flex;gap:10px;align-items:center;flex-wrap:wrap;padding:6px 18px 10px;font-size:13px;}
button{font:inherit;background:var(--panel);color:var(--fg);border:1px solid var(--grid);border-radius:6px;padding:4px 10px;cursor:pointer;}
button:hover{border-color:var(--mut);}
.wrap{padding:0 18px 22px;}svg{width:100%;display:block;}
.axlab{fill:var(--mut);font-size:11px;}.grid{stroke:var(--grid);stroke-width:1;}
.rowlab{fill:var(--fg);font-size:11px;}.roundline{stroke:var(--mut);stroke-width:1;stroke-dasharray:4 4;}
.roundlab{fill:var(--mut);font-size:10px;}
.tip{position:fixed;pointer-events:none;background:var(--fg);color:var(--bg);padding:6px 9px;border-radius:6px;font-size:12px;line-height:1.45;opacity:0;transition:opacity .07s;z-index:9;white-space:nowrap;}
.hint{color:var(--mut);font-size:12px;}
</style></head><body>
<header><h1>__TITLE__</h1><div class="sub" id="sub"></div></header>
<div class="ctl">
  <button id="reset">Reset zoom</button>
  <span class="hint">wheel = zoom · drag = pan · hover = details</span>
  <span id="rounds"></span>
</div>
<div class="wrap">
  <svg id="gpu" role="img" aria-label="GPU utilization over time"></svg>
  <svg id="gantt" role="img" aria-label="Phase timeline (cross-round)"></svg>
  <div id="lifelegend" class="ctl" style="padding:2px 0 0"></div>
  <svg id="samples" role="img" aria-label="Per-sample lifecycle"></svg>
</div>
<div class="tip" id="tip"></div>
<script>
const D=__DATA__;
const PAL=["#2a78d6","#008300","#e87ba4","#eda100","#1baf7a","#eb6834","#4a3aa7","#e34948"];
const W=1200, GPU_H=190, PADL=64, PADR=16;
const tip=document.getElementById("tip");
const phaseNames=[...new Set(D.phases.map(p=>p.name))];
const colorOf={}; phaseNames.forEach((n,i)=>colorOf[n]=PAL[i%PAL.length]);
const gpuIds=[...new Set(D.gpu.map(g=>g.g))].sort();
const ROW_H=16, GAP=3, GANTT_H=Math.max(80, phaseNames.length*(ROW_H+GAP)+40);
const lifeStages=[...new Set(D.life.map(x=>x.stage))];
const stageColor={}; lifeStages.forEach((s,i)=>stageColor[s]=PAL[i%PAL.length]);
const sampMap=new Map(); D.life.forEach(x=>{if(!sampMap.has(x.k))sampMap.set(x.k,[]);sampMap.get(x.k).push(x);});
const sampKeys=[...sampMap.keys()].sort((a,b)=>Math.min(...sampMap.get(a).map(s=>s.s))-Math.min(...sampMap.get(b).map(s=>s.s)));
const SROW=3, SGAP=1, SAMP_H=Math.max(40, sampKeys.length*(SROW+SGAP)+34);
if(D.life.length){document.getElementById("lifelegend").innerHTML =
  lifeStages.map(s=>`<span style="display:inline-flex;gap:5px;align-items:center;margin-right:14px;font-size:12px"><span style="width:12px;height:12px;border-radius:3px;background:${stageColor[s]}"></span>${s}</span>`).join("");}
else{document.getElementById("samples").style.display="none";}
document.getElementById("sub").textContent =
  `span ${D.span.toFixed(1)}s · ${D.phases.length} phase spans · ${phaseNames.length} phase types · ${gpuIds.length} GPUs · ${D.gpu.length} gpu samples · ${sampKeys.length} samples`;

// round boundaries: start of each rollout_offload (fallback to actor_train)
let boundaryName = phaseNames.includes("rollout_offload") ? "rollout_offload"
  : phaseNames.includes("actor_train") ? "actor_train" : phaseNames[0];
const rounds = D.phases.filter(p=>p.name===boundaryName).map(p=>p.s).sort((a,b)=>a-b);

// shared x-domain [x0,x1] (seconds); zoom/pan mutate it
let x0=0, x1=D.span;
function sx(t,w){return PADL + (t-x0)/(x1-x0)*(w-PADL-PADR);}
function invx(px,w){return x0 + (px-PADL)/(w-PADL-PADR)*(x1-x0);}

function ticks(){const n=8,step=niceStep((x1-x0)/n),out=[];for(let t=Math.ceil(x0/step)*step;t<=x1;t+=step)out.push(+t.toFixed(6));return out;}
function niceStep(r){const p=Math.pow(10,Math.floor(Math.log10(r)));const f=r/p;return (f<1.5?1:f<3?2:f<7?5:10)*p;}

function draw(){
  drawGpu(); drawGantt(); if(D.life.length)drawSamples();
}
function drawSamples(){
  const svg=document.getElementById("samples"),w=W,h=SAMP_H;svg.setAttribute("viewBox",`0 0 ${w} ${h}`);
  let s=axis(svg,w,h);
  s+=`<clipPath id="cs"><rect x="${PADL}" y="0" width="${w-PADL-PADR}" height="${h}"/></clipPath><g clip-path="url(#cs)">`;
  sampKeys.forEach((k,i)=>{
    const y=8+i*(SROW+SGAP);
    sampMap.get(k).forEach(seg=>{
      let xa=sx(seg.s,w),xb=sx(seg.e,w);
      if(xb<PADL||xa>w-PADR)return;
      xa=Math.max(xa,PADL);xb=Math.min(xb,w-PADR);
      s+=`<rect class="lf" x="${xa.toFixed(1)}" y="${y}" width="${Math.max(xb-xa,0.6).toFixed(1)}" height="${SROW}" fill="${stageColor[seg.stage]}" data-k="${k}" data-st="${seg.stage}" data-s="${seg.s}" data-e="${seg.e}"/>`;
    });
  });
  s+=`</g><text class="axlab" x="${PADL}" y="6">per-sample lifecycle — ${sampKeys.length} samples (each row = 1 request)</text>`;
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
  const svg=document.getElementById("gantt"),w=W,h=GANTT_H;svg.setAttribute("viewBox",`0 0 ${w} ${h}`);
  let s=axis(svg,w,h);
  phaseNames.forEach((nm,i)=>{
    const y=8+i*(ROW_H+GAP);
    s+=`<text class="rowlab" x="4" y="${y+ROW_H-3}">${nm}</text>`;
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
// zoom/pan on both svgs (shared x)
function attachZoom(svg){
  svg.addEventListener("wheel",e=>{e.preventDefault();const w=W;const tc=invx(e.offsetX/svg.clientWidth*w,w);
    const k=e.deltaY<0?0.82:1.22;x0=tc-(tc-x0)*k;x1=tc+(x1-tc)*k;x0=Math.max(0,x0);x1=Math.min(D.span,x1);if(x1-x0<0.05)x1=x0+0.05;draw();},{passive:false});
  let drag=null;
  svg.addEventListener("mousedown",e=>drag={px:e.clientX});
  window.addEventListener("mouseup",()=>drag=null);
  window.addEventListener("mousemove",e=>{if(!drag)return;const w=W;const dt=(e.clientX-drag.px)/svg.clientWidth*w/(w-PADL-PADR)*(x1-x0);
    x0-=dt;x1-=dt;if(x0<0){x1-=x0;x0=0;}if(x1>D.span){x0-=x1-D.span;x1=D.span;}drag.px=e.clientX;draw();});
}
["gpu","gantt","samples"].forEach(id=>attachZoom(document.getElementById(id)));
document.getElementById("reset").onclick=()=>{x0=0;x1=D.span;draw();};
// round jump buttons
const rb=document.getElementById("rounds");
rounds.forEach((r,i)=>{const b=document.createElement("button");b.textContent="R"+i;b.onclick=()=>{const nxt=rounds[i+1]??D.span;const pad=(nxt-r)*0.05;x0=Math.max(0,r-pad);x1=Math.min(D.span,nxt+pad);draw();};rb.appendChild(b);});
if(rounds.length)rb.insertAdjacentHTML("afterbegin",`<span class="hint">rounds (${boundaryName}): </span>`);
draw();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump-dir", required=True, help="the run's <dump_details>/dashboard directory")
    ap.add_argument("--out", default="dashboard.html")
    ap.add_argument("--title", default="miles-D rollout dashboard")
    args = ap.parse_args()

    phases, gpu, life = load_streams(args.dump_dir)
    html = build_html(phases, gpu, life, title=args.title)
    with open(args.out, "w") as f:
        f.write(html)
    print(f"wrote {args.out}: {len(phases)} phase spans, {len(gpu)} gpu samples, {len(life)} lifecycle segs")


if __name__ == "__main__":
    main()
