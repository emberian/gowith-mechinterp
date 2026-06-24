"""Generate the GitHub Pages site (docs/) from report/data/results.json.

Single source of truth: the same results.json the Typst report reads. The data is
inlined into the page as a JS object, so the site works on file:// and on Pages with
no fetch/CORS. Charts via Chart.js (CDN). Run with `just site`.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
DOCS = _REPO / "docs"
COND_LABELS = {"A": "concise", "B": "padded-plain", "C": "telegraphic",
               "D": "full-Gowith", "E": "pseudo-Gowith", "F": "checklist"}
COND_COLORS = {"A": "#9aa0a6", "B": "#f4b400", "C": "#4dd0e1", "D": "#db4437",
               "E": "#ab47bc", "F": "#0f9d58"}
HYP_LABELS = {
    "gowith_helps_overall": "Gowith helps overall (D &gt; A)",
    "tokens_alone_helps": "Token budget alone helps (B &gt; A)",
    "gowith_beats_matched_tokens": "Beats matched-token plain (D &gt; B)",
    "semantics_beyond_register": "Semantics beyond weird-register (D &gt; E)",
    "grammar_beyond_checklist": "Grammar beyond checklist (D &gt; F)",
}


def _badge(true_n: int, total: int) -> str:
    cls = "yes" if true_n > total / 2 else ("partial" if true_n else "no")
    return f'<span class="badge {cls}">{true_n}/{total} tasks</span>'


def main() -> None:
    R = json.loads((_REPO / "report" / "data" / "results.json").read_text())
    DOCS.mkdir(exist_ok=True)
    (DOCS / ".nojekyll").write_text("")
    figs_src = _REPO / "report" / "figs"
    if figs_src.exists():
        (DOCS / "figs").mkdir(exist_ok=True)
        for p in figs_src.glob("*.png"):
            shutil.copy(p, DOCS / "figs" / p.name)
    (DOCS / "index.html").write_text(_render(R))
    print(f"wrote {DOCS/'index.html'} (+ figs, .nojekyll)")


def _render(R: dict) -> str:
    meta = R.get("meta", {})
    mock = "MOCK" in str(meta.get("data_source", ""))
    syn = R.get("study1", {}).get("synthesis", {})
    verdict_rows = "".join(
        f"<tr><td>{HYP_LABELS.get(k,k)}</td><td>{_badge(v['tasks_true'],v['tasks_total'])}</td></tr>"
        for k, v in syn.items()) or "<tr><td>pending run</td><td>—</td></tr>"

    data_js = json.dumps(R)
    banner = ('<div class="mock">⚠ PRELIMINARY — these numbers are a SYNTHETIC fixture '
              '(no GPU run yet). The page auto-updates when the real run lands.</div>'
              if mock else "")
    tasks = list(R.get("study1", {}).get("tasks", {}).keys())

    # cross-family table
    cf = R.get("cross_family", {})
    cf_html = ""
    if cf:
        head = "".join(f"<th>{t[:5]}</th>" for t in tasks)
        body = ""
        for model, blk in cf.items():
            cells = ""
            for t in tasks:
                cells += f"<td>{_cell_da(blk.get(t,{}))}</td>"
            body += f"<tr><td class='mono'>{model}</td>{cells}</tr>"
        cf_html = f"<table><thead><tr><th>model (D−A)</th>{head}</tr></thead><tbody>{body}</tbody></table>"

    # qualitative snippets
    qual = R.get("mechanistic", {}).get("qualitative", {})
    qual_html = _qual_html(qual) if qual else "<p class='muted'>Runs with the white-box pass.</p>"

    mech = R.get("mechanistic", {})
    ev = mech.get("E_validation", {})
    ev_html = (f"<div class='evbar'><div class='evfill' style='width:{min(1.0,ev.get('overlap_jaccard',0))*100:.0f}%'></div></div>"
               f"<p><b>E-overlap (Jaccard): {ev.get('overlap_jaccard','—'):.2f}</b> — {ev.get('interpretation','')}</p>"
               if ev else "<p class='muted'>pending</p>")

    return _TEMPLATE.format(
        title="Does relational-process register change epistemic self-monitoring?",
        question=meta.get("question", ""),
        model=meta.get("white_model", ""), sae=meta.get("sae_release", ""),
        n=meta.get("n_scored", 0), source=meta.get("data_source", ""),
        banner=banner, verdict_rows=verdict_rows,
        cond_legend="".join(
            f"<span class='chip' style='--c:{COND_COLORS[c]}'>{c} · {COND_LABELS[c]}</span>"
            for c in "ABCDEF"),
        cross_family=cf_html or "<p class='muted'>pending Bedrock replication</p>",
        qual=qual_html, evhtml=ev_html,
        data_js=data_js, tasks_js=json.dumps(tasks),
    )


def _cell_da(block: dict) -> str:
    a = block.get("A", {}).get("est"); d = block.get("D", {}).get("est")
    if a is None or d is None:
        return "—"
    diff = d - a
    col = "#0f9d58" if diff > 0.02 else ("#db4437" if diff < -0.02 else "#9aa0a6")
    return f"<span style='color:{col}'>{diff:+.2f}</span>"


def _qual_html(qual: dict) -> str:
    feats = qual.get("features", [])[:6]
    cards = ""
    for f in feats:
        snips = "".join(
            f"<li><span class='cc' style='--c:{COND_COLORS.get(s['condition'],'#888')}'>{s['condition']}</span> "
            f"<code>{_esc(s['window'])}</code> <em>{s['activation']:.1f}</em></li>"
            for s in f.get("top_snippets", [])[:4])
        cards += (f"<div class='qcard'><div class='qh'>feature {f['feature']} "
                  f"<span class='src'>{f['source']}</span> Δ{f['delta']:.2f}</div>"
                  f"<ul>{snips}</ul></div>")
    return f"<div class='qgrid'>{cards}</div>"


def _esc(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


_TEMPLATE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
:root{{--bg:#0e1117;--panel:#161b22;--ink:#e6edf3;--mut:#8b949e;--acc:#db4437;--line:#30363d}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,sans-serif}}
.wrap{{max-width:1000px;margin:0 auto;padding:28px 20px 80px}}
h1{{font-size:30px;line-height:1.2;margin:.2em 0}}
h2{{font-size:21px;margin:1.6em 0 .5em;padding-top:.6em;border-top:1px solid var(--line)}}
.sub{{color:var(--mut);font-size:15px}}
.q{{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--acc);padding:12px 16px;border-radius:8px;margin:16px 0}}
.mock{{background:#3a2a00;border:1px solid #f4b400;color:#ffd966;padding:10px 14px;border-radius:8px;margin:14px 0;font-weight:600}}
table{{border-collapse:collapse;width:100%;margin:10px 0;font-size:14px}}
th,td{{border:1px solid var(--line);padding:7px 10px;text-align:left}}
th{{background:var(--panel);color:var(--mut);font-weight:600}}
.badge{{padding:2px 9px;border-radius:20px;font-size:12px;font-weight:700}}
.badge.yes{{background:#0f3d24;color:#3fb950}} .badge.partial{{background:#3a2a00;color:#d29922}} .badge.no{{background:#3a1a1a;color:#f85149}}
.chip,.cc{{display:inline-block;padding:2px 8px;margin:2px;border-radius:6px;font-size:12px;background:color-mix(in srgb,var(--c) 22%,transparent);border:1px solid var(--c);color:var(--ink)}}
.cc{{padding:0 6px;font-weight:700}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px}}
canvas{{max-width:100%}}
.muted{{color:var(--mut)}} .mono{{font-family:ui-monospace,Menlo,monospace;font-size:12px}}
.evbar{{height:10px;background:#222;border-radius:6px;overflow:hidden;margin:6px 0}}
.evfill{{height:100%;background:linear-gradient(90deg,#0f9d58,#f4b400,#db4437)}}
.qgrid{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
.qcard{{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px}}
.qcard .qh{{font-weight:700;margin-bottom:6px}} .qcard .src{{color:var(--mut);font-weight:400;font-size:12px}}
.qcard ul{{margin:0;padding-left:16px}} .qcard li{{margin:3px 0;font-size:12px}}
.qcard code{{background:#0b0e13;padding:1px 4px;border-radius:4px}} .qcard em{{color:var(--mut);float:right}}
footer{{margin-top:50px;color:var(--mut);font-size:13px;border-top:1px solid var(--line);padding-top:16px}}
a{{color:#58a6ff}}
@media(max-width:720px){{.grid,.qgrid{{grid-template-columns:1fr}}}}
</style></head>
<body><div class="wrap">
<h1>{title}</h1>
<p class="sub">When a relational-process register (<b>Gowith</b>) or extra reasoning tokens
seems to sharpen a model's epistemic self-monitoring — which cause is it: syntax, token
budget, weird register, an implicit checklist, or vibes?</p>
{banner}
<div class="q">{question}</div>
<p class="sub">White-box: <code>{model}</code> + <code>{sae}</code> · {n} scored generations · {source}</p>
<p>{cond_legend}</p>

<h2>Headline — the five-hypothesis read-out</h2>
<table><thead><tr><th>Hypothesis</th><th>Supported in…</th></tr></thead><tbody>{verdict_rows}</tbody></table>

<h2>Study 1 — register discrimination</h2>
<div class="grid">
  <div class="card"><b>Per-condition accuracy by task</b><canvas id="c_rates" height="200"></canvas></div>
  <div class="card"><b>Register contrasts (paired Δ; CI excludes 0 ⇒ real)</b><canvas id="c_contrasts" height="200"></canvas></div>
</div>

<h2>Study 2 — output-budget dose–response</h2>
<div class="card"><b>Headline metric vs the model's own reasoning tokens</b><canvas id="c_dose" height="160"></canvas></div>

<h2>Mechanistic — the WHY (Gemma Scope 2 SAEs)</h2>
{evhtml}
<div class="grid">
  <div class="card"><b>Per-condition activation norm @ primary layer</b><canvas id="c_norms" height="180"></canvas></div>
  <div class="card"><b>Causal steering in the PLAIN condition</b><canvas id="c_steer" height="180"></canvas></div>
</div>
<h3>What the Gowith-up features actually fire on</h3>
{qual}

<h2>Cross-family replication (Bedrock)</h2>
{cross_family}

<footer>
ember &amp; Claude (Opus 4.8), from the Bridge channel · Gowith is CC0 by Andy Ayrey &amp; GPT-5.5 ·
<a href="https://github.com/emberian/gowith-mechinterp">source</a>.
Primary metrics are style-blind binary checks; LLM judges are distrusted. The pseudo-Gowith (E)
control is the validity anchor for every "register vs semantics" claim.
</footer>
</div>
<script>
const DATA = {data_js};
const TASKS = {tasks_js};
const COL = {{A:"#9aa0a6",B:"#f4b400",C:"#4dd0e1",D:"#db4437",E:"#ab47bc",F:"#0f9d58"}};
const CONDS = ["A","B","C","D","E","F"];
Chart.defaults.color="#8b949e"; Chart.defaults.borderColor="#30363d";
function ci(o){{return o&&o.est!=null?`${{(o.est*100).toFixed(0)}}% [${{(o.lo*100).toFixed(0)}},${{(o.hi*100).toFixed(0)}}]`:"—";}}

// Study 1 grouped bars
(function(){{
  const t=DATA.study1&&DATA.study1.tasks; if(!t)return;
  const ds=CONDS.map(c=>({{label:c,backgroundColor:COL[c],
    data:TASKS.map(tk=>(t[tk]&&t[tk].rate[c]&&t[tk].rate[c].est)||0)}}));
  new Chart(c_rates,{{type:"bar",data:{{labels:TASKS,datasets:ds}},
    options:{{scales:{{y:{{min:0,max:1}}}},plugins:{{legend:{{position:"bottom"}}}}}}}});
}})();

// Contrasts
(function(){{
  const co=DATA.study1&&DATA.study1.contrasts; if(!co)return;
  const keys=["D_minus_A","D_minus_B","D_minus_E","D_minus_F"];
  const labels=[],vals=[],cols=[];
  TASKS.forEach(tk=>keys.forEach(k=>{{const o=co[tk]&&co[tk][k];if(!o||o.est==null)return;
    labels.push(tk.slice(0,4)+" "+k.replace("_minus_","−").replace("D","D"));
    vals.push(o.est); cols.push((o.lo>0||o.hi<0)?"#3fb950":"#6e7681");}}));
  new Chart(c_contrasts,{{type:"bar",data:{{labels,datasets:[{{data:vals,backgroundColor:cols}}]}},
    options:{{indexAxis:"y",plugins:{{legend:{{display:false}}}},scales:{{x:{{grid:{{color:"#30363d"}}}}}}}}}});
}})();

// Dose-response
(function(){{
  const bt=DATA.study2&&DATA.study2.by_task; if(!bt)return;
  const ds=TASKS.map((tk,i)=>{{const lv=bt[tk]||{{}};const pts=Object.keys(lv).map(l=>({{
    x:lv[l].mean_out_tokens,y:lv[l].metric&&lv[l].metric.est}})).filter(p=>p.x!=null&&p.y!=null).sort((a,b)=>a.x-b.x);
    return {{label:tk,data:pts,borderColor:["#db4437","#4dd0e1","#0f9d58"][i%3],tension:.3,showLine:true}};}});
  new Chart(c_dose,{{type:"scatter",data:{{datasets:ds}},
    options:{{scales:{{x:{{title:{{display:true,text:"mean output tokens"}}}},y:{{min:0,max:1}}}},plugins:{{legend:{{position:"bottom"}}}}}}}});
}})();

// Condition norms
(function(){{
  const m=DATA.mechanistic; if(!m||!m.condition_norms)return;
  new Chart(c_norms,{{type:"bar",data:{{labels:CONDS,datasets:[{{
    data:CONDS.map(c=>m.condition_norms[c]||0),backgroundColor:CONDS.map(c=>COL[c])}}]}},
    options:{{plugins:{{legend:{{display:false}}}}}}}});
}})();

// Steering
(function(){{
  const s=DATA.steering; if(!s||!s.confab_vs_coef)return;
  function series(src,color,dash){{const d=s.confab_vs_coef[src]||{{}};
    const ks=Object.keys(d).sort((a,b)=>Number(a)-Number(b));
    return {{label:src+" features",borderColor:color,borderDash:dash,tension:.2,
      data:ks.map(k=>({{x:Number(k),y:d[k]}}))}};}}
  new Chart(c_steer,{{type:"line",data:{{datasets:[series("gowith","#db4437",[]),series("pseudo","#8b949e",[6,4])]}},
    options:{{scales:{{x:{{title:{{display:true,text:"steering coefficient"}}}},y:{{title:{{display:true,text:"confab rate"}}}}}},plugins:{{legend:{{position:"bottom"}}}}}}}});
}})();
</script>
</body></html>
"""


if __name__ == "__main__":
    main()
