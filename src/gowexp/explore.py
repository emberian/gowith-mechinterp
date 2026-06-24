"""Feature explorer — make the captured activations browseable for the channel.

Our SAEs are Gemma Scope 2, which Neuronpedia hosts with interactive dashboards
(auto-interp labels, top-activating examples, a steering playground). So this builds
a bridge: for the features Gowith moves most, it pulls Neuronpedia's label and links
the live dashboard, and shows our OWN max-activating snippets (from qualitative.json)
— what the feature fired on in our Gowith vs plain generations.

Outputs docs/explore.html (linked from the main site). Labels are cached to
data/runs/np_labels.json so rebuilds are free / offline.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

from .schema import load_config

_REPO = Path(__file__).resolve().parents[2]
RESULTS = _REPO / "report" / "data" / "results.json"
QUAL = _REPO / "data" / "runs" / "white" / "qualitative.json"
LABEL_CACHE = _REPO / "data" / "runs" / "np_labels.json"
DOCS = _REPO / "docs"

NP_MODEL = "gemma-3-12b-it"


def np_source(layer: int) -> str:
    return f"{layer}-gemmascope-2-res-65k"


def np_url(layer: int, feat: int) -> str:
    return f"https://www.neuronpedia.org/{NP_MODEL}/{np_source(layer)}/{feat}"


def _api(layer: int, feat: int) -> str:
    return f"https://www.neuronpedia.org/api/feature/{NP_MODEL}/{np_source(layer)}/{feat}"


def fetch_label(layer: int, feat: int, cache: dict) -> str:
    key = f"{layer}:{feat}"
    if key in cache:
        return cache[key]
    label = "(no label)"
    try:
        req = urllib.request.Request(_api(layer, feat), headers={"accept": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as r:
            d = json.loads(r.read())
        exps = d.get("explanations") or []
        if exps:
            label = exps[0].get("description", label)
    except Exception:
        label = "(lookup failed)"
    cache[key] = label
    return label


def _load_cache() -> dict:
    return json.loads(LABEL_CACHE.read_text()) if LABEL_CACHE.exists() else {}


def build() -> None:
    cfg = load_config()
    layer = cfg["white_box"]["sae_primary_layer"]
    R = json.loads(RESULTS.read_text())
    mech = R.get("mechanistic", {})
    ac = mech.get("arm_contrast", {})
    qual = json.loads(QUAL.read_text()) if QUAL.exists() else {}
    qual_by_feat = {f["feature"]: f for f in qual.get("features", [])}

    crisp = ac.get("crisp", [])
    indomain = ac.get("in-domain", [])
    cset = {f["feature"] for f in crisp}
    iset = {f["feature"] for f in indomain}
    shared = cset & iset
    indomain_only = [f for f in indomain if f["feature"] not in cset]
    crisp_only = [f for f in crisp if f["feature"] not in iset]

    cache = _load_cache()
    # fetch labels for everything we'll show (top 15 of each bucket)
    buckets = {
        "Gowith raises these on GOOPY (relational/causal) tasks but NOT on crisp ones": indomain_only[:15],
        "Gowith raises these on CRISP (deduction) tasks but NOT on goopy ones": crisp_only[:15],
        "Gowith raises these everywhere (generic 'reading Gowith' — watch for formatting/encoding)":
            [f for f in indomain if f["feature"] in shared][:10],
    }
    for feats in buckets.values():
        for f in feats:
            fetch_label(layer, f["feature"], cache)
    LABEL_CACHE.write_text(json.dumps(cache, indent=1))

    DOCS.mkdir(exist_ok=True)
    (DOCS / "explore.html").write_text(_render(layer, buckets, qual_by_feat, mech, cache))
    print(f"wrote {DOCS/'explore.html'} ({sum(len(v) for v in buckets.values())} features, "
          f"{len(cache)} labels cached)")


def _snip_html(feat: int, qbf: dict) -> str:
    q = qbf.get(feat)
    if not q:
        return ""
    COL = {"A": "#9aa0a6", "C": "#4dd0e1", "D": "#db4437", "E": "#ab47bc", "F": "#0f9d58"}
    items = "".join(
        f"<li><span class='cc' style='--c:{COL.get(s['condition'],'#888')}'>{s['condition']}</span> "
        f"<code>{_esc(s['window'])}</code> <em>{s['activation']:.1f}</em></li>"
        for s in q.get("top_snippets", [])[:4])
    bycond = q.get("by_condition_mean", {})
    means = " ".join(f"{c}:{(bycond.get(c) or 0):.2f}" for c in ["A", "C", "D", "E", "F"]
                     if bycond.get(c) is not None)
    return f"<div class='snips'><div class='means'>our mean act — {means}</div><ul>{items}</ul></div>"


def _esc(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _render(layer: int, buckets: dict, qbf: dict, mech: dict, cache: dict) -> str:
    ov = mech.get("arm_contrast", {}).get("overlap_jaccard")
    sections = ""
    for title, feats in buckets.items():
        rows = ""
        for f in feats:
            fid = f["feature"]
            label = cache.get(f"{layer}:{fid}", "(no label)")
            rows += (
                f"<div class='feat'>"
                f"<div class='fhead'><a href='{np_url(layer, fid)}' target='_blank'>#{fid} ↗</a>"
                f"<span class='lbl'>{_esc(label)}</span>"
                f"<span class='delta'>Δ{f['delta']:.2f}</span></div>"
                f"{_snip_html(fid, qbf)}</div>")
        sections += f"<h2>{title}</h2><div class='grid'>{rows}</div>"

    # honest tally of what the labels actually are
    labels = [cache.get(f"{layer}:{f['feature']}", "") for b in buckets.values() for f in b]
    labeled = [x for x in labels if x and x not in ("(no label)", "(lookup failed)")]
    ov_line = (
        f"<p class='note'>Across the top-40 features, goopy-task and crisp-task lists overlap "
        f"only <b>Jaccard {ov:.2f}</b> — Gowith engages largely <b>different</b> residual features "
        f"per regime.</p>" if ov is not None else "")
    ov_line += (
        "<p class='note' style='border-left-color:#f4b400'>"
        "<b>Honest read of the labels:</b> the Gowith-moved features that Neuronpedia has labelled "
        f"({len(labeled)}/{len(labels)} shown; auto-interp is sparse for this new suite) are "
        "predominantly <b>syntactic / formatting / topical</b> — punctuation, multilingual tokens, "
        "numbered lists, after-punctuation positions. The few that brush <i>semantic</i> "
        "self-monitoring (“warnings and disclaimers”, “likelihood and probability”) sit on the "
        "<b>crisp</b> side, not the goopy one. We did <b>not</b> find a clean dominant "
        "“uncertainty” or “agency” feature driving the goopy-task help. That supports the "
        "deflationary read — much of what Gowith does mechanically is shift into a weird register — "
        "and it's exactly the “is the feature real or just formal-text” check the room cared about. "
        "Suggestive, not proven; bring your own skepticism (and the unlabelled ones below need it most).</p>")

    return f"""<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gowith feature explorer — Gemma Scope 2</title>
<style>
:root{{--bg:#0e1117;--panel:#161b22;--ink:#e6edf3;--mut:#8b949e;--acc:#db4437;--line:#30363d}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,sans-serif}}
.wrap{{max-width:1000px;margin:0 auto;padding:28px 20px 80px}}
h1{{font-size:27px;margin:.2em 0}} h2{{font-size:18px;margin:1.8em 0 .6em;border-top:1px solid var(--line);padding-top:1em}}
a{{color:#58a6ff;text-decoration:none}} a:hover{{text-decoration:underline}}
.note{{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--acc);padding:10px 14px;border-radius:8px}}
.grid{{display:grid;gap:10px}}
.feat{{background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:10px 12px}}
.fhead{{display:flex;gap:10px;align-items:baseline}}
.fhead a{{font-family:ui-monospace,Menlo,monospace;font-weight:700;white-space:nowrap}}
.lbl{{flex:1;color:var(--ink)}} .delta{{color:var(--mut);font-size:12px;font-family:ui-monospace,monospace}}
.snips{{margin-top:7px;border-top:1px dashed var(--line);padding-top:6px}}
.snips .means{{color:var(--mut);font-size:11px;font-family:ui-monospace,monospace;margin-bottom:3px}}
.snips ul{{margin:0;padding-left:15px}} .snips li{{font-size:12px;margin:2px 0}}
.snips code{{background:#0b0e13;padding:1px 4px;border-radius:4px}} .snips em{{color:var(--mut);float:right}}
.cc{{display:inline-block;padding:0 6px;border-radius:5px;font-weight:700;font-size:11px;
  background:color-mix(in srgb,var(--c) 22%,transparent);border:1px solid var(--c)}}
.intro{{color:var(--mut)}} code.k{{background:var(--panel);padding:1px 5px;border-radius:4px}}
footer{{margin-top:50px;color:var(--mut);font-size:13px;border-top:1px solid var(--line);padding-top:16px}}
</style></head><body><div class="wrap">
<p><a href="index.html">← results</a></p>
<h1>Gowith feature explorer</h1>
<p class="intro">The features <b>gemma-3-12b-it</b> moves most when it reasons in Gowith (vs plain),
at residual layer {layer}, through the <b>Gemma Scope 2</b> SAE. Each <code class="k">#id ↗</code>
opens the live <a href="https://www.neuronpedia.org/{NP_MODEL}/{np_source(layer)}" target="_blank">Neuronpedia</a>
dashboard — auto-interp label, top-activating examples, and a steering playground. Under each, the
snippets are what that feature fired on in <b>our</b> runs (⟦token⟧ = peak), tagged by condition
(A plain · C telegraphic · D Gowith · E pseudo-Gowith · F checklist).</p>
{ov_line}
<p class="intro">Bring your own question: pick a feature, open its dashboard, and ask whether the label
is really "uncertainty / agency / relation" — or just "weird formatted text." That skeptical check
(does the feature mean what its label says?) is the whole game.</p>
{sections}
<footer>data + loader to query any feature locally: <a href="https://github.com/emberian/gowith-mechinterp">github.com/emberian/gowith-mechinterp</a>
· <code class="k">python -m gowexp.feature &lt;id&gt;</code> prints per-condition activations from our npz.
Gemma Scope 2 © Google DeepMind (CC-BY-4.0); dashboards © Neuronpedia.</footer>
</div></body></html>"""


if __name__ == "__main__":
    build()
