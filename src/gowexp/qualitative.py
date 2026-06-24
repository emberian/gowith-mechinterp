"""Qualitative activation-difference analysis (box-side).

The quantitative mechanistic pass gives feature *indices* + activation *deltas*. This
pass answers "what do those features actually fire on?" — the honest, data-grounded
interpretation the channel demanded ("is it really 'uncertainty' or just 'formal text'?").

For the features that most distinguish the regimes (top D-vs-A and E-vs-A from
feature_summary.json), we re-generate a stratified sample of prompts, forward them,
and record the ANSWER-span tokens that maximally activate each feature, tagged by
condition. Neuronpedia auto-interp labels are added later, locally, in analyze.py.

Output: data/runs/white/qualitative.json
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from tqdm import tqdm

from . import conditions as C
from .model import build_prompt_ids, generate, load_lm, resid_at_positions
from .sae import encode, load_sae, resolve_sae_id
from .schema import Item, load_config, read_jsonl

_REPO = Path(__file__).resolve().parents[2]
OUT = _REPO / "data" / "runs" / "white"
SAMPLE_CONDS = ["A", "C", "D", "E", "F"]  # the regimes we contrast qualitatively


def _decode_window(lm, full_ids, pos, lo, radius=6) -> str:
    """Decode a small token window around an answer-span peak, marking the peak token."""
    seq = full_ids.tolist()
    a, b = max(lo, pos - radius), min(len(seq), pos + 2)
    pre = lm.tokenizer.decode(seq[a:pos], skip_special_tokens=True)
    peak = lm.tokenizer.decode([seq[pos]], skip_special_tokens=True)
    post = lm.tokenizer.decode(seq[pos + 1:b], skip_special_tokens=True)
    return f"{pre}⟦{peak}⟧{post}".replace("\n", " ")


def main() -> None:
    cfg = load_config()
    wb = cfg["white_box"]
    layer = wb["sae_primary_layer"]
    per_task = int(os.environ.get("GOWEXP_QUAL_ITEMS", "16"))
    top_snips = int(os.environ.get("GOWEXP_QUAL_SNIPPETS", "6"))

    summary = json.loads((OUT / "feature_summary.json").read_text())
    lsum = summary["layers"][str(layer)]
    targets = {f["feature"]: {"source": "D_vs_A", "delta": f["delta"]}
               for f in lsum.get("top_D_vs_A", [])[:20]}
    for f in lsum.get("top_E_vs_A", [])[:20]:
        targets.setdefault(f["feature"], {"source": "E_vs_A", "delta": f["delta"]})
    feat_ids = sorted(targets)
    if not feat_ids:
        print("no target features in feature_summary; run run_white first")
        return

    # stratified item sample
    items = [Item.from_dict(d) for d in read_jsonl(_REPO / "data" / "items.jsonl")]
    by_task: dict[str, list[Item]] = {}
    for it in items:
        by_task.setdefault(it.task, []).append(it)
    sample = [it for t in by_task for it in by_task[t][:per_task]]

    lm = load_lm(wb["model_id"], dtype=wb["dtype"], revision=wb["model_revision"])
    sae = load_sae(wb["sae_release"], resolve_sae_id(cfg, layer), device=lm.device)
    fidx = torch.tensor(feat_ids, device=lm.device)

    # per feature: a heap-ish list of (activation, condition, window); per (feature,condition) means
    snippets: dict[int, list[dict]] = {f: [] for f in feat_ids}
    cond_sum: dict[int, dict[str, list[float]]] = {f: {c: [] for c in SAMPLE_CONDS} for f in feat_ids}

    for it in tqdm(sample, desc="qualitative"):
        for cond in SAMPLE_CONDS:
            cp = C.render(it, cond)
            input_ids = build_prompt_ids(lm, cp.system, cp.user)
            in_len = input_ids.shape[1]
            _txt, full = generate(lm, input_ids, max_new_tokens=wb["max_new_tokens"], do_sample=False)
            if full.shape[0] <= in_len:
                continue
            resid = resid_at_positions(lm, full, [layer])[layer]      # [seq, d_model]
            ans = resid[in_len:]                                       # answer span
            feats = encode(sae, ans)[:, fidx].float()                 # [n_out, k]
            peak_vals, peak_pos = feats.max(dim=0)                    # per feature
            means = feats.mean(dim=0)
            for j, f in enumerate(feat_ids):
                cond_sum[f][cond].append(float(means[j]))
                snippets[f].append({
                    "condition": cond, "activation": float(peak_vals[j]),
                    "window": _decode_window(lm, full, in_len + int(peak_pos[j]), in_len),
                    "item": it.id})

    # keep top-K snippets per feature by activation, dedup by (item,condition)
    out_feats = []
    for f in feat_ids:
        seen, top = set(), []
        for s in sorted(snippets[f], key=lambda x: -x["activation"]):
            key = (s["item"], s["condition"])
            if key in seen:
                continue
            seen.add(key)
            top.append(s)
            if len(top) >= top_snips:
                break
        out_feats.append({
            "feature": f, "source": targets[f]["source"], "delta": targets[f]["delta"],
            "by_condition_mean": {c: (sum(v) / len(v) if v else None)
                                  for c, v in cond_sum[f].items()},
            "top_snippets": top,
        })

    (OUT / "qualitative.json").write_text(json.dumps(
        {"primary_layer": layer, "sample_items_per_task": per_task,
         "conditions": SAMPLE_CONDS, "features": out_feats}, indent=1))
    print(f"wrote qualitative.json: {len(out_feats)} features, "
          f"{len(sample)} items × {len(SAMPLE_CONDS)} conditions")


if __name__ == "__main__":
    main()
