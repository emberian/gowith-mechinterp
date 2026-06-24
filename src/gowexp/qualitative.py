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
from .model import generate_batch, load_lm
from .sae import encode, load_sae, resolve_sae_id
from .schema import Item, load_config, read_jsonl

_REPO = Path(__file__).resolve().parents[2]
OUT = _REPO / "data" / "runs" / "white"
SAMPLE_CONDS = ["A", "C", "D", "E", "F"]  # the regimes we contrast qualitatively


def _decode_window(lm, gen_ids: list[int], pos: int, radius: int = 6) -> str:
    """Decode a small token window around an answer-token peak, marking the peak token.
    pos indexes into the answer-span tokens (gen_ids)."""
    a, b = max(0, pos - radius), min(len(gen_ids), pos + 2)
    pre = lm.tokenizer.decode(gen_ids[a:pos], skip_special_tokens=True)
    peak = lm.tokenizer.decode([gen_ids[pos]], skip_special_tokens=True)
    post = lm.tokenizer.decode(gen_ids[pos + 1:b], skip_special_tokens=True)
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

    qbatch = int(os.environ.get("GOWEXP_BATCH", "16"))
    for cond in SAMPLE_CONDS:
        for s0 in tqdm(range(0, len(sample), qbatch), desc=f"qual:{cond}"):
            batch = sample[s0:s0 + qbatch]
            res = generate_batch(
                lm, [(C.render(it, cond).system, C.render(it, cond).user) for it in batch],
                [layer], wb["max_new_tokens"])
            for it, r in zip(batch, res):
                rr = r["resids"].get(layer)
                if rr is None or rr.shape[0] == 0:
                    continue
                feats = encode(sae, rr.float())[:, fidx].float()     # [n_out, k]
                peak_vals, peak_pos = feats.max(dim=0)
                means = feats.mean(dim=0)
                gen_ids = r["gen_ids"]
                for j, f in enumerate(feat_ids):
                    cond_sum[f][cond].append(float(means[j]))
                    snippets[f].append({
                        "condition": cond, "activation": float(peak_vals[j]),
                        "window": _decode_window(lm, gen_ids, int(peak_pos[j])),
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
