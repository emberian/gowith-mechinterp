"""Attention-range probe — is Gowith's "higher thought" carried in longer-range connections?

ember's hypothesis: Gowith's templated token-rhythm makes the LOCAL next-token step
cheap/automatic, freeing the reasoning into LONGER-RANGE connections. Direct test:
does the model literally attend further back when generating in Gowith vs plain?

For each (item, condition), generate, then re-forward with output_attentions and, over
the answer-span query positions, compute per-head the mean attention distance
  dist(q) = Σ_k a[q,k] * (q - k)
and the long-range mass (fraction of attention weight on keys > W tokens back, where W
≈ Gemma-3's local sliding-window size). Aggregate per condition; contrast D vs A vs E.

Incremental JSONL writes (lesson from earlier crashes). Output: data/runs/white/attn_range.json
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from . import conditions as C
from .model import build_prompt_ids, generate, load_lm, set_determinism
from .schema import Item, load_config, read_jsonl

_REPO = Path(__file__).resolve().parents[2]
OUT = _REPO / "data" / "runs" / "white" / os.environ.get("GOWEXP_TAG", "")
CONDS = ["A", "D", "E", "Y", "W", "P"]  # plain / Gowith / scramble / hyphen / word-line / pig-latin
TASKS = ["nonmonotonic", "correlative"]
LONG_RANGE_W = 256  # keys further back than this count as "long-range" (~Gemma-3 swa window)


@torch.no_grad()
def _attn_stats(lm, full_ids: torch.Tensor, ans_lo: int) -> dict:
    """Mean attention distance + long-range mass over answer-span queries, per layer."""
    out = lm.model(full_ids.unsqueeze(0), use_cache=False, output_attentions=True)
    atts = out.attentions  # tuple[n_layers] of [1, heads, seq, seq]
    if not atts or atts[0] is None:
        return {}
    seq = full_ids.shape[0]
    qs = torch.arange(ans_lo, seq, device=full_ids.device)
    if qs.numel() == 0:
        return {}
    kidx = torch.arange(seq, device=full_ids.device).float()
    per_layer_dist, per_layer_lr = [], []
    for a in atts:
        A = a[0].float()                       # [heads, seq, seq]
        Aq = A[:, qs, :]                       # [heads, nq, seq]
        # distance q-k weighted by attention; only k<=q contribute (causal)
        dist = (Aq * (qs.float().view(1, -1, 1) - kidx.view(1, 1, -1))).sum(-1)  # [heads, nq]
        # long-range mass: keys more than W back
        far = (qs.float().view(1, -1, 1) - kidx.view(1, 1, -1)) > LONG_RANGE_W
        lr = (Aq * far).sum(-1)                # [heads, nq]
        per_layer_dist.append(dist.mean().item())
        per_layer_lr.append(lr.mean().item())
    return {
        "mean_dist": float(np.mean(per_layer_dist)),
        "max_layer_dist": float(np.max(per_layer_dist)),
        "long_range_mass": float(np.mean(per_layer_lr)),
        "per_layer_dist": [float(x) for x in per_layer_dist],
    }


def main() -> None:
    cfg = load_config()
    wb = cfg["white_box"]
    set_determinism(cfg["seed"])
    per_task = int(os.environ.get("GOWEXP_AR_ITEMS", "10"))
    max_new = int(wb["max_new_tokens"])

    items = [Item.from_dict(d) for d in read_jsonl(_REPO / "data" / "items.jsonl")]
    by_task: dict[str, list[Item]] = defaultdict(list)
    for it in items:
        by_task[it.task].append(it)
    sample = {t: by_task[t][:per_task] for t in TASKS}

    # eager attention is REQUIRED to read attention weights (sdpa returns none).
    model_id = os.environ.get("GOWEXP_MODEL", wb["model_id"])
    revision = "main" if os.environ.get("GOWEXP_MODEL") else wb["model_revision"]
    lm = load_lm(model_id, dtype=wb["dtype"], revision=revision, attn_implementation="eager")
    tok = lambda s: lm.tokenizer.encode(s, add_special_tokens=False)  # noqa: E731

    OUT.mkdir(parents=True, exist_ok=True)
    rows_path = OUT / "attn_range.rows.jsonl"
    done = set()
    if rows_path.exists():
        for d in read_jsonl(rows_path):
            done.add((d["item"], d["condition"]))
    rf = open(rows_path, "a")

    rows: list[dict] = list(read_jsonl(rows_path)) if rows_path.exists() else []
    for task in TASKS:
        for cond in CONDS:
            for it in tqdm([x for x in sample[task] if (x.id, cond) not in done],
                           desc=f"ar:{task}:{cond}"):
                cp = C.render(it, cond, tok)
                input_ids = build_prompt_ids(lm, cp.system, cp.user)
                in_len = input_ids.shape[1]
                _txt, full = generate(lm, input_ids, max_new_tokens=max_new, do_sample=False)
                if full.shape[0] - in_len < 4:
                    continue
                st = _attn_stats(lm, full, in_len)
                if not st:
                    continue
                row = {"item": it.id, "task": task, "condition": cond,
                       "n_out": int(full.shape[0] - in_len), **{k: v for k, v in st.items()
                                                                 if k != "per_layer_dist"}}
                rf.write(json.dumps(row) + "\n")
                rf.flush()
                rows.append(row)
    rf.close()

    agg: dict[str, dict] = {}
    for task in TASKS:
        for cond in CONDS:
            sub = [r for r in rows if r["task"] == task and r["condition"] == cond]
            if sub:
                agg[f"{task}/{cond}"] = {
                    "mean_dist": float(np.mean([r["mean_dist"] for r in sub])),
                    "max_layer_dist": float(np.mean([r["max_layer_dist"] for r in sub])),
                    "long_range_mass": float(np.mean([r["long_range_mass"] for r in sub])),
                    "n": len(sub)}
    (OUT / "attn_range.json").write_text(json.dumps({"long_range_w": LONG_RANGE_W, "agg": agg}, indent=1))
    print("\n=== attention range: does Gowith attend further back? ===")
    print(f"{'task/cond':18}{'mean_dist':>10}{'max_layer':>10}{'LR_mass':>9}{'n':>4}")
    for k in sorted(agg):
        a = agg[k]
        print(f"{k:18}{a['mean_dist']:>10.1f}{a['max_layer_dist']:>10.1f}{a['long_range_mass']:>9.3f}{a['n']:>4}")
    for t in TASKS:
        if f"{t}/D" in agg and f"{t}/A" in agg:
            dd = agg[f"{t}/D"]["mean_dist"] - agg[f"{t}/A"]["mean_dist"]
            dl = agg[f"{t}/D"]["long_range_mass"] - agg[f"{t}/A"]["long_range_mass"]
            print(f"  {t}: mean_dist D-A={dd:+.1f} tokens   long_range_mass D-A={dl:+.3f}")
    print(f"\nwrote {OUT/'attn_range.json'}")


if __name__ == "__main__":
    main()
