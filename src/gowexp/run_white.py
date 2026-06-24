"""White-box run on Gemma-3-12B-it: greedy generations + per-condition SAE features.

For each rendered prompt we:
  1. generate greedily (deterministic -> behavioral score + reproducible activations),
  2. re-forward the full [prompt+answer] sequence once, grab resid_post at the target
     layers, encode through the Gemma Scope 2 SAE, and mean-pool over the ANSWER span.

Outputs (data/runs/white/):
  generations.jsonl    one Generation row per prompt (text + token counts)  [always fetched]
  sae_means_L{L}.npz   [n_prompts, d_sae] float16 answer-span mean features  [optional fetch]
  row_index.json       row -> (item_id, condition)
  feature_summary.json per-condition mean vectors + top D-vs-A / E-vs-A features [always fetched]

Headline metrics come from the greedy pass with item-level bootstrap CIs; stochastic
samples are optional (GOWEXP_SAMPLES>0).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .model import build_prompt_ids, generate, load_lm, resid_at_positions, set_determinism
from .sae import d_sae, encode, load_sae, resolve_sae_id
from .schema import Generation, RenderedPrompt, load_config, read_jsonl

_REPO = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("GOWEXP_OUT", str(_REPO / "data" / "runs" / "white")))


def _condition_means(means: np.ndarray, conditions: list[str]) -> dict[str, np.ndarray]:
    """Mean feature vector per condition (float32)."""
    out: dict[str, np.ndarray] = {}
    conds = np.array(conditions)
    for c in sorted(set(conditions)):
        out[c] = means[conds == c].astype(np.float32).mean(axis=0)
    return out


def _top_diff(a: np.ndarray, b: np.ndarray, k: int = 40) -> list[dict]:
    """Top features by (a - b), returning index + both means + delta."""
    delta = a - b
    idx = np.argsort(-delta)[:k]
    return [{"feature": int(i), "a": float(a[i]), "b": float(b[i]), "delta": float(delta[i])}
            for i in idx]


def main() -> None:
    cfg = load_config()
    wb = cfg["white_box"]
    set_determinism(cfg["seed"])
    OUT.mkdir(parents=True, exist_ok=True)

    prompts = [RenderedPrompt.from_dict(d) for d in read_jsonl(_REPO / "data" / "prompts.jsonl")]
    limit = int(os.environ.get("GOWEXP_LIMIT", "0"))
    if limit:
        prompts = prompts[:limit]
        print(f"GOWEXP_LIMIT={limit}: smoke subset")
    n = len(prompts)
    layers = wb["sae_layers"]
    primary = wb["sae_primary_layer"]

    print(f"loading {wb['model_id']} ...")
    lm = load_lm(wb["model_id"], dtype=wb["dtype"], revision=wb["model_revision"])
    print(f"  {lm.n_layers} layers on {lm.device}")

    print("loading SAEs ...")
    saes = {L: load_sae(wb["sae_release"], resolve_sae_id(cfg, L), device=lm.device) for L in layers}
    dsae = d_sae(saes[primary])
    print(f"  d_sae={dsae} at layers {layers}")

    # preallocate per-layer answer-span mean feature arrays; RESUME from existing npz.
    means = {L: np.zeros((n, dsae), dtype=np.float16) for L in layers}
    for L in layers:
        f = OUT / f"sae_means_L{L}.npz"
        if f.exists():
            arr = np.load(f)["means"]
            if arr.shape == means[L].shape:
                means[L] = arr

    # row order is stable because prompts.jsonl is frozen -> row i maps to the same cell.
    row_index = [{"row": i, "item_id": p.item_id, "condition": p.condition, "task": p.task}
                 for i, p in enumerate(prompts)]

    # resume: skip (item_id, condition) cells already in generations.jsonl; append.
    gen_path = OUT / "generations.jsonl"
    done: set[tuple[str, str]] = set()
    if gen_path.exists():
        for d in read_jsonl(gen_path):
            done.add((d["item_id"], d["condition"]))
    if done:
        print(f"resuming: {len(done)}/{n} cells already generated")
    gen_f = open(gen_path, "a")

    def _save_npz() -> None:
        for L in layers:
            np.savez_compressed(OUT / f"sae_means_L{L}.npz", means=means[L])

    n_new = 0
    for i, p in enumerate(tqdm(prompts, desc="white-box")):
        if (p.item_id, p.condition) in done:
            continue
        input_ids = build_prompt_ids(lm, p.system, p.user)
        in_len = input_ids.shape[1]
        completion, full = generate(lm, input_ids, max_new_tokens=wb["max_new_tokens"],
                                    do_sample=False)
        ans_lo, ans_hi = in_len, full.shape[0]
        n_out = ans_hi - ans_lo

        if n_out > 0:
            resid = resid_at_positions(lm, full, layers)  # {L: [seq, d_model]}
            for L in layers:
                ans = resid[L][ans_lo:ans_hi]                 # [n_out, d_model]
                feats = encode(saes[L], ans)                  # [n_out, d_sae]
                means[L][i] = feats.float().mean(dim=0).to("cpu", torch.float16).numpy()
            del resid

        g = Generation(item_id=p.item_id, task=p.task, condition=p.condition,
                       model=wb["model_id"], sample_idx=0, decode="greedy",
                       text=completion, n_input_tokens=int(in_len), n_output_tokens=int(n_out),
                       sae_record=f"row:{i}")
        gen_f.write(json.dumps(g.to_dict(), ensure_ascii=False) + "\n")
        n_new += 1
        if n_new % 200 == 0:
            gen_f.flush()
            _save_npz()

    gen_f.close()
    _save_npz()
    (OUT / "row_index.json").write_text(json.dumps(row_index))

    # ---- on-box feature summary (small, always fetched) --------------------
    conditions = [r["condition"] for r in row_index]
    summary = {"layers": {}, "primary_layer": primary, "d_sae": dsae}
    for L in layers:
        cm = _condition_means(means[L], conditions)
        layer_summary = {
            "condition_means_l2norm": {c: float(np.linalg.norm(v)) for c, v in cm.items()},
            "top_D_vs_A": _top_diff(cm["D"], cm["A"]) if "A" in cm and "D" in cm else [],
            "top_E_vs_A": _top_diff(cm["E"], cm["A"]) if "A" in cm and "E" in cm else [],
            "top_D_vs_B": _top_diff(cm["D"], cm["B"]) if "B" in cm and "D" in cm else [],
        }
        summary["layers"][str(L)] = layer_summary
    (OUT / "feature_summary.json").write_text(json.dumps(summary, indent=1))

    print(f"\nwrote {len(done) + n_new} generations (+{n_new} new) + SAE means {layers} -> {OUT}")
    print("feature_summary.json written (per-condition means + top differential features)")


if __name__ == "__main__":
    main()
