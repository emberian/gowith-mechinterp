"""Finite-time sensitivity / Lyapunov-style probe of the generation trajectory.

Inspired by "Generalization at the Edge of Stability" (Tuci et al., 2026): there,
RDS-sharpness is the leading expansion rate (top singular value of the update Jacobian)
of the optimizer's dynamical system. Autoregressive generation is *also* a dynamical
system, so the inference-time analogue is: perturb the residual/embedding at one token
and measure how that perturbation grows or decays at downstream positions.

Method (teacher-forced, deterministic, cheap — no regeneration):
  * take the fixed generated sequence for an (item, condition),
  * clean forward -> final-layer residual H0[p] at every position,
  * for source positions s and random unit directions u: add eps*u to the INPUT
    embedding at position s, forward -> H1; record d_k = ||H1[s+k]-H0[s+k]||,
  * propagation curve d_k/d_0 and the finite-time exponent = slope of log(d_k) over k.

High exponent / slow decay  = sensitive, "edge-of-stability", influence spreads across
tokens. Low / fast decay = contractive, basin-holding. Compare A/B/D/E.

Output: data/runs/white/lyapunov.json
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
# GOWEXP_TAG routes a non-Gemma probe run to its own subdir (e.g. white/r1-14b/).
OUT = _REPO / "data" / "runs" / "white" / os.environ.get("GOWEXP_TAG", "")
CONDS = ["A", "B", "D", "E"]
TASKS = ["nonmonotonic", "correlative"]


@torch.no_grad()
def _final_resid(lm, embeds: torch.Tensor) -> torch.Tensor:
    """Final-layer hidden state [seq, d] for given input embeddings [1, seq, d]."""
    out = lm.model(inputs_embeds=embeds, use_cache=False, output_hidden_states=True)
    return out.hidden_states[-1][0]  # [seq, d]


def main() -> None:
    cfg = load_config()
    wb = cfg["white_box"]
    set_determinism(cfg["seed"])
    per_task = int(os.environ.get("GOWEXP_LY_ITEMS", "8"))
    K = int(os.environ.get("GOWEXP_LY_K", "24"))         # horizon (downstream tokens)
    n_src = int(os.environ.get("GOWEXP_LY_SRC", "4"))    # source positions per gen
    n_dir = int(os.environ.get("GOWEXP_LY_DIR", "3"))    # random directions per source
    eps_rel = float(os.environ.get("GOWEXP_LY_EPS", "0.02"))
    max_new = int(wb["max_new_tokens"])

    items = [Item.from_dict(d) for d in read_jsonl(_REPO / "data" / "items.jsonl")]
    by_task: dict[str, list[Item]] = defaultdict(list)
    for it in items:
        by_task[it.task].append(it)
    sample = {t: by_task[t][:per_task] for t in TASKS}

    # Model-agnostic: GOWEXP_MODEL overrides the white-box Gemma so we can probe the
    # trajectory dynamics of ANY open model (e.g. a reasoning model) — no SAE needed.
    model_id = os.environ.get("GOWEXP_MODEL", wb["model_id"])
    revision = "main" if os.environ.get("GOWEXP_MODEL") else wb["model_revision"]
    lm = load_lm(model_id, dtype=wb["dtype"], revision=revision)
    embed = lm.model.get_input_embeddings()
    tok = lambda s: lm.tokenizer.encode(s, add_special_tokens=False)  # noqa: E731 (B/E need it)
    rng = np.random.default_rng(cfg["seed"])

    rows: list[dict] = []
    for task in TASKS:
        for cond in CONDS:
            for it in tqdm(sample[task], desc=f"ly:{task}:{cond}"):
                cp = C.render(it, cond, tok)
                input_ids = build_prompt_ids(lm, cp.system, cp.user)
                in_len = input_ids.shape[1]
                _txt, full = generate(lm, input_ids, max_new_tokens=max_new, do_sample=False)
                seq = full.unsqueeze(0)
                if seq.shape[1] - in_len < 6:
                    continue
                with torch.no_grad():
                    base = embed(seq)                       # [1, seq, d]
                    H0 = _final_resid(lm, base)             # [seq, d]
                    tok_norm = base.norm(dim=-1).mean().item()
                    eps = eps_rel * tok_norm
                    # source positions spread across the ANSWER span
                    ans_lo, ans_hi = in_len, seq.shape[1] - 1
                    srcs = np.linspace(ans_lo, max(ans_lo, ans_hi - K), n_src).astype(int)
                    curves = []  # d_k / d_0 averaged over (src, dir)
                    for s in srcs:
                        kmax = min(K, seq.shape[1] - 1 - s)
                        if kmax < 3:
                            continue
                        for _ in range(n_dir):
                            u = torch.tensor(rng.standard_normal(base.shape[-1]),
                                             device=base.device, dtype=base.dtype)
                            u = u / (u.norm() + 1e-6)
                            pert = base.clone()
                            pert[0, s] = pert[0, s] + eps * u
                            H1 = _final_resid(lm, pert)
                            d = (H1 - H0).float().norm(dim=-1)  # [seq]
                            d0 = d[s].item() + 1e-6
                            curve = [(d[s + k].item() / d0) for k in range(kmax + 1)]
                            curves.append(curve)
                    if not curves:
                        continue
                    # average curve up to common length
                    L = min(len(c) for c in curves)
                    arr = np.array([c[:L] for c in curves])  # [n, L]
                    mean_curve = arr.mean(axis=0)
                    # finite-time exponent: slope of log(d_k/d_0) vs k (k>=1)
                    ks = np.arange(1, L)
                    logd = np.log(mean_curve[1:] + 1e-9)
                    expo = float(np.polyfit(ks, logd, 1)[0]) if len(ks) >= 2 else float("nan")
                    rows.append({
                        "item": it.id, "task": task, "condition": cond, "n_out": int(seq.shape[1] - in_len),
                        "exponent": expo,                  # >0 grows, <0 decays
                        "amp_at_K": float(mean_curve[min(L - 1, K)]),  # influence still alive at horizon
                        "curve": [float(x) for x in mean_curve[:min(L, K + 1)]],
                    })

    agg: dict[str, dict] = {}
    for task in TASKS:
        for cond in CONDS:
            sub = [r for r in rows if r["task"] == task and r["condition"] == cond]
            if sub:
                agg[f"{task}/{cond}"] = {
                    "exponent": float(np.nanmean([r["exponent"] for r in sub])),
                    "amp_at_K": float(np.nanmean([r["amp_at_K"] for r in sub])),
                    "n": len(sub)}
    (OUT / "lyapunov.json").write_text(json.dumps(
        {"K": K, "eps_rel": eps_rel, "rows": rows, "agg": agg}, indent=1))
    print("\n=== generation sensitivity (finite-time Lyapunov-style exponent) ===")
    print(f"{'task/cond':18}{'exponent':>10}{'amp@K':>10}{'n':>4}  (exponent>0 spreads, <0 damps)")
    for k in sorted(agg):
        a = agg[k]
        print(f"{k:18}{a['exponent']:>10.4f}{a['amp_at_K']:>10.3f}{a['n']:>4}")
    for task in TASKS:
        if f"{task}/D" in agg and f"{task}/A" in agg:
            print(f"  {task}: exponent D-A={agg[f'{task}/D']['exponent']-agg[f'{task}/A']['exponent']:+.4f}  "
                  f"D-E={agg[f'{task}/D']['exponent']-agg.get(f'{task}/E',{}).get('exponent',float('nan')):+.4f}")
    print(f"\nwrote {OUT/'lyapunov.json'}")


if __name__ == "__main__":
    main()
