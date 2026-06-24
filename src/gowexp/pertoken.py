"""Per-token activation dynamics — does Gowith "shunt computation outwards"?

The main run mean-pooled SAE features over the answer span, discarding per-token
structure. This probe keeps it: for a sample of items across conditions, it captures
the per-token residual trajectory at the primary layer and computes, per generation:

  * participation_ratio — effective dimensionality of the per-token residual trajectory
        PR = (Σλ)² / Σλ² over the centred token×d_model matrix's covariance.
        HIGH = computation spread across many dimensions ("shunted outwards").
  * step_rel — per-token ‖resid[t]-resid[t-1]‖ / ‖resid[t]‖ : basin-holding vs advancing.
  * sae_l0 — active SAE features per token (feature participation per position).
  * novelty — fraction of a token's active features not seen earlier in the sequence
        (each token "advances one bit" vs re-ringing the same basin = "robe tokens").

Conditions A (plain) / B (padded-plain) / D (Gowith) / E (pseudo-Gowith) so the
different-text confound is controlled exactly as in the behavioral arm.

Output: data/runs/white/pertoken.json (small per-generation summaries + per-condition agg).
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
from .model import generate_batch, load_lm, set_determinism
from .sae import load_sae, resolve_sae_id
from .schema import Item, load_config, read_jsonl

_REPO = Path(__file__).resolve().parents[2]
OUT = _REPO / "data" / "runs" / "white"
CONDS = ["A", "B", "D", "E"]
TASKS = ["nonmonotonic", "correlative"]  # one crisp, one goopy


def participation_ratio(traj: torch.Tensor) -> float:
    """traj: [seq, d_model] residuals. PR = (Σλ)²/Σλ² over the centred covariance
    spectrum (λ = singular_value²) — effective dimensionality of the trajectory.
    SVD on the normalised matrix is numerically robust (Gemma has large act norms)."""
    x = traj.float()
    x = x - x.mean(dim=0, keepdim=True)
    x = x / (x.norm() + 1e-6)  # global scale: keeps SVD well-conditioned
    try:
        s = torch.linalg.svdvals(x).double()
    except Exception:
        try:
            s = torch.linalg.svdvals(x.cpu()).double()
        except Exception:
            return float("nan")
    lam = s * s
    ssum = lam.sum()
    return float((ssum * ssum) / ((lam * lam).sum() + 1e-12)) if ssum > 0 else 0.0


def main() -> None:
    cfg = load_config()
    wb = cfg["white_box"]
    layer = wb["sae_primary_layer"]
    set_determinism(cfg["seed"])
    per_task = int(os.environ.get("GOWEXP_PT_ITEMS", "12"))
    max_new = int(wb["max_new_tokens"])

    items = [Item.from_dict(d) for d in read_jsonl(_REPO / "data" / "items.jsonl")]
    by_task: dict[str, list[Item]] = defaultdict(list)
    for it in items:
        by_task[it.task].append(it)
    sample = {t: by_task[t][:per_task] for t in TASKS}

    lm = load_lm(wb["model_id"], dtype=wb["dtype"], revision=wb["model_revision"])
    sae = load_sae(wb["sae_release"], resolve_sae_id(cfg, layer), device=lm.device)
    tok = lambda s: lm.tokenizer.encode(s, add_special_tokens=False)  # noqa: E731

    rows: list[dict] = []
    for task in TASKS:
        for cond in CONDS:
            for it in tqdm(sample[task], desc=f"pt:{task}:{cond}"):
                cp = C.render(it, cond, tok)
                # batch of 1 so we get the full per-token resid trajectory back
                res = generate_batch(lm, [(cp.system, cp.user)], [layer], max_new)[0]
                traj = res["resids"].get(layer)  # [n_out, d_model] (per generated token)
                if traj is None or traj.shape[0] < 4:
                    continue
                pr = participation_ratio(traj)
                # per-token step size (relative)
                d = torch.diff(traj.float(), dim=0)
                step = (d.norm(dim=1) / (traj.float()[1:].norm(dim=1) + 1e-6)).detach().cpu().numpy()
                # per-token SAE features -> L0 and novelty
                with torch.no_grad():
                    feats = sae.encode(traj.float()).detach().cpu().numpy()  # [n_out, d_sae]
                active = feats > 0
                l0 = active.sum(axis=1)
                seen = np.zeros(active.shape[1], dtype=bool)
                novelty = np.zeros(active.shape[0])
                for t in range(active.shape[0]):
                    a = active[t]
                    new = a & ~seen
                    novelty[t] = new.sum() / max(1, a.sum())
                    seen |= a
                rows.append({
                    "item": it.id, "task": task, "condition": cond, "n_out": int(traj.shape[0]),
                    "participation_ratio": pr,
                    "pr_per_token": pr / int(traj.shape[0]),  # normalize for length
                    "step_rel_mean": float(np.mean(step)), "step_rel_cv": float(np.std(step) / (np.mean(step) + 1e-9)),
                    "l0_mean": float(np.mean(l0)), "l0_cv": float(np.std(l0) / (np.mean(l0) + 1e-9)),
                    "novelty_mean": float(np.mean(novelty)),
                    "novelty_tail": float(np.mean(novelty[len(novelty) // 2:])),  # 2nd half: still progressing?
                })

    # aggregate per (task, condition)
    agg: dict[str, dict] = {}
    metrics = ["participation_ratio", "pr_per_token", "step_rel_mean", "step_rel_cv",
               "l0_mean", "novelty_mean", "novelty_tail", "n_out"]
    for task in TASKS:
        for cond in CONDS:
            sub = [r for r in rows if r["task"] == task and r["condition"] == cond]
            if sub:
                agg[f"{task}/{cond}"] = {m: float(np.nanmean([r[m] for r in sub])) for m in metrics}
                agg[f"{task}/{cond}"]["n"] = len(sub)

    (OUT / "pertoken.json").write_text(json.dumps({"layer": layer, "rows": rows, "agg": agg}, indent=1))
    # print the headline contrast
    print("\n=== per-token dynamics: does Gowith shunt computation outwards? ===")
    print(f"{'task/cond':22}{'PR':>8}{'PR/tok':>9}{'step_cv':>9}{'L0':>7}{'novel':>8}{'nov_tail':>9}")
    for k in sorted(agg):
        a = agg[k]
        print(f"{k:22}{a['participation_ratio']:>8.1f}{a['pr_per_token']:>9.3f}{a['step_rel_cv']:>9.2f}"
              f"{a['l0_mean']:>7.0f}{a['novelty_mean']:>8.3f}{a['novelty_tail']:>9.3f}")
    for task in TASKS:
        if f"{task}/D" in agg and f"{task}/A" in agg:
            dpr = agg[f"{task}/D"]["pr_per_token"] - agg[f"{task}/A"]["pr_per_token"]
            print(f"  {task}: PR/tok  D-A={dpr:+.3f}  D-E="
                  f"{agg[f'{task}/D']['pr_per_token']-agg.get(f'{task}/E',{}).get('pr_per_token',0):+.3f}")
    print(f"\nwrote {OUT/'pertoken.json'}")


if __name__ == "__main__":
    main()
