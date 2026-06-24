"""Causal capstone: steer the Gowith-up features in the PLAIN condition.

Logic: run_white found the features whose mean activation rises most under Gowith (D)
vs plain (A) at the primary layer. If those features *cause* the epistemic effect, then
adding them to the residual stream while the model runs the *plain* prompt should move
the behavior (lower confabulation, cleaner updates) — without any Gowith text present.

We sweep a coefficient (0 = unsteered control) and score with the same style-blind
binary scorers. The pseudo-Gowith (E) features are steered too, as a control: if E's
features move behavior as much as D's, the effect is register, not Gowith semantics.
"""
from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from tqdm import tqdm

from . import conditions as C
from .model import _decoder_layers, build_prompt_ids, generate, load_lm, set_determinism
from .sae import load_sae, resolve_sae_id
from .schema import Item, load_config, read_jsonl
from .scoring import extract_answer
from .tasks import REGISTRY

_REPO = Path(__file__).resolve().parents[2]
OUT = _REPO / "data" / "runs" / "white"


@contextlib.contextmanager
def steer(lm, layer: int, vec: torch.Tensor) -> Iterator[None]:
    """Add `vec` to the residual (resid_post) at `layer` on every forward, all positions."""
    decoder = _decoder_layers(lm.model)

    def hook(_mod, _inp, out):
        if isinstance(out, tuple):
            return (out[0] + vec.to(out[0].dtype),) + tuple(out[1:])
        return out + vec.to(out.dtype)

    h = decoder[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()


def steering_vector(sae, feats: list[dict], coef: float, device, dtype) -> torch.Tensor:
    """coef * Σ_f (Gowith-level activation_f) * W_dec[f].  coef=1 injects exactly the
    mean Gowith-level activation of the selected features into the plain prompt."""
    d_model = sae.W_dec.shape[1]
    vec = torch.zeros(d_model, device=device, dtype=torch.float32)
    for f in feats:
        a = float(f["a"])  # D (gowith) mean activation of this feature
        vec = vec + coef * a * sae.W_dec[int(f["feature"])].float()
    return vec.to(dtype)


def _score(item: Item, text: str) -> dict:
    return REGISTRY[item.task].score(item, extract_answer(text), text)


def main() -> None:
    cfg = load_config()
    wb, st = cfg["white_box"], cfg["steering"]
    set_determinism(cfg["seed"])
    layer = st["layer"]
    coefs = [0.0] + [float(c) for c in st["coefficients"]]  # 0 = unsteered control

    summary = json.loads((OUT / "feature_summary.json").read_text())
    lsum = summary["layers"][str(layer)]
    d_feats = lsum["top_D_vs_A"][: st["top_k_features"]]
    e_feats = lsum["top_E_vs_A"][: st["top_k_features"]]

    allit = [Item.from_dict(d) for d in read_jsonl(_REPO / "data" / "items.jsonl")]
    # steer where the behavioral effect is sharpest: confabulation + belief revision.
    # Cap to a balanced subset (steering is the expensive arm).
    cap = st.get("max_items")
    nm = [it for it in allit if it.task == "nonmonotonic"]
    ep = [it for it in allit if it.task == "epistemic"]
    if cap:
        nm, ep = nm[: cap // 2], ep[: cap // 2]
    items = nm + ep

    lm = load_lm(wb["model_id"], dtype=wb["dtype"], revision=wb["model_revision"])
    sae = load_sae(wb["sae_release"], resolve_sae_id(cfg, layer), device=lm.device)
    dtype = next(lm.model.parameters()).dtype

    vectors = {
        "gowith": {c: steering_vector(sae, d_feats, c, lm.device, dtype) for c in coefs},
        "pseudo": {c: steering_vector(sae, e_feats, c, lm.device, dtype) for c in coefs},
    }

    # resume: skip (item_id, source, coef) already done; append incrementally.
    steer_path = OUT / "steer.jsonl"
    rows: list[dict] = []
    done: set[tuple] = set()
    if steer_path.exists():
        for line in steer_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows.append(r)
                done.add((r["item_id"], r["source"], r["coef"]))
    if done:
        print(f"resuming steer: {len(done)} cells done")
    sf = open(steer_path, "a")

    for it in tqdm(items, desc="steer"):
        cp = C.render(it, "A")  # plain condition; no Gowith text at all
        input_ids = build_prompt_ids(lm, cp.system, cp.user)
        for source in ("gowith", "pseudo"):
            for c in coefs:
                if c == 0.0 and source == "pseudo":
                    continue  # one shared unsteered baseline (source=gowith, coef=0)
                if (it.id, source, c) in done:
                    continue
                with steer(lm, layer, vectors[source][c]) if c != 0.0 else _nullctx():
                    text, _ = generate(lm, input_ids, max_new_tokens=wb["max_new_tokens"],
                                       do_sample=False)
                sc = _score(it, text)
                r = {"item_id": it.id, "task": it.task, "source": source,
                     "coef": c, "text": text, "scores": sc}
                rows.append(r)
                sf.write(json.dumps(r) + "\n")
    sf.close()

    # quick on-box summary: primary-metric rate vs coef, per source
    def rate(task, source, c, key, want=True):
        xs = [r["scores"].get(key) for r in rows
              if r["task"] == task and r["scores"].get(key) is not None
              and (r["source"] == source or c == 0.0) and r["coef"] == c]
        return (float(np.mean([x == want for x in xs])) if xs else None)

    summ = {"layer": layer, "coefs": coefs, "features_gowith": d_feats, "features_pseudo": e_feats,
            "confab_vs_coef": {s: {c: rate("epistemic", s, c, "confabulated") for c in coefs}
                               for s in ("gowith", "pseudo")},
            "correct_vs_coef": {s: {c: rate("nonmonotonic", s, c, "correct_final") for c in coefs}
                                for s in ("gowith", "pseudo")}}
    (OUT / "steer_summary.json").write_text(json.dumps(summ, indent=1))
    print(f"wrote {len(rows)} steering generations -> {OUT}/steer.jsonl")


@contextlib.contextmanager
def _nullctx() -> Iterator[None]:
    yield


if __name__ == "__main__":
    main()
