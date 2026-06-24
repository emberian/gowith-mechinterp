"""Gemma Scope 2 SAE loading (via SAELens), encoding, and steering vectors.

Gemma Scope 2 residual SAEs are JumpReLU SAEs. We load with SAELens, encode captured
residuals to feature activations, and expose each feature's decoder direction (W_dec[f])
for the causal steering capstone.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

_REPO = Path(__file__).resolve().parents[2]


def resolve_sae_id(cfg: dict[str, Any], layer: int) -> str:
    """Prefer the exact id discovered by infra/preflight.py; else build from config."""
    pf = _REPO / "infra" / "preflight.json"
    if pf.exists():
        data = json.loads(pf.read_text())
        ids = data.get("sae_ids", {})
        if str(layer) in ids:
            return ids[str(layer)]
    w = cfg["white_box"]
    return f"layer_{layer}_width_{w['sae_width']}_l0_{w['sae_l0']}"


def load_sae(release: str, sae_id: str, device: str = "cuda", dtype=torch.float32):
    """Load a single SAELens SAE. Handles the 1- and 3-tuple return signatures."""
    from sae_lens import SAE

    res = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
    sae = res[0] if isinstance(res, tuple) else res
    sae = sae.to(device=device, dtype=dtype)
    sae.eval()
    return sae


@torch.no_grad()
def encode(sae: Any, acts: torch.Tensor) -> torch.Tensor:
    """Residual activations [..., d_model] -> feature activations [..., d_sae]."""
    x = acts.to(device=next(sae.parameters()).device, dtype=next(sae.parameters()).dtype)
    return sae.encode(x)


def d_sae(sae: Any) -> int:
    return int(sae.cfg.d_sae)


def decoder_direction(sae: Any, feature: int) -> torch.Tensor:
    """Unit residual-space direction for a feature (W_dec[f]). Used for steering."""
    w = sae.W_dec[feature]
    return w / (w.norm() + 1e-8)
