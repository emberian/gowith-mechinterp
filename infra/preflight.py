#!/usr/bin/env python3
# =============================================================================
# infra/preflight.py — the GATE before we spend GPU time. Runs ON the box.
#
# Verifies, defensively and informatively:
#   1. CUDA is live; prints GPU name + total VRAM.
#   2. We can actually reach the GATED model google/gemma-3-12b-it AND the SAE
#      repo google/gemma-scope-2-12b-it (model_info + a tiny config.json
#      download). On 401/403 it prints EXACT remediation and exits non-zero.
#   3. For the configured SAE (release/width/l0), it DISCOVERS the real sae_id
#      strings for every requested layer by listing the repo's resid_post/
#      subfolder — because the config's friendly width "64k" maps to the repo's
#      actual "65k" folder. It prints every available id per layer and the
#      resolved one, and fails if any requested layer can't be resolved.
#   4. Records the model commit sha.
#
# Everything resolved is written to infra/preflight.json next to this file, so
# the downstream run (gowexp.run_white / gowexp.steer) consumes resolved ids
# rather than re-guessing names.
#
# Exit codes: 0 = all gates passed; non-zero = a gate failed (message printed).
#
# Run:  PYTHONPATH=src python infra/preflight.py   (with the [gpu] venv active)
# =============================================================================
from __future__ import annotations

import json
import re
import sys
import traceback
from pathlib import Path

# --- paths -------------------------------------------------------------------
HERE = Path(__file__).resolve().parent           # .../gowexp/infra
REPO = HERE.parent                               # .../gowexp
CONFIG = REPO / "config" / "experiment.yaml"
OUT = HERE / "preflight.json"

MODEL_REPO = "google/gemma-3-12b-it"
SAE_REPO = "google/gemma-scope-2-12b-it"          # the actual HF repo backing the configured release


def die(msg: str, code: int = 1) -> "None":
    """Print a clearly delimited failure and exit non-zero."""
    print("\n" + "=" * 78)
    print("PREFLIGHT FAILED")
    print("-" * 78)
    print(msg.rstrip())
    print("=" * 78)
    sys.exit(code)


def ok(msg: str) -> None:
    print(f"[ok] {msg}")


def info(msg: str) -> None:
    print(f"     {msg}")


def section(msg: str) -> None:
    print(f"\n== {msg} ==")


# --- load config (single source of truth) ------------------------------------
def load_config() -> dict:
    try:
        import yaml
    except Exception as e:  # noqa: BLE001
        die(f"pyyaml is required to read {CONFIG} but failed to import: {e!r}")
    if not CONFIG.exists():
        die(f"config not found: {CONFIG}")
    with CONFIG.open() as f:
        return yaml.safe_load(f)


def norm_width(w: str) -> str:
    """
    Normalize a friendly width like '64k' to the canonical token used in the
    Gemma Scope repo folder names.

    DeepMind's recommended default is labelled '64k' colloquially, but the repo
    publishes it under 'width_65k' (2**16 = 65536 features). We map the common
    near-power-of-two labels onto the real tokens, and otherwise pass through.
    """
    w = str(w).strip().lower()
    alias = {
        "64k": "65k",   # 2**16 = 65536 -> repo uses 65k
        "65536": "65k",
        "16384": "16k",
        "262144": "262k",
        "1048576": "1m",
    }
    return alias.get(w, w)


# --- gate 1: CUDA ------------------------------------------------------------
def check_cuda(result: dict) -> None:
    section("GPU / CUDA")
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        die(
            "could not import torch. The pinned CUDA wheel install in "
            "infra/setup_remote.sh must have failed.\n"
            f"  underlying error: {e!r}"
        )
    if not torch.cuda.is_available():
        die(
            "torch.cuda.is_available() is False.\n"
            "  This box has no usable GPU/driver from torch's view. Likely a\n"
            "  CPU-only torch wheel slipped in, or the NVIDIA driver isn't loaded.\n"
            "  Fix: re-run infra/setup_remote.sh (it pins a cu124 wheel), and on\n"
            "       the box check `nvidia-smi`.\n"
            f"  torch={torch.__version__} cuda_build={torch.version.cuda}"
        )
    name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024**3)
    ok(f"CUDA available — {name}")
    info(f"torch {torch.__version__} (cuda build {torch.version.cuda})")
    info(f"VRAM: {vram_gb:.1f} GB  |  SMs: {props.multi_processor_count}")
    result["gpu"] = {
        "name": name,
        "vram_gb": round(vram_gb, 1),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
    }
    # gemma-3-12b in bf16 is ~24 GB weights; warn (don't fail) if VRAM looks tight.
    if vram_gb < 40:
        info(
            f"WARNING: {vram_gb:.0f} GB VRAM is below the ~48 GB this run assumes "
            "(L40S). The model + SAEs may not fit; expect OOM."
        )


# --- HF helpers --------------------------------------------------------------
def _is_auth_error(exc: Exception) -> bool:
    """Detect 401/403/gated/repo-not-found-due-to-auth from huggingface_hub."""
    txt = f"{type(exc).__name__}: {exc}".lower()
    if any(s in txt for s in ("401", "403", "gated", "unauthorized", "forbidden",
                              "access to model", "awaiting a review", "restricted")):
        return True
    # GatedRepoError / RepositoryNotFoundError live under huggingface_hub.errors
    try:
        from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError  # type: ignore
        if isinstance(exc, (GatedRepoError, RepositoryNotFoundError)):
            return True
    except Exception:  # noqa: BLE001
        pass
    return False


def gating_help(repo: str) -> str:
    return (
        f"cannot access '{repo}' with the token on this box.\n\n"
        f"  If it is GATED, you must accept its license while signed in:\n"
        f"    1. open  https://huggingface.co/{repo}\n"
        f"    2. click 'Agree and access repository' (accept the license)\n"
        f"    3. ensure a READ token is set on this box:\n"
        f"         ~/.cache/huggingface/token   (setup_remote.sh copies it up)\n"
        f"       or:  huggingface-cli login\n"
        f"    4. the HF account that accepted the license MUST be the one whose\n"
        f"       token is on the box.\n\n"
        f"  Verify the token is present:\n"
        f"    test -s ~/.cache/huggingface/token && echo 'token present' || echo 'NO TOKEN'\n"
    )


# --- gate 2 + 4: HF access + model commit ------------------------------------
def check_hf_access(result: dict):
    section("Hugging Face access")
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except Exception as e:  # noqa: BLE001
        die(f"huggingface_hub not importable: {e!r}. Did '.[gpu]' install fail?")

    api = HfApi()

    # -- the gated model -- model_info gives us the pinned commit sha (gate 4).
    try:
        mi = api.model_info(MODEL_REPO)
    except Exception as e:  # noqa: BLE001
        if _is_auth_error(e):
            die(gating_help(MODEL_REPO))
        die(f"unexpected error reading model_info({MODEL_REPO}): {e!r}")
    model_sha = getattr(mi, "sha", None)
    ok(f"reachable: {MODEL_REPO}")
    info(f"resolved commit sha: {model_sha}")
    result["model"] = {"repo": MODEL_REPO, "commit": model_sha,
                       "gated": str(getattr(mi, "gated", None))}

    # Prove we can actually DOWNLOAD (not just read metadata) — pull a tiny file.
    try:
        cfg_path = hf_hub_download(MODEL_REPO, filename="config.json")
        info(f"downloaded {MODEL_REPO}/config.json -> {cfg_path}")
    except Exception as e:  # noqa: BLE001
        if _is_auth_error(e):
            die(gating_help(MODEL_REPO))
        die(f"could not download config.json from {MODEL_REPO}: {e!r}")

    # -- the SAE repo --
    try:
        sae_mi = api.model_info(SAE_REPO)
        ok(f"reachable: {SAE_REPO}")
        info(f"SAE repo commit sha: {getattr(sae_mi, 'sha', None)}")
        result["sae_repo"] = {"repo": SAE_REPO, "commit": getattr(sae_mi, "sha", None)}
    except Exception as e:  # noqa: BLE001
        if _is_auth_error(e):
            die(gating_help(SAE_REPO))
        die(f"unexpected error reading model_info({SAE_REPO}): {e!r}")

    return api


# --- gate 3: discover the real SAE ids ---------------------------------------
def resolve_sae_ids(api, cfg: dict, result: dict) -> None:
    section("SAE id resolution (Gemma Scope 2 resid_post)")
    wb = cfg["white_box"]
    layers = list(wb["sae_layers"])
    width_friendly = str(wb["sae_width"])
    width = norm_width(width_friendly)          # '64k' -> '65k'
    l0 = str(wb["sae_l0"]).strip().lower()      # 'medium'
    hook = "resid_post"

    info(f"requested: layers={layers} width={width_friendly} (->{width}) l0={l0} hook={hook}")
    if width != width_friendly:
        info(f"NOTE: config width '{width_friendly}' maps to repo width token '{width}'.")

    # List the whole repo once; filter to the resid_post/ top-level sae dirs.
    try:
        files = api.list_repo_files(SAE_REPO)
    except Exception as e:  # noqa: BLE001
        if _is_auth_error(e):
            die(gating_help(SAE_REPO))
        die(f"could not list files of {SAE_REPO}: {e!r}")

    # We want EXACTLY the top-level 'resid_post/<sae_id>/...' (not resid_post_all/).
    sae_ids = sorted({
        f.split("/")[1]
        for f in files
        if f.startswith(f"{hook}/") and len(f.split("/")) >= 3
    })

    resolved: dict[str, str] = {}
    missing: list[int] = []
    available_widths: set[str] = set()
    for sid in sae_ids:
        m = re.search(r"width_([0-9a-z]+)_l0", sid)
        if m:
            available_widths.add(m.group(1))

    for layer in layers:
        layer_ids = [sid for sid in sae_ids if sid.startswith(f"layer_{layer}_")]
        # exact match on the canonical naming layer_<L>_width_<W>_l0_<L0>
        want = f"layer_{layer}_width_{width}_l0_{l0}"
        match = want if want in layer_ids else None
        # fall back to a tolerant regex if the exact string isn't found
        if match is None:
            rx = re.compile(rf"^layer_{layer}_width_{re.escape(width)}_l0_{re.escape(l0)}$")
            cands = [sid for sid in layer_ids if rx.match(sid)]
            match = cands[0] if cands else None

        print(f"\n  layer {layer}:")
        if not layer_ids:
            print(f"    (no resid_post SAEs found for layer {layer}!)")
        else:
            # Print everything available for this layer so a human can see options.
            print(f"    available ids ({len(layer_ids)}):")
            for sid in layer_ids:
                marker = "  <== RESOLVED" if sid == match else ""
                print(f"      - {sid}{marker}")
        if match:
            resolved[str(layer)] = match
            ok(f"layer {layer} -> sae_id '{match}'")
        else:
            missing.append(layer)
            print(f"    !! no id matches width={width} l0={l0} for layer {layer}")

    result["sae"] = {
        "repo": SAE_REPO,
        "release": wb["sae_release"],
        "hook": hook,
        "width_config": width_friendly,
        "width_resolved": width,
        "l0": l0,
        "layers": layers,
        "primary_layer": wb.get("sae_primary_layer"),
        "available_widths": sorted(available_widths),
        "ids": resolved,                 # {"24": "layer_24_width_65k_l0_medium", ...}
    }

    if missing:
        die(
            f"could not resolve a sae_id for layer(s) {missing} "
            f"at width={width} l0={l0}.\n"
            f"  widths actually present in {SAE_REPO}: {sorted(available_widths)}\n"
            f"  Fix config/experiment.yaml white_box.sae_width / sae_l0 to a present combo."
        )

    # Cross-check against sae-lens's registry IF it's importable — purely advisory.
    try:
        from sae_lens.loading.pretrained_saes_directory import (  # type: ignore
            get_pretrained_saes_directory,
        )
        directory = get_pretrained_saes_directory()
        rel = wb["sae_release"]
        if rel in directory:
            ok(f"sae-lens registry knows release '{rel}' (loader cross-check passed)")
        else:
            info(
                f"sae-lens registry does not list release '{rel}' under that exact "
                "name; run_white may need to load by repo_id + sae_id directly. "
                "(advisory only — HF ids above are authoritative.)"
            )
    except Exception:  # noqa: BLE001
        info("sae-lens registry cross-check skipped (not importable here) — fine.")


# --- main --------------------------------------------------------------------
def main() -> int:
    print("gowexp preflight — gating GPU spend\n" + "-" * 40)
    cfg = load_config()
    result: dict = {
        "config_path": str(CONFIG),
        "model_repo": MODEL_REPO,
        "sae_repo": SAE_REPO,
    }
    try:
        check_cuda(result)
        api = check_hf_access(result)
        resolve_sae_ids(api, cfg, result)
    except SystemExit:
        # die() already printed; propagate the non-zero exit.
        raise
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        die(f"unexpected preflight error: {e!r}")

    # Persist resolved facts for the downstream run.
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    section("RESULT")
    print(json.dumps(result, indent=2))
    ok(f"wrote {OUT}")
    print("\nPREFLIGHT PASSED — safe to run the white-box stage.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
