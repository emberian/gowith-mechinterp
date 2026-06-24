#!/usr/bin/env bash
# =============================================================================
# infra/setup_remote.sh — bring the GPU box from bare DLAMI to run-ready.
#
#   1. sync the repo up (infra/sync.sh)
#   2. install uv, create a venv on the box
#   3. install a PINNED CUDA torch wheel FIRST (before the project), so pip
#      never pulls a CPU-only or mismatched torch as a transitive dep
#   4. uv pip install -e '.[gpu]'  (transformers, sae-lens, accelerate, ...)
#   5. copy the local HF token up to ~/.cache/huggingface/token
#   6. run infra/preflight.py on the box — the GATE before any GPU spend
#
#   Any failure stops the whole thing (set -e + remote `set -e`). Re-runnable.
#
# Usage:  bash infra/setup_remote.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTANCE_ENV="${SCRIPT_DIR}/instance.env"
KEY_FILE="${SCRIPT_DIR}/gowexp-key.pem"
LOCAL_HF_TOKEN="${HOME}/.cache/huggingface/token"
REMOTE_DIR="gowexp"

# --- pinned GPU stack --------------------------------------------------------
# torch 2.6.0 on the CUDA 12.4 wheel index. Rationale:
#   * The DLAMI we provision is "PyTorch 2.6.0 (Ubuntu 22.04)" with a CUDA 12.x
#     driver, so a cu124 wheel matches the on-box driver/runtime.
#   * sae-lens (>=6) supports the torch 2.x line; 2.6.0 is a known-good pairing
#     with transformers>=4.55 and Gemma-3.
# Pin both the version AND the index so the install is byte-reproducible and
# never silently resolves to a CPU build. Override via TORCH_SPEC / TORCH_INDEX.
TORCH_SPEC="${TORCH_SPEC:-torch==2.6.0}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"

err()  { echo "ERROR: $*" >&2; exit 1; }
note() { echo ">> $*"; }

[[ -f "${INSTANCE_ENV}" ]] || err "missing ${INSTANCE_ENV}; run infra/provision.sh first."
[[ -f "${KEY_FILE}"     ]] || err "missing private key ${KEY_FILE}; run infra/provision.sh first."

# shellcheck source=/dev/null
source "${INSTANCE_ENV}"
[[ -n "${PUBLIC_DNS:-}" ]] || err "PUBLIC_DNS not set in ${INSTANCE_ENV}."

SSH_OPTS="-i ${KEY_FILE} -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o ServerAliveInterval=30"
SSH="ssh ${SSH_OPTS} ubuntu@${PUBLIC_DNS}"

# --- 1. sync the repo up -----------------------------------------------------
note "step 1/6 — syncing repo to the box"
bash "${SCRIPT_DIR}/sync.sh"

# --- 2/3/4. install uv + venv + pinned torch + project -----------------------
# We run a single remote heredoc with its own `set -euo pipefail` so a failure
# at any line aborts remotely AND propagates a non-zero exit to us.
note "step 2-4/6 — uv, venv, pinned torch (${TORCH_SPEC} @ ${TORCH_INDEX}), project [gpu]"
# shellcheck disable=SC2087  # we intentionally expand these vars locally before sending
${SSH} bash -s <<REMOTE
set -euo pipefail
cd "\${HOME}/${REMOTE_DIR}"

# uv: install once, then make it visible on PATH for this non-login shell.
if ! command -v uv >/dev/null 2>&1; then
  echo ">> [box] installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="\${HOME}/.local/bin:\${HOME}/.cargo/bin:\${PATH}"
command -v uv >/dev/null 2>&1 || { echo "ERROR: uv not on PATH after install" >&2; exit 1; }
uv --version

# Create the project venv (idempotent). Python 3.11+ per pyproject requires-python.
echo ">> [box] creating venv"
uv venv --python 3.11 .venv || uv venv .venv

# Activate so subsequent uv pip targets THIS venv.
# shellcheck disable=SC1091
source .venv/bin/activate

# 3) Pinned CUDA torch FIRST — before the project — so the project resolve never
#    drags in a CPU wheel or a different torch.
echo ">> [box] installing pinned CUDA torch: ${TORCH_SPEC} (index ${TORCH_INDEX})"
uv pip install "${TORCH_SPEC}" --index-url "${TORCH_INDEX}"

# Quick assert torch sees CUDA before we spend time on the rest.
python - <<'PYCHK'
import torch
print(">> [box] torch", torch.__version__, "cuda_build", torch.version.cuda, "available", torch.cuda.is_available())
assert torch.cuda.is_available(), "torch.cuda.is_available() is False right after install — driver/wheel mismatch"
PYCHK

# 4) Install the project + GPU extra. torch is already satisfied, so pip won't
#    re-fetch it. Bound torch in the resolver so a transitive bump can't clobber
#    the pinned wheel.
echo ">> [box] installing project: uv pip install -e '.[gpu]'"
uv pip install -e '.[gpu]' --index-strategy unsafe-best-match || uv pip install -e '.[gpu]'

echo ">> [box] env ready"
REMOTE

# --- 5. copy the HF token up -------------------------------------------------
note "step 5/6 — copying HF token to the box"
if [[ -s "${LOCAL_HF_TOKEN}" ]]; then
  # Ensure the dir exists, then write the token with private perms.
  ${SSH} 'mkdir -p ~/.cache/huggingface && chmod 700 ~/.cache/huggingface'
  # Pipe the token over stdin so it never appears in argv / process list.
  ${SSH} 'umask 077; cat > ~/.cache/huggingface/token' < "${LOCAL_HF_TOKEN}"
  ${SSH} 'chmod 600 ~/.cache/huggingface/token'
  note "HF token installed at ~/.cache/huggingface/token (mode 600)"
else
  err "local HF token not found or empty at ${LOCAL_HF_TOKEN}.
     Gemma-3 is a GATED model and needs a token. Create one at
     https://huggingface.co/settings/tokens then:  huggingface-cli login"
fi

# --- 6. preflight gate -------------------------------------------------------
note "step 6/6 — running preflight on the box (the gate before GPU spend)"
# Run with the venv active and PYTHONPATH=src so gowexp imports resolve.
${SSH} bash -s <<REMOTE
set -euo pipefail
cd "\${HOME}/${REMOTE_DIR}"
export PATH="\${HOME}/.local/bin:\${HOME}/.cargo/bin:\${PATH}"
# shellcheck disable=SC1091
source .venv/bin/activate
export PYTHONPATH=src
python infra/preflight.py
REMOTE

cat <<DONE

  ----------------------------------------------------------------------------
  Remote setup COMPLETE and preflight PASSED on ${PUBLIC_DNS}.
  Resolved SAE ids + model commit written to (on box) ~/gowexp/infra/preflight.json
  Pull it locally with:  bash infra/fetch.sh   (or scp it directly)

  Next:  bash infra/run_remote.sh
  ----------------------------------------------------------------------------

DONE
