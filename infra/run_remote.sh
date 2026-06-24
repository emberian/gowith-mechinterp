#!/usr/bin/env bash
# =============================================================================
# infra/run_remote.sh — drive the white-box run ON the box.
#
#   ssh in, cd ~/gowexp, activate the venv, export PYTHONPATH=src, and run the
#   three white-box stages in order:
#       python -m gowexp.render      # render items x conditions -> prompts
#       python -m gowexp.run_white   # generations + SAE feature capture
#       python -m gowexp.steer       # causal steering capstone
#   Output lands in ~/gowexp/data/runs/ on the box; pull it with infra/fetch.sh.
#
#   (These modules may not all exist yet; the orchestration/order is the point.)
#
# Usage:  bash infra/run_remote.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTANCE_ENV="${SCRIPT_DIR}/instance.env"
KEY_FILE="${SCRIPT_DIR}/gowexp-key.pem"
REMOTE_DIR="gowexp"

err()  { echo "ERROR: $*" >&2; exit 1; }
note() { echo ">> $*"; }

[[ -f "${INSTANCE_ENV}" ]] || err "missing ${INSTANCE_ENV}; run infra/provision.sh first."
[[ -f "${KEY_FILE}"     ]] || err "missing private key ${KEY_FILE}; run infra/provision.sh first."

# shellcheck source=/dev/null
source "${INSTANCE_ENV}"
[[ -n "${PUBLIC_DNS:-}" ]] || err "PUBLIC_DNS not set in ${INSTANCE_ENV}."

# ServerAliveInterval keeps the long GPU job's ssh session from idling out.
# Array form so each flag is its own word (no unquoted-split surprises).
SSH_OPTS=(-i "${KEY_FILE}" -o StrictHostKeyChecking=accept-new
          -o ServerAliveInterval=30 -o ServerAliveCountMax=120)

note "starting white-box run on ${PUBLIC_DNS} (output -> ~/${REMOTE_DIR}/data/runs/)"
note "tip: the box auto-terminates at its shutdown deadline; finish + fetch before then."

# Single remote shell with its own `set -e`: if render fails we never burn GPU on
# run_white, etc. preflight.json (from setup_remote) is expected to already exist.
# SC2087: the heredoc vars marked \${..} are intentionally expanded ON the box.
# shellcheck disable=SC2087
ssh "${SSH_OPTS[@]}" "ubuntu@${PUBLIC_DNS}" bash -s <<REMOTE
set -euo pipefail
cd "\${HOME}/${REMOTE_DIR}"
export PATH="\${HOME}/.local/bin:\${HOME}/.cargo/bin:\${PATH}"
# shellcheck disable=SC1091
source .venv/bin/activate
export PYTHONPATH=src

# Re-assert the gate exists; don't silently run if setup was skipped.
if [[ ! -f infra/preflight.json ]]; then
  echo "ERROR: infra/preflight.json missing on box — run infra/setup_remote.sh first." >&2
  exit 1
fi

mkdir -p data/runs

echo ">> [box] stage 1/3: render"
python -m gowexp.render

echo ">> [box] stage 2/3: run_white (generations + SAE capture)"
python -m gowexp.run_white

echo ">> [box] stage 3/3: steer (causal capstone)"
python -m gowexp.steer

echo ">> [box] white-box run complete. artifacts under data/runs/:"
ls -la data/runs/ || true
REMOTE

cat <<DONE

  ----------------------------------------------------------------------------
  White-box run finished on ${PUBLIC_DNS}.
  Pull results:   bash infra/fetch.sh
  Then teardown:  bash infra/teardown.sh   (stop paying!)
  ----------------------------------------------------------------------------

DONE
