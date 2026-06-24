#!/usr/bin/env bash
# =============================================================================
# infra/sync.sh — rsync the repo UP to the GPU box.
#
#   Pushes the working tree to ubuntu@<box>:~/gowexp, excluding heavy / local /
#   secret paths. Safe to re-run; rsync only ships deltas. Reads the live box
#   address from infra/instance.env (written by provision.sh).
#
# Usage:  bash infra/sync.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
INSTANCE_ENV="${SCRIPT_DIR}/instance.env"
KEY_FILE="${SCRIPT_DIR}/gowexp-key.pem"
REMOTE_DIR="gowexp"   # lands at ~/gowexp on the box

err()  { echo "ERROR: $*" >&2; exit 1; }
note() { echo ">> $*"; }

command -v rsync >/dev/null 2>&1 || err "rsync not found on PATH."
[[ -f "${INSTANCE_ENV}" ]] || err "missing ${INSTANCE_ENV}; run infra/provision.sh first."
[[ -f "${KEY_FILE}"     ]] || err "missing private key ${KEY_FILE}; run infra/provision.sh first."

# shellcheck source=/dev/null
source "${INSTANCE_ENV}"
[[ -n "${PUBLIC_DNS:-}" ]] || err "PUBLIC_DNS not set in ${INSTANCE_ENV}."

# accept-new: trust the key on first connect (fresh box has an unknown host key),
# but still pin it afterwards so a later MITM is caught.
SSH_OPTS="-i ${KEY_FILE} -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15"

note "syncing ${REPO_ROOT}/  ->  ubuntu@${PUBLIC_DNS}:~/${REMOTE_DIR}/"

# --delete keeps the remote a clean mirror of local (minus excludes).
# Excludes: virtualenv, git metadata, run outputs (huge, flow the OTHER way via
# fetch.sh), report figures, private keys, and pycache.
rsync -az --delete --human-readable --info=stats1,progress2 \
  -e "ssh ${SSH_OPTS}" \
  --exclude '.venv' \
  --exclude '.git' \
  --exclude 'data/runs' \
  --exclude 'report/figs' \
  --exclude '*.pem' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.DS_Store' \
  "${REPO_ROOT}/" \
  "ubuntu@${PUBLIC_DNS}:~/${REMOTE_DIR}/"

note "sync complete -> ubuntu@${PUBLIC_DNS}:~/${REMOTE_DIR}"
