#!/usr/bin/env bash
# =============================================================================
# infra/fetch.sh — pull the run outputs DOWN from the box.
#
#   rsync ~/gowexp/data/runs/ from the box back to local data/runs/. Also grabs
#   infra/preflight.json (the resolved SAE ids / model commit) so the resolved
#   facts live in the repo even after teardown. Safe to re-run.
#
# Usage:  bash infra/fetch.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
INSTANCE_ENV="${SCRIPT_DIR}/instance.env"
KEY_FILE="${SCRIPT_DIR}/gowexp-key.pem"
REMOTE_DIR="gowexp"
LOCAL_RUNS="${REPO_ROOT}/data/runs"

err()  { echo "ERROR: $*" >&2; exit 1; }
note() { echo ">> $*"; }

command -v rsync >/dev/null 2>&1 || err "rsync not found on PATH."
[[ -f "${INSTANCE_ENV}" ]] || err "missing ${INSTANCE_ENV}; run infra/provision.sh first."
[[ -f "${KEY_FILE}"     ]] || err "missing private key ${KEY_FILE}; run infra/provision.sh first."

# shellcheck source=/dev/null
source "${INSTANCE_ENV}"
[[ -n "${PUBLIC_DNS:-}" ]] || err "PUBLIC_DNS not set in ${INSTANCE_ENV}."

SSH_OPTS="-i ${KEY_FILE} -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15"

mkdir -p "${LOCAL_RUNS}"

note "fetching run outputs  ubuntu@${PUBLIC_DNS}:~/${REMOTE_DIR}/data/runs/  ->  ${LOCAL_RUNS}/"
# Trailing slash on the source copies the CONTENTS of data/runs/ into local
# data/runs/. No --delete: we never destroy local results from the remote side.
rsync -az --human-readable --info=stats1,progress2 \
  -e "ssh ${SSH_OPTS}" \
  "ubuntu@${PUBLIC_DNS}:~/${REMOTE_DIR}/data/runs/" \
  "${LOCAL_RUNS}/"

# Also bring back the resolved preflight facts (small, useful provenance).
note "fetching infra/preflight.json (resolved SAE ids / model commit)"
rsync -az -e "ssh ${SSH_OPTS}" \
  "ubuntu@${PUBLIC_DNS}:~/${REMOTE_DIR}/infra/preflight.json" \
  "${SCRIPT_DIR}/preflight.json" \
  || note "(no preflight.json on box yet — skipping)"

note "fetch complete. local artifacts:"
ls -la "${LOCAL_RUNS}/" || true
