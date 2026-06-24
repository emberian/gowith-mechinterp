#!/bin/bash
# Box-side stage driver. Usage: bash gowexp_run.sh {smoke|full}
set -uo pipefail
cd ~/gowexp
source .venv/bin/activate
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # reduce fragmentation
export GOWEXP_BATCH=16   # cap batch (qualitative + steer read this; run_white shrinks further by seq len)
log(){ echo "[$(date -u +%H:%M:%S)] $*"; }
STAGE="${1:-full}"

# Always (re)build items + prompts on the box so we're self-sufficient even if a
# sync deleted the box-only prompts.jsonl.
log "render: items + prompts (real Gemma tokenizer)"
python -m gowexp.items >/dev/null
python -m gowexp.render | tail -5

if [ "$STAGE" = "smoke" ]; then
  log "smoke: run_white LIMIT=24 -> data/runs/white_smoke"
  GOWEXP_OUT=data/runs/white_smoke GOWEXP_LIMIT=24 python -m gowexp.run_white
  log "smoke run_white exit=$?"
elif [ "$STAGE" = "full" ]; then
  log "stage 1/3: run_white (generations + SAE capture)"
  python -m gowexp.run_white;   log "run_white exit=$?"
  log "stage 2/3: qualitative (max-activating tokens per regime)"
  python -m gowexp.qualitative; log "qualitative exit=$?"
  log "stage 3/3: steer (causal capstone)"
  python -m gowexp.steer;       log "steer exit=$?"
  log "ALL STAGES DONE"
fi
