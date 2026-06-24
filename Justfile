# Gowith mechinterp experiment — orchestration.
# `just` is the entrypoint; every step is reproducible from frozen config + items.

set shell := ["bash", "-uc"]
export PYTHONPATH := "src"

default:
    @just --list

# --- local: author + render -------------------------------------------------

# Build the frozen base items (data/items.jsonl) from the task modules.
items:
    python -m gowexp.items

# Render every item × condition into prompts (needs the model tokenizer).
render:
    python -m gowexp.render

# Sanity-check conditions render and B/E match D (mock tokenizer, no ML deps).
smoke:
    python -m gowexp.selftest

# --- GPU box: the white-box run --------------------------------------------

# Generations + SAE feature capture across all conditions (run ON the box).
run-white:
    python -m gowexp.run_white

# Causal steering capstone (run ON the box, after run-white).
steer:
    python -m gowexp.steer

# --- local: cross-family replication + scoring + analysis ------------------

# Behavioral replication across Bedrock model families.
run-black:
    python -m gowexp.run_black

# Apply style-blind binary scorers (+ optional judge panel).
score:
    python -m gowexp.score

# Stats, contrasts, figures -> report/data/results.json + report/figs/*.
analyze:
    python -m gowexp.analyze

# --- report + site ----------------------------------------------------------

report:
    typst compile report/report.typ report/report.pdf
    @echo "wrote report/report.pdf"

site:
    python -m gowexp.build_site
    @echo "wrote docs/ (GitHub Pages)"

# Everything that can run without the GPU box.
local-all: items score analyze report site

# --- AWS infra --------------------------------------------------------------

provision:
    bash infra/provision.sh

fetch:
    bash infra/fetch.sh

teardown:
    bash infra/teardown.sh
