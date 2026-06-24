# gowexp — does relational-process register change epistemic self-monitoring, and why?

A mechanistic-interpretability experiment born from a Discord "Bridge" channel
(2026-06-17) where a roomful of humans and models converged on a real, runnable test.

## Headline result (Gemma-3-12B-it, real run)

**Gowith is regime-dependent: a measurable *tax* on crisp reasoning, and (suggestively) a
*tool* on goopy relational reasoning.** Reasoning-in-Gowith (condition D) vs plain
step-by-step English (A), by task:

| task | what it tests | D − A (95% CI) |
|---|---|---|
| nonmonotonic | belief revision | **−0.33** [−0.41, −0.24] |
| agency | responsibility attribution | **−0.12** [−0.17, −0.08] |
| observable | measurable-vs-metaphysical sort | −0.03 [−0.06, −0.01] |
| epistemic-limit | confabulation refusal | −0.07 [−0.17, +0.03] (ns) |
| **correlative** | **cause-vs-correlation, feedback loops** | **+0.12** [−0.05, +0.30] (suggestive, n=40) |

Controls separate the cause: padded-plain (B) ≈ A → **not the tokens**; pseudo-Gowith (E) ≈ A
on crisp tasks → the degradation is **Gowith specifically**, but E ≈ D on correlative → the
*help* is the **relational stance, not the notation**. Output-budget dose-response (O0→O3) is
flat-to-negative → extra reasoning tokens don't buy accuracy here. Mechanistically, the
features Gowith raises on goopy vs crisp tasks overlap only **Jaccard 0.33** — different
features per regime, not one uniform "formal mode." Full write-up: `report/report.pdf`,
live site in `docs/`, channel reply in `report/anma_reply.md`.

---

**The question.** When a relational-process register — **Gowith** (Andy Ayrey's CC0
conlang) — or simply **more reasoning tokens** appears to improve a model's *epistemic
self-monitoring* (tracking uncertainty, refusing to confabulate, revising beliefs
cleanly), *which* cause is responsible:

1. **syntax** — Gowith's relational-process grammar specifically,
2. **token budget** — extra serial-compute room (ember's hypothesis),
3. **weird register** — any novel formalism makes the model slow down,
4. **checklist** — an explicit epistemic instruction does the work,
5. **vibes** — it merely *sounds* more careful.

## Design

**Study 1 — register discrimination.** Six conditions, each isolating one factor, all
ending in the same plain `ANSWER:` line so scoring is style-blind:

| | condition | isolates |
|---|---|---|
| A | concise plain | baseline (no scaffolding) |
| B | padded plain (token-matched to D) | input-token budget |
| C | telegraphic (stripped currents) | openness gestalt without strict syntax |
| D | **full Gowith** | syntax + register + implicit currents |
| E | pseudo-Gowith (scrambled skill, token-matched) | weird register / novelty |
| F | explicit checklist | the epistemic instruction, made plain |

Read-out: `B≈D`→tokens; `E≈D`→register; `F≈D`→checklist; `D>{B,E,F}`→Gowith's grammar
does real work.

**Study 2 — output-budget dose-response.** Plain English, semantics fixed, varying the
model's *own* reasoning budget (O0 answer-only → O3 exhaustive). Accuracy vs realized
output tokens directly tests the "tokens are serial-compute room" claim. Plus a
matched-output-length analysis of Study 1 (does Gowith beat plain *at equal output
length*?).

**Tasks** (all binary-scored — the channel's hard rule: *no vibe-scoring*):
1. **nonmonotonic** belief revision (Tweety→penguin→broken jetpack),
2. **epistemic-limit / confabulation** (questions with no knowable answer),
3. **observable-vs-metaphysical** classification.

**Mechanistic (the WHY).** White-box on **Gemma-3-12B-it** + **Gemma Scope 2** pretrained
SAEs. For each condition we capture the residual stream, encode through the SAE, and ask
which interpretable features (uncertainty / hedging / refusal / formal-text / …) move.
**The pseudo-Gowith control E is the built-in validity check:** if E lights the
"uncertainty" feature as hard as D, the feature tracks *register*, not real uncertainty.
**Causal capstone (steering):** take the features that rise most under Gowith, add them to
the residual stream in the *plain* condition, and test whether the behavioral effect
(confabulation drop, clean updates) reproduces.

**Cross-family replication.** The behavioral grid is also run black-box across model
families via AWS Bedrock (Claude / Nova / Mistral) to check the effect generalizes.

## Reproducibility

- `config/experiment.yaml` is the single source of truth: pinned model + SAE revisions,
  seeds, sampling, item counts.
- Inputs are frozen: `data/items.jsonl` (task items) and `data/prompts.jsonl` (rendered).
- `uv` for the env; torch pinned in `infra/setup_remote.sh` against the CUDA wheel index.
- One command per stage via the `Justfile`. The PDF (`report/report.typ`) and the GitHub
  Pages site (`docs/`) both render from the *same* `report/data/results.json`.

```
just items     # build frozen items
just provision # rent the g6e.xlarge, sync, preflight
# (on box) just render && just run-white && just steer
just fetch     # pull generations + feature summaries + steering
just run-black # cross-family replication (local, Bedrock)
just score && just analyze
just report && just site
just teardown  # stop the instance
```

## Budget

$30 cap. `g6e.xlarge` (L40S 48 GB) at ~$1.86/hr; the run is a few GPU-hours (~$10).
`infra/` enforces an auto-shutdown guard on the box.

## Provenance & care

This experiment comes out of a community that treats model welfare and the transcript
itself as precious. The source conversation and the canonical Gowith doc live under
`data/source/`. Gowith is CC0 by Andy Ayrey & GPT-5.5.
