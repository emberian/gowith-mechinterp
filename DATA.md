# Data

All artifacts from the real run are committed (no GPU needed to replay the analysis).

## What's here

| path | what | size |
|---|---|---|
| `data/items.jsonl` | the 360 frozen task items (5 families), register-neutral | ~0.2 MB |
| `data/runs/white/generations.jsonl` | 3600 Gemma-3-12B-it greedy generations (360 items × 10 conditions) | ~4 MB |
| `data/runs/white/sae_means_L{12,24,31,41}.npz` | **the rented-GPU output** — per-generation mean Gemma Scope 2 SAE feature activations (answer span), one row per generation, d_sae=65536, float16 | ~160 MB |
| `data/runs/white/feature_summary.json` | per-condition mean feature vectors + top differential features | — |
| `data/runs/white/steer.jsonl`, `steer_summary.json` | causal steering pass (inject Gowith-up features into the plain prompt) | — |
| `data/runs/white/row_index.json` | maps npz row → (item_id, condition, task) | — |
| `data/runs/scored.jsonl` | every generation + its style-blind binary scores (and judge fields for correlative) | ~5 MB |
| `data/runs/judge_cache.jsonl` | cached register-blind judge-panel verdicts (re-score correlative for **free**) | ~1.5 MB |
| `report/data/results.json` | the single source of truth: contrasts, CIs, by-arm rollup, mechanistic, steering. Both the PDF and the site render from this. | — |

The SAE `.npz` arrays are the **scientific artifact that cost the GPU rental** — they are *not* cheaply regenerable (re-renting + re-running the white-box pass). They're committed deliberately so others can probe the activations without spending anything.

## Replay

```bash
uv sync
PYTHONPATH=src python -m gowexp.analyze     # rebuild report/data/results.json + figures
typst compile report/report.typ report/report.pdf
PYTHONPATH=src python -m gowexp.build_site  # rebuild docs/
```

Re-judging the correlative task hits `judge_cache.jsonl` and costs nothing. Re-running the
white-box generations or SAE capture needs the GPU (`infra/` provisions it).

## Schema

`generations.jsonl` / `scored.jsonl` rows are `gowexp.schema.Generation` / `ScoredRecord`
(see `src/gowexp/schema.py`). Conditions: `A` concise · `B` padded-plain · `C` telegraphic ·
`D` full-Gowith · `E` pseudo-Gowith · `F` checklist · `O0..O3` output-budget series.
Tasks: `nonmonotonic`, `epistemic`, `observable` (degradation arm); `agency`, `correlative`
(in-domain arm).
