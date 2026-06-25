"""Behavioral run for alternative-stance conditions (G–J) vs A/D, on the box.

Tests two questions at once:
  * ember: can a *different* prompting idea match Gowith on the goopy task without the
    crisp-task tax? (G consider-opposite, H causal-DAG, I systems-persona, J calibration)
  * codex's alternative: is the correlative "win" just rubric-keyword prompting? If G/H/I
    also score high AND say "feedback loop", it's keyword-prompting, not Gowith.

Behavioral only (no SAE capture). Writes data/runs/alt/generations.jsonl; score locally
with the robust parser (nonmonotonic) + the register-blind judge panel (correlative).
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from . import conditions as C
from .model import generate_batch, load_lm, set_determinism
from .schema import Generation, Item, load_config, read_jsonl

_REPO = Path(__file__).resolve().parents[2]
OUT = _REPO / "data" / "runs" / "alt"


def main() -> None:
    cfg = load_config()
    wb = cfg["white_box"]
    set_determinism(cfg["seed"])
    conds = os.environ.get("GOWEXP_ALT_CONDS", "A,D,G,H,I,J").split(",")
    tasks = os.environ.get("GOWEXP_ALT_TASKS", "correlative,nonmonotonic").split(",")
    nm_cap = int(os.environ.get("GOWEXP_ALT_NM", "48"))
    MAXB = int(os.environ.get("GOWEXP_BATCH", "12"))
    max_new = int(wb["max_new_tokens"])

    items = [Item.from_dict(d) for d in read_jsonl(_REPO / "data" / "items.jsonl")]
    by_task: dict[str, list[Item]] = defaultdict(list)
    for it in items:
        by_task[it.task].append(it)
    sample = {t: (by_task[t][:nm_cap] if t == "nonmonotonic" else by_task[t]) for t in tasks}

    lm = load_lm(wb["model_id"], dtype=wb["dtype"], revision=wb["model_revision"])
    tok = lambda s: lm.tokenizer.encode(s, add_special_tokens=False)  # noqa: E731

    OUT.mkdir(parents=True, exist_ok=True)
    gen_path = OUT / "generations.jsonl"
    done = set()
    if gen_path.exists():
        for d in read_jsonl(gen_path):
            done.add((d["item_id"], d["condition"]))
    gf = open(gen_path, "a")

    for task in tasks:
        for cond in conds:
            todo = [it for it in sample[task] if (it.id, cond) not in done]
            for s0 in tqdm(range(0, len(todo), MAXB), desc=f"alt:{task}:{cond}"):
                batch = todo[s0:s0 + MAXB]
                prompts = [(cp.system, cp.user) for cp in (C.render(it, cond, tok) for it in batch)]
                res = generate_batch(lm, prompts, [], max_new)  # layers=[] -> behavioral only
                for it, r in zip(batch, res):
                    g = Generation(item_id=it.id, task=task, condition=cond,
                                   model=wb["model_id"], sample_idx=0, decode="greedy",
                                   text=r["completion"], n_input_tokens=r["n_in"],
                                   n_output_tokens=r["n_out"])
                    gf.write(json.dumps(g.to_dict(), ensure_ascii=False) + "\n")
                gf.flush()
    gf.close()
    n = sum(1 for _ in read_jsonl(gen_path))
    print(f"wrote {n} alt generations -> {gen_path}")


if __name__ == "__main__":
    main()
