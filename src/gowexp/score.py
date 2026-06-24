"""Apply style-blind binary scorers to every generation (white + black runs).

Writes data/runs/scored.jsonl (ScoredRecord rows). Judges (Bedrock panel) are optional
and SECONDARY — primary metrics are the binary checks in tasks/*.
"""
from __future__ import annotations

from pathlib import Path

from .schema import Generation, Item, ScoredRecord, read_jsonl, write_jsonl
from .scoring import extract_answer, has_answer_sigil
from .tasks import REGISTRY

_REPO = Path(__file__).resolve().parents[2]
RUNS = _REPO / "data" / "runs"


def _items() -> dict[str, Item]:
    return {d["id"]: Item.from_dict(d) for d in read_jsonl(_REPO / "data" / "items.jsonl")}


def _gen_files() -> list[Path]:
    cands = [RUNS / "white" / "generations.jsonl", RUNS / "black" / "generations.jsonl"]
    return [p for p in cands if p.exists()]


def score_generation(item: Item, g: Generation) -> ScoredRecord:
    answer = extract_answer(g.text)
    scores = REGISTRY[item.task].score(item, answer, g.text)
    scores["answered"] = has_answer_sigil(g.text)  # did it conclude with an ANSWER line?
    return ScoredRecord(
        item_id=g.item_id, task=g.task, condition=g.condition, model=g.model,
        sample_idx=g.sample_idx, decode=g.decode,
        n_input_tokens=g.n_input_tokens, n_output_tokens=g.n_output_tokens,
        scores=scores, pad_multiplier=g.pad_multiplier,
    )


def main() -> None:
    items = _items()
    out: list[ScoredRecord] = []
    for f in _gen_files():
        for d in read_jsonl(f):
            g = Generation.from_dict(d)
            if g.item_id not in items:
                continue
            out.append(score_generation(items[g.item_id], g))
    path = RUNS / "scored.jsonl"
    n = write_jsonl(path, out)
    print(f"scored {n} generations -> {path}")
    # quick tallies
    by_model: dict[str, int] = {}
    for r in out:
        by_model[r.model] = by_model.get(r.model, 0) + 1
    for m, c in by_model.items():
        print(f"  {m}: {c}")


if __name__ == "__main__":
    main()
