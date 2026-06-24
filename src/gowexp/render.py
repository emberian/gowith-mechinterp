"""Render every item × condition into data/prompts.jsonl using the model tokenizer.

Runs on the GPU box (the Gemma tokenizer is gated). B/E length-matching is exact here
because we use the real tokenizer. Output is a frozen artifact of the run.
"""
from __future__ import annotations

from pathlib import Path

from . import conditions as C
from .items import build_all
from .schema import Item, RenderedPrompt, load_config, read_jsonl, write_jsonl

_REPO = Path(__file__).resolve().parents[2]


def _load_items() -> list[Item]:
    f = _REPO / "data" / "items.jsonl"
    if f.exists():
        return [Item.from_dict(d) for d in read_jsonl(f)]
    return build_all()


def all_conditions(cfg: dict) -> list[str]:
    register = list(cfg["conditions"]["order"])          # A..F
    budget = list(cfg["output_budget"]["levels"].keys())  # O0..O3
    return register + budget


def main() -> None:
    from transformers import AutoTokenizer

    cfg = load_config()
    # A real render precedes a real run; clear any synthetic-data sentinel.
    (_REPO / "data" / "runs" / "MOCK").unlink(missing_ok=True)
    model_id = cfg["white_box"]["model_id"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=cfg["white_box"]["model_revision"])
    tok = lambda s: tokenizer.encode(s, add_special_tokens=False)  # noqa: E731

    items = _load_items()
    conds = all_conditions(cfg)
    rows: list[RenderedPrompt] = []
    for it in items:
        for cond in conds:
            cp = C.render(it, cond, tok)
            n = len(tok(cp.system)) + len(tok(cp.user))
            rows.append(RenderedPrompt(
                item_id=it.id, task=it.task, condition=cond,
                system=cp.system, user=cp.user, n_input_tokens=n,
            ))
    out = _REPO / "data" / "prompts.jsonl"
    write_jsonl(out, rows)
    print(f"rendered {len(rows)} prompts ({len(items)} items × {len(conds)} conditions) -> {out}")
    # report B/E vs D match per task as a sanity line
    by = {}
    for r in rows:
        by.setdefault((r.task, r.condition), []).append(r.n_input_tokens)
    import statistics
    for task in sorted({r.task for r in rows}):
        d = statistics.mean(by[(task, "D")])
        b = statistics.mean(by[(task, "B")])
        e = statistics.mean(by[(task, "E")])
        print(f"  {task}: D={d:.0f} B={b:.0f} E={e:.0f} "
              f"(B/D={b/d:.3f} E/D={e/d:.3f})")


if __name__ == "__main__":
    main()
