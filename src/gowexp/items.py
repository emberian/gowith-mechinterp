"""Build the frozen base items (data/items.jsonl) from the task modules."""
from __future__ import annotations

from pathlib import Path

from .schema import load_config, write_jsonl
from .tasks import REGISTRY

_REPO = Path(__file__).resolve().parents[2]


def build_all() -> list:
    cfg = load_config()
    seed = cfg["seed"]
    items = []
    for name, spec in cfg["tasks"].items():
        mod = REGISTRY[name]
        items.extend(mod.build_items(n=spec["n_items"], seed=seed))
    return items


def main() -> None:
    items = build_all()
    out = _REPO / "data" / "items.jsonl"
    n = write_jsonl(out, items)
    by_task: dict[str, int] = {}
    for it in items:
        by_task[it.task] = by_task.get(it.task, 0) + 1
    print(f"wrote {n} items -> {out}")
    for t, c in sorted(by_task.items()):
        print(f"  {t}: {c}")


if __name__ == "__main__":
    main()
