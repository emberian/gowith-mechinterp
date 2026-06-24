"""Cross-family behavioral replication via AWS Bedrock (Study 1, black-box arm).

Runs the six register conditions (A–F) of the behavioral grid through several
Bedrock model families (Anthropic Claude, Amazon Nova, Mistral) to test whether
the register effect generalizes beyond the white-box Gemma model. Outputs land in
the repo's :class:`~gowexp.schema.Generation` schema so the existing scorer and
analysis consume them unchanged.

    PYTHONPATH=src python -m gowexp.run_black            # SMOKE by default (cheap)
    PYTHONPATH=src python -m gowexp.run_black --full     # the full grid (costs $)
    PYTHONPATH=src python -m gowexp.run_black --full --limit 8   # 8 items/task

Safety rails:
  * Defaults to ``--smoke`` (2 items/task, 1 sample): an accidental run is cheap.
  * Always prints a token + dollar COST ESTIMATE before issuing any request; the
    full grid additionally requires the ``--full`` flag (not just absence of smoke).
  * Resumable: existing (item_id, condition, model, sample_idx) rows are skipped,
    so an interrupted run picks up where it left off and never double-bills.

NOTE on length-matching: conditions B and E are padded to D's input-token length
using an UNGATED proxy tokenizer (GPT-2), since the Gemma tokenizer is gated and
this is a robustness check, not the primary measurement. The pad targets are thus
approximate for the actual Bedrock tokenizers — acceptable here because the B≈D /
E≈D contrast only needs to hold *within* the proxy, and the binary scorer is
style-blind regardless.
"""
from __future__ import annotations

import argparse
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from tqdm import tqdm

from . import conditions as C
from .bedrock import chat
from .items import build_all
from .schema import Generation, Item, load_config, read_jsonl, write_jsonl

_REPO = Path(__file__).resolve().parents[2]
_ITEMS_PATH = _REPO / "data" / "items.jsonl"
_OUT_PATH = _REPO / "data" / "runs" / "black" / "generations.jsonl"

# The six register conditions of Study 1 (in canonical order). B and E need the tokenizer.
CONDITIONS = ["A", "B", "C", "D", "E", "F"]

# ----------------------------------------------------------------------------
# Model ids: config holds the bare model ids (source of truth), but on-demand
# Converse invocation of the newer Nova / Claude models requires a cross-region
# *inference profile* id (us.<...>); Mistral is invoked by its bare id. We map at
# call time and keep the config's bare id in the `model` field of each output row
# so downstream analysis groups by the canonical id.
# ----------------------------------------------------------------------------
_INFERENCE_PROFILE = {
    "amazon.nova-2-lite-v1:0": "us.amazon.nova-2-lite-v1:0",
    "anthropic.claude-haiku-4-5-20251001-v1:0": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
}


def invocation_id(model_id: str) -> str:
    """Map a config model id to the id actually passed to Converse."""
    return _INFERENCE_PROFILE.get(model_id, model_id)


def family_of(model_id: str) -> str:
    """anthropic | amazon | mistral, derived from the model id prefix."""
    head = model_id.split(".", 1)[0]
    # tolerate inference-profile prefixes like "us.anthropic.<...>"
    if head in {"us", "global", "eu", "apac"}:
        head = model_id.split(".", 2)[1]
    return head


# ----------------------------------------------------------------------------
# Rough Bedrock on-demand prices, USD per 1,000,000 tokens (input, output).
# Verify against https://aws.amazon.com/bedrock/pricing/ — these are for the cost
# *estimate* only and never affect what is billed. (Fetched 2026-06.)
# ----------------------------------------------------------------------------
_PRICE_PER_M: dict[str, tuple[float, float]] = {
    "anthropic.claude-haiku-4-5-20251001-v1:0": (1.00, 5.00),
    "amazon.nova-2-lite-v1:0": (0.06, 0.24),
    "mistral.ministral-3-8b-instruct": (0.15, 0.15),
}
_DEFAULT_PRICE = (1.00, 5.00)  # conservative fallback if a model id is unpriced


# ----------------------------------------------------------------------------
# Tokenizer proxy for B/E length-matching (ungated GPT-2; see module docstring).
# ----------------------------------------------------------------------------
def _proxy_tok() -> Callable[[str], list[int]]:
    from transformers import AutoTokenizer

    gpt2 = AutoTokenizer.from_pretrained("gpt2")
    return lambda s: gpt2.encode(s)


# ----------------------------------------------------------------------------
# Items
# ----------------------------------------------------------------------------
def load_items() -> list[Item]:
    if _ITEMS_PATH.exists():
        return [Item.from_dict(d) for d in read_jsonl(_ITEMS_PATH)]
    return build_all()


def _limit_items(items: list[Item], limit: int | None) -> list[Item]:
    """Cap items *per task* (so a smoke run still touches every task family)."""
    if limit is None:
        return items
    seen: dict[str, int] = {}
    out: list[Item] = []
    for it in items:
        if seen.get(it.task, 0) < limit:
            out.append(it)
            seen[it.task] = seen.get(it.task, 0) + 1
    return out


# ----------------------------------------------------------------------------
# The unit of work
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Cell:
    item: Item
    condition: str
    model_id: str       # config (canonical) id, written to the row
    sample_idx: int


def _key(item_id: str, condition: str, model_id: str, sample_idx: int) -> tuple:
    return (item_id, condition, model_id, sample_idx)


def build_cells(
    items: list[Item],
    models: list[str],
    samples_per_cell: int,
    done: set[tuple],
) -> list[Cell]:
    cells: list[Cell] = []
    for model_id in models:
        for it in items:
            for cond in CONDITIONS:
                for s in range(samples_per_cell):
                    if _key(it.id, cond, model_id, s) in done:
                        continue
                    cells.append(Cell(it, cond, model_id, s))
    return cells


# ----------------------------------------------------------------------------
# Cost estimate (before any request fires)
# ----------------------------------------------------------------------------
def estimate_cost(
    cells: list[Cell],
    tok: Callable[[str], list[int]],
    max_tokens: int,
) -> dict[str, Any]:
    """Estimate input/output tokens and USD per model for the pending cells.

    Input tokens come from the actually-rendered prompts (proxy tokenizer).
    Output tokens are assumed to be `max_tokens` per cell — the conservative
    upper bound, since most generations stop earlier.
    """
    per_model: dict[str, dict[str, float]] = {}
    # cache rendered token counts per (item_id, condition) to avoid re-rendering
    render_cache: dict[tuple[str, str], int] = {}

    for cell in cells:
        ckey = (cell.item.id, cell.condition)
        if ckey not in render_cache:
            cp = C.render(cell.item, cell.condition, tok)
            render_cache[ckey] = len(tok(cp.system)) + len(tok(cp.user))
        in_toks = render_cache[ckey]

        m = per_model.setdefault(
            cell.model_id, {"cells": 0, "in_tokens": 0.0, "out_tokens": 0.0, "usd": 0.0}
        )
        m["cells"] += 1
        m["in_tokens"] += in_toks
        m["out_tokens"] += max_tokens
        pin, pout = _PRICE_PER_M.get(cell.model_id, _DEFAULT_PRICE)
        m["usd"] += in_toks / 1e6 * pin + max_tokens / 1e6 * pout

    total = {
        "cells": sum(int(v["cells"]) for v in per_model.values()),
        "in_tokens": sum(v["in_tokens"] for v in per_model.values()),
        "out_tokens": sum(v["out_tokens"] for v in per_model.values()),
        "usd": sum(v["usd"] for v in per_model.values()),
    }
    return {"per_model": per_model, "total": total}


def print_cost(est: dict[str, Any], max_tokens: int) -> None:
    print("\n=== COST ESTIMATE (upper bound: output assumed = max_tokens) ===")
    print(f"  {'model':<48} {'cells':>6} {'in_tok':>10} {'out_tok':>10} {'~USD':>9}")
    for model_id, v in sorted(est["per_model"].items()):
        print(
            f"  {model_id:<48} {int(v['cells']):>6} "
            f"{int(v['in_tokens']):>10} {int(v['out_tokens']):>10} {v['usd']:>9.4f}"
        )
    t = est["total"]
    print(
        f"  {'TOTAL':<48} {int(t['cells']):>6} "
        f"{int(t['in_tokens']):>10} {int(t['out_tokens']):>10} {t['usd']:>9.4f}"
    )
    print(f"  (output token count is a ceiling at max_tokens={max_tokens}; real cost is lower)\n")


# ----------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------
def _run_cell(
    cell: Cell,
    tok: Callable[[str], list[int]],
    max_tokens: int,
    temperature: float,
    region: str,
) -> Generation:
    cp = C.render(cell.item, cell.condition, tok)
    res = chat(
        model_id=invocation_id(cell.model_id),
        system=cp.system,
        user=cp.user,
        max_tokens=max_tokens,
        temperature=temperature,
        region=region,
    )
    return Generation(
        item_id=cell.item.id,
        task=cell.item.task,
        condition=cell.condition,
        model=cell.model_id,
        sample_idx=cell.sample_idx,
        decode="sample",  # every black-box cell is a (temperature) sample
        text=res["text"],
        n_input_tokens=res["in_tokens"],
        n_output_tokens=res["out_tokens"],
        meta={"family": family_of(cell.model_id), "stop_reason": res.get("stop_reason", "")},
    )


def load_done(path: Path) -> set[tuple]:
    """Existing (item_id, condition, model, sample_idx) tuples, for resumability."""
    if not path.exists():
        return set()
    done: set[tuple] = set()
    for d in read_jsonl(path):
        done.add(_key(d["item_id"], d["condition"], d["model"], d["sample_idx"]))
    return done


def _append_jsonl(path: Path, gen: Generation, lock: threading.Lock) -> None:
    """Append one row, holding a lock so concurrent workers don't interleave lines."""
    import json

    line = json.dumps(gen.to_dict(), ensure_ascii=False)
    with lock:
        with open(path, "a") as f:
            f.write(line + "\n")


def run(
    cells: list[Cell],
    tok: Callable[[str], list[int]],
    *,
    max_tokens: int,
    temperature: float,
    region: str,
    workers: int,
    out_path: Path,
) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    n_ok = 0
    n_err = 0

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(_run_cell, cell, tok, max_tokens, temperature, region): cell
            for cell in cells
        }
        for fut in tqdm(as_completed(futs), total=len(futs), desc="bedrock", unit="gen"):
            cell = futs[fut]
            try:
                gen = fut.result()
            except Exception as e:  # noqa: BLE001 — log + continue so one bad cell doesn't sink the run
                n_err += 1
                tqdm.write(
                    f"[ERR] {cell.model_id} {cell.item.id} {cell.condition} "
                    f"s{cell.sample_idx}: {type(e).__name__}: {e}"
                )
                continue
            _append_jsonl(out_path, gen, lock)
            n_ok += 1

    if n_err:
        print(f"\n{n_err} cell(s) errored (left for a resume run).")
    return n_ok


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--smoke",
        dest="smoke",
        action="store_true",
        help="cheap run: 2 items/task, 1 sample (DEFAULT).",
    )
    mode.add_argument(
        "--full",
        dest="full",
        action="store_true",
        help="the full grid: all items, samples_per_cell from config. Costs money.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap items PER TASK (cheap subset). Default: smoke=2, full=all.",
    )
    p.add_argument(
        "--samples",
        type=int,
        default=None,
        help="override samples_per_cell (default: smoke=1, full=config value).",
    )
    p.add_argument(
        "--models",
        type=str,
        default=None,
        help="comma-separated model ids to override config black_box.models.",
    )
    p.add_argument("--workers", type=int, default=8, help="ThreadPoolExecutor size (default 8).")
    p.add_argument(
        "--yes",
        action="store_true",
        help="skip the interactive confirm (for non-interactive / scripted runs).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="print the cost estimate and exit without issuing any request.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # Default to smoke unless --full was explicitly chosen.
    smoke = not args.full

    cfg = load_config()
    bb = cfg["black_box"]
    region: str = bb["region"]
    max_tokens: int = int(bb["max_tokens"])
    temperature: float = float(bb["temperature"])

    if args.models:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        models = list(bb["models"])

    # Resolve grid size from mode + overrides.
    if smoke:
        limit = args.limit if args.limit is not None else 2
        samples = args.samples if args.samples is not None else 1
    else:
        limit = args.limit  # None => all items
        samples = args.samples if args.samples is not None else int(bb["samples_per_cell"])

    items = _limit_items(load_items(), limit)

    print(f"mode={'SMOKE' if smoke else 'FULL'}  region={region}  models={len(models)}")
    print(f"items={len(items)} (limit/task={limit})  conditions={len(CONDITIONS)}  samples={samples}")
    print(f"  -> {len(models) * len(items) * len(CONDITIONS) * samples} cells before resume-skip")

    tok = _proxy_tok()

    out_path = _OUT_PATH
    done = load_done(out_path)
    if done:
        print(f"resume: {len(done)} existing rows in {out_path} will be skipped")

    cells = build_cells(items, models, samples, done)
    if not cells:
        print("nothing to do (all requested cells already present). done.")
        return 0

    est = estimate_cost(cells, tok, max_tokens)
    print_cost(est, max_tokens)

    if args.dry_run:
        print("--dry-run: not issuing any request.")
        return 0

    # Gate: the full grid must be opted into, and (unless --yes / non-interactive)
    # confirmed, so an accidental `python -m gowexp.run_black --full` still pauses.
    if not smoke and not args.yes:
        if sys.stdin.isatty():
            resp = input(f"Proceed with FULL run (~${est['total']['usd']:.2f})? [y/N] ").strip().lower()
            if resp not in {"y", "yes"}:
                print("aborted.")
                return 1
        else:
            print("FULL run in a non-interactive shell requires --yes. aborting.")
            return 1

    n = run(
        cells,
        tok,
        max_tokens=max_tokens,
        temperature=temperature,
        region=region,
        workers=args.workers,
        out_path=out_path,
    )
    total = len(list(read_jsonl(out_path))) if out_path.exists() else n
    print(f"\nwrote {n} new generations -> {out_path}  ({total} rows total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
