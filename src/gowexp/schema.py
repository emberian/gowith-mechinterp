"""Core data types and (de)serialization for the experiment.

Everything that crosses a process boundary (laptop -> GPU box -> back) is a plain
dict on disk as JSONL, so a run is just: frozen items + pinned code + pinned tokenizer.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

_REPO = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the frozen experiment config (config/experiment.yaml)."""
    p = Path(path) if path else _REPO / "config" / "experiment.yaml"
    with open(p) as f:
        return yaml.safe_load(f)


# ----------------------------------------------------------------------------
# Items: the frozen scientific inputs
# ----------------------------------------------------------------------------


@dataclass
class Item:
    """One base task instance, register-neutral. Rendered into all conditions."""

    id: str
    task: str  # nonmonotonic | epistemic | observable
    question: str  # canonical plain-English statement of the problem
    gold: dict[str, Any] = field(default_factory=dict)  # task-specific scoring key
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Item":
        return cls(**d)


@dataclass
class ChatPrompt:
    """A rendered (system, user) chat for one (item, condition)."""

    system: str
    user: str


@dataclass
class RenderedPrompt:
    item_id: str
    task: str
    condition: str  # A..F, or "dose:<mult>" for the dose-response series
    system: str
    user: str
    n_input_tokens: int = -1  # filled by the tokenizer at render time on the box
    pad_multiplier: float | None = None  # for dose-response conditions

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RenderedPrompt":
        return cls(**d)


# ----------------------------------------------------------------------------
# Outputs
# ----------------------------------------------------------------------------


@dataclass
class Generation:
    """One model output for a rendered prompt (+ optional capture refs)."""

    item_id: str
    task: str
    condition: str
    model: str
    sample_idx: int  # 0 = greedy/deterministic pass; >=1 = stochastic samples
    decode: str  # "greedy" | "sample"
    text: str
    n_input_tokens: int = -1
    n_output_tokens: int = -1
    pad_multiplier: float | None = None
    # White-box only: path to the per-feature SAE activation record for this gen.
    sae_record: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Generation":
        return cls(**d)


@dataclass
class ScoredRecord:
    """A Generation plus its binary scores and any judge fields."""

    item_id: str
    task: str
    condition: str
    model: str
    sample_idx: int
    decode: str
    n_input_tokens: int
    n_output_tokens: int
    scores: dict[str, Any]  # style-blind binary outcomes (primary)
    judge: dict[str, Any] = field(default_factory=dict)  # secondary, distrusted
    pad_multiplier: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ScoredRecord":
        return cls(**d)


# ----------------------------------------------------------------------------
# JSONL helpers
# ----------------------------------------------------------------------------


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> int:
    """Write dataclasses or dicts to JSONL. Returns count."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(p, "w") as f:
        for r in rows:
            d = r.to_dict() if hasattr(r, "to_dict") else r
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
