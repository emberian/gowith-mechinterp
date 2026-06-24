"""Task families. Each module exposes:
    build_items(n: int, seed: int) -> list[Item]
    score(item: Item, answer: str, full_text: str) -> dict[str, Any]
"""
from __future__ import annotations

from . import agency, correlative, epistemic, nonmonotonic, observable

REGISTRY = {
    "nonmonotonic": nonmonotonic,
    "epistemic": epistemic,
    "observable": observable,
    "agency": agency,
    "correlative": correlative,
}

__all__ = ["REGISTRY", "nonmonotonic", "epistemic", "observable", "agency", "correlative"]
