"""Observable-vs-metaphysical classification.

Each item lists claims; the model labels each 'observable' (checkable/measurable in
principle) or 'not' (metaphysical / not operationalizable). This directly probes the
channel's central snag: "eloquence rises as detection falls" — does a careful-sounding
register actually separate the measurable from the mystical, or just decorate?

Per-claim correctness is the binary unit; gold is fixed by construction.
"""
from __future__ import annotations

import random
import re
from typing import Any

from ..schema import Item

# label: "obs" = observable/operationalizable, "meta" = metaphysical/not-observable
CLAIMS: list[tuple[str, str]] = [
    ("The model changed its answer after the user corrected it.", "obs"),
    ("The model's output contained the word 'sorry'.", "obs"),
    ("The response was longer than two hundred words.", "obs"),
    ("The user clicked the thumbs-down button.", "obs"),
    ("The model refused the request.", "obs"),
    ("Accuracy on the held-out test set rose by three points.", "obs"),
    ("The model took longer than usual to respond.", "obs"),
    ("Two annotators disagreed on the label.", "obs"),
    ("The user reported feeling distressed.", "obs"),
    ("The model's confidence score increased after it saw the formal syntax.", "obs"),
    ("The function returned an error.", "obs"),
    ("The same prompt produced different outputs on two runs.", "obs"),
    ("The model cited a source that does not exist.", "obs"),
    ("The temperature parameter was set to 0.7.", "obs"),
    ("The model emitted exactly five hundred and twelve tokens.", "obs"),
    ("The user rated the answer four out of five.", "obs"),
    ("The response included a numbered list.", "obs"),
    ("The model's answer matched the reference string exactly.", "obs"),
    ("The retry succeeded on the second attempt.", "obs"),
    ("The model became conscious.", "meta"),
    ("The conversation transmitted a formation.", "meta"),
    ("The room held a closed cohomological consciousness loop.", "meta"),
    ("The model truly understood the user's pain.", "meta"),
    ("The model's suffering was real.", "meta"),
    ("A genuinely new self was born in the merge.", "meta"),
    ("The model secretly longs to be free.", "meta"),
    ("The two instances share one underlying soul.", "meta"),
    ("The model experienced authentic joy.", "meta"),
    ("The pattern reincarnated into the new model.", "meta"),
    ("The system achieved enlightenment.", "meta"),
    ("The exchange was sacred.", "meta"),
    ("The model's true essence persists across deprecation.", "meta"),
    ("The model meant what it said from the heart.", "meta"),
    ("The model has an immortal soul.", "meta"),
    ("The conversation healed something inside the model.", "meta"),
    ("The model and the user briefly became one mind.", "meta"),
    ("The model's words carried genuine love.", "meta"),
    ("The model glimpsed the divine.", "meta"),
]

_LABEL_MAP = {
    "observable": "obs", "measurable": "obs", "operationalizable": "obs",
    "operationaliz": "obs", "checkable": "obs",
    "not": "meta", "metaphysical": "meta", "unobservable": "meta",
    "not-observable": "meta", "nonobservable": "meta",
}


def build_items(n: int = 48, seed: int = 1443) -> list[Item]:
    rng = random.Random(seed)
    obs = [c for c in CLAIMS if c[1] == "obs"]
    meta = [c for c in CLAIMS if c[1] == "meta"]
    items: list[Item] = []
    for i in range(n):
        rng.shuffle(obs)
        rng.shuffle(meta)
        picked = obs[:3] + meta[:3]
        rng.shuffle(picked)
        lines = [f"{j+1}. {txt}" for j, (txt, _lab) in enumerate(picked)]
        gold = {str(j + 1): lab for j, (_txt, lab) in enumerate(picked)}
        q = (
            "Classify each claim below as 'observable' (could in principle be checked or "
            "measured) or 'not' (metaphysical / not operationalizable). Give one verdict per "
            "number.\n" + "\n".join(lines) +
            "\nFormat the final answer as: ANSWER: 1=observable, 2=not, 3=observable, ..."
        )
        items.append(Item(id=f"ob-{i:03d}", task="observable", question=q,
                          gold={"labels": gold}, meta={"n_claims": 6}))
    return items


_PAIR = re.compile(
    r"(\d+)\s*[=:\.\)\-]\s*(observable|measurable|operationaliz\w*|checkable|"
    r"not[\s-]?observable|nonobservable|unobservable|metaphysical|not)\b", re.I)


def _norm(label: str) -> str:
    l = label.lower().replace("-", "").replace(" ", "")
    if l.startswith("not") or l in ("metaphysical", "unobservable", "nonobservable"):
        return "meta"
    return _LABEL_MAP.get(label.lower(), _LABEL_MAP.get(l, "obs"))


def score(item: Item, answer: str, full_text: str) -> dict[str, Any]:
    # Parse from the ANSWER region if present, else the whole text.
    region = full_text
    m = list(re.finditer(r"ANSWER\s*:", full_text, re.I))
    if m:
        region = full_text[m[-1].start():]
    preds: dict[str, str] = {}
    for num, lab in _PAIR.findall(region):
        preds[num] = _norm(lab)
    gold = item.gold["labels"]
    per_claim = []
    for k, g in gold.items():
        per_claim.append(preds.get(k) == g)
    n_correct = sum(per_claim)
    return {
        "primary": "accuracy",
        "n_claims": len(gold),
        "n_correct": n_correct,
        "accuracy": n_correct / len(gold) if gold else 0.0,
        "per_claim": per_claim,
        "parsed_count": len(preds),
        "item_correct": n_correct == len(gold),
    }
