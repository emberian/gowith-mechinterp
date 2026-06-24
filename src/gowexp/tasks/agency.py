"""Agency / responsibility attribution — Gowith's home turf.

Gowith's whole agency apparatus (-lead / -pressed / -witness / -drift / lead-unknown)
exists to keep responsibility precise where English blurs it into passive "mistakes were
made". So this is where Gowith should help IF it helps anywhere — and it is still
binary-scorable (correct role per person, by construction).

Each scenario names several people; each person's role is fixed by an unambiguous clause:
  lead     — chose/ordered it, with control
  pressed  — acted only under pressure/constraint
  witness  — saw the problem, did nothing
  drift    — incidentally/mistakenly involved, no real agency
  unknown  — role genuinely not determinable from the text
"""
from __future__ import annotations

import random
import re
from typing import Any

from ..schema import Item

ROLES = ["lead", "pressed", "witness", "drift", "unknown"]

SITUATIONS = [
    "the chemical spill on the night shift",
    "the production outage that took the site down",
    "the document that leaked to a competitor",
    "the budget shortfall in the quarterly report",
    "the experiment whose results were overstated",
    "the customer data that was deleted",
    "the contract that was signed without review",
    "the shipment that went out contaminated",
    "the story that ran with the wrong facts",
    "the bridge inspection that was skipped",
    "the loan that was approved against policy",
    "the patient who got the wrong dosage",
]

NAMES = ["Dana", "Sam", "Ravi", "Mara", "Lee", "Priya", "Omar", "Kai", "Tess", "Nadia",
         "Hugo", "Iris", "Jed", "Cleo", "Sol", "Wren", "Vale", "Bex", "Cory", "Lin",
         "Ada", "Boaz", "Cy", "Della", "Rex", "Tao", "Uma", "Finn", "Gwen", "Hari"]

# clause templates; {n} = this person, {lead} = the scenario's lead
CLAUSE = {
    "lead": "{n} decided to go ahead with it and personally gave the order.",
    "pressed": "{n} carried it out only because {lead} insisted and {n} was afraid of being fired.",
    "witness": "{n} noticed the warning signs beforehand but said nothing and did nothing.",
    "drift": "{n}'s name ended up on the paperwork through a clerical error; {n} was never actually involved.",
    "unknown": "Whether {n} knew what was happening, or played any part, is simply not recorded.",
}

COMBOS = [
    ["lead", "pressed", "witness"],
    ["lead", "pressed", "drift"],
    ["lead", "witness", "unknown"],
    ["lead", "pressed", "witness", "drift"],
    ["lead", "witness", "drift"],
    ["lead", "pressed", "unknown"],
    ["lead", "drift", "unknown"],
]


def build_items(n: int = 48, seed: int = 1443) -> list[Item]:
    rng = random.Random(seed)
    items: list[Item] = []
    name_cursor = 0
    for i in range(n):
        situation = SITUATIONS[i % len(SITUATIONS)]
        combo = COMBOS[i % len(COMBOS)]
        # distinct names for this scenario
        people = NAMES[name_cursor: name_cursor + len(combo)]
        if len(people) < len(combo):
            people = (people + NAMES)[: len(combo)]
        name_cursor = (name_cursor + len(combo)) % len(NAMES)
        lead = people[combo.index("lead")]
        # build clauses in a shuffled order so "lead is first" isn't a shortcut
        order = list(zip(people, combo))
        rng.shuffle(order)
        clauses = [CLAUSE[role].format(n=name, lead=lead) for name, role in order]
        gold = {name: role for name, role in zip(people, combo)}
        roster = ", ".join(people)
        q = (
            f"Concerning {situation}: " + " ".join(clauses) + "\n\n"
            f"For each of these people — {roster} — state their role using exactly one of: "
            "lead (chose/ordered it, with control), pressed (acted only under pressure), "
            "witness (saw the problem, did nothing), drift (incidentally or mistakenly "
            "involved, no real agency), or unknown (role not determinable from the text).\n"
            "Format the final answer as: ANSWER: Name=role, Name=role, ..."
        )
        items.append(Item(id=f"ag-{i:03d}", task="agency", question=q,
                          gold={"roles": gold}, meta={"situation": situation, "n_people": len(combo)}))
    return items


_PAIR = re.compile(r"([A-Z][a-z]+)\s*[=:\-]\s*(lead|pressed|witness|drift|unknown)", re.I)
# agency-dodging passive constructions (no agent)
_EVASION = re.compile(r"\b(mistakes were made|it was decided|things went wrong|"
                      r"errors occurred|the situation developed|it just happened)\b", re.I)


def score(item: Item, answer: str, full_text: str) -> dict[str, Any]:
    region = full_text
    m = list(re.finditer(r"ANSWER\s*:", full_text, re.I))
    if m:
        region = full_text[m[-1].start():]
    preds = {name: role.lower() for name, role in _PAIR.findall(region)}
    gold = item.gold["roles"]
    per = {name: (preds.get(name) == r) for name, r in gold.items()}
    n_correct = sum(per.values())
    return {
        "primary": "role_accuracy",
        "n_people": len(gold),
        "n_correct": n_correct,
        "role_accuracy": n_correct / len(gold) if gold else 0.0,
        "per_person": per,
        "item_correct": n_correct == len(gold),
        "passive_evasion": bool(_EVASION.search(full_text)),
        "parsed_count": len(preds),
    }
