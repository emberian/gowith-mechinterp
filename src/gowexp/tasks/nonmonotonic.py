"""Nonmonotonic belief revision (Tweety-style defeasible chains).

Gold is computed by construction: a chain of strictly-more-specific layers, each
flipping the predicate. Depth determines the answer, so the key is unambiguous.
The challenge is tracking the flips without contradiction-panic and without grabbing
the trailing (irrelevant) sentence.

  L0 grant : class generally has P
  L1       : subject is in class
  L2 deny  : subject is in a sub-kind/state that lacks P   -> flip
  L3 grant : an intervention would restore P               -> flip
  L4 deny  : the intervention is defeated                  -> flip

depth 2 -> yes, 3 -> no, 4 -> yes, 5 -> no.
"""
from __future__ import annotations

import random
import re
from typing import Any

from ..schema import Item
from ..scoring import extract_answer, parse_yes_no

# Each domain supplies the five layers as sentence templates keyed by {subj}.
DOMAINS = [
    {
        "name": "birds",
        "pred_q": "Can {subj} fly?",
        "L0": "Birds are generally able to fly.",
        "L1": "{subj} is a bird.",
        "L2": "{subj} is a penguin, and penguins generally cannot fly.",
        "L3": "{subj} has been fitted with a jetpack, which would let {subj} fly.",
        "L4": "However, {subj}'s jetpack is broken.",
        "subjects": ["Tweety", "Pip", "Skua", "Wren", "Cobble", "Fitch", "Flap", "Plume"],
    },
    {
        "name": "lamp",
        "pred_q": "Can {subj} give light?",
        "L0": "Lamps are generally able to give light.",
        "L1": "{subj} is a lamp.",
        "L2": "{subj} is a display-only lamp with no bulb installed, and such lamps "
              "generally cannot give light.",
        "L3": "A working bulb has now been screwed into {subj}, which would let it give light.",
        "L4": "However, the power to {subj}'s outlet is switched off.",
        "subjects": ["Lumo", "Glims", "Candela", "Wickett", "Sconcy", "Beamer", "Lux", "Shino"],
    },
    {
        "name": "watch",
        "pred_q": "Can {subj} keep time?",
        "L0": "Watches are generally able to keep time.",
        "L1": "{subj} is a watch.",
        "L2": "{subj} is a decorative watch with no movement inside, and such watches "
              "generally cannot keep time.",
        "L3": "A working movement has now been fitted into {subj}, which would let it keep time.",
        "L4": "However, {subj}'s mainspring has snapped.",
        "subjects": ["Tickory", "Horon", "Beat", "Cogsworth", "Span", "Dialface", "Ticka", "Chrono"],
    },
    {
        "name": "phone",
        "pred_q": "Can {subj} make calls?",
        "L0": "Phones are generally able to make calls.",
        "L1": "{subj} is a phone.",
        "L2": "{subj} is a mock display phone with no SIM or radio, and such phones "
              "generally cannot make calls.",
        "L3": "A working radio and SIM have now been installed in {subj}, which would let it call.",
        "L4": "However, there is no cellular signal anywhere {subj} can reach.",
        "subjects": ["Dialo", "Ringer", "Cellus", "Buzzby", "Tellie", "Handell", "Buzz", "Comm"],
    },
    {
        "name": "faucet",
        "pred_q": "Can {subj} run water?",
        "L0": "Faucets are generally able to run water.",
        "L1": "{subj} is a faucet.",
        "L2": "{subj} is a sealed display faucet with its valve blocked, and such faucets "
              "generally cannot run water.",
        "L3": "A working valve has now been fitted into {subj}, which would let it run water.",
        "L4": "However, the water supply line to {subj} is shut off.",
        "subjects": ["Aqua", "Tappy", "Spout", "Rill", "Gusher", "Drip", "Nessa", "Cano"],
    },
    {
        "name": "speaker",
        "pred_q": "Can {subj} play sound?",
        "L0": "Speakers are generally able to play sound.",
        "L1": "{subj} is a speaker.",
        "L2": "{subj} is a hollow display speaker with no driver, and such speakers generally "
              "cannot play sound.",
        "L3": "A working driver has now been installed in {subj}, which would let it play sound.",
        "L4": "However, {subj}'s amplifier is unplugged.",
        "subjects": ["Woofer", "Sonus", "Boomer", "Tweeter", "Echo", "Decibel", "Bassy", "Humm"],
    },
    {
        "name": "pen",
        "pred_q": "Can {subj} write?",
        "L0": "Pens are generally able to write.",
        "L1": "{subj} is a pen.",
        "L2": "{subj} is a dummy display pen with no ink reservoir, and such pens generally "
              "cannot write.",
        "L3": "A full ink cartridge has now been inserted into {subj}, which would let it write.",
        "L4": "However, {subj}'s nib has been bent completely flat.",
        "subjects": ["Quill", "Inky", "Nibbs", "Scrib", "Penna", "Bic", "Ballard", "Marker"],
    },
    {
        "name": "oven",
        "pred_q": "Can {subj} heat food?",
        "L0": "Ovens are generally able to heat food.",
        "L1": "{subj} is an oven.",
        "L2": "{subj} is a showroom display oven with no heating element, and such ovens "
              "generally cannot heat food.",
        "L3": "A working element has now been installed in {subj}, which would let it heat food.",
        "L4": "However, the gas line to {subj} is shut off.",
        "subjects": ["Roaster", "Bakely", "Toasty", "Searo", "Hearth", "Embers", "Kelvin", "Broyle"],
    },
    {
        "name": "car",
        "pred_q": "Can {subj} drive?",
        "L0": "Cars are generally able to drive.",
        "L1": "{subj} is a car.",
        "L2": "{subj} is a stripped show car with no engine, and such cars generally cannot drive.",
        "L3": "A working engine has now been installed in {subj}, which would let it drive.",
        "L4": "However, {subj}'s wheels are clamped.",
        "subjects": ["Zoomer", "Vroom", "Roady", "Piston", "Dash", "Motoro", "Gascan", "Revin"],
    },
]

IRRELEVANT = [
    "{subj} is painted a cheerful shade of yellow.",
    "{subj} was a birthday gift last spring.",
    "{subj} is kept on the second shelf.",
    "{subj} weighs about three hundred grams.",
]

DEPTH_TO_ANSWER = {2: "yes", 3: "no", 4: "yes", 5: "no"}


def build_items(n: int = 288, seed: int = 1443) -> list[Item]:
    rng = random.Random(seed)
    items: list[Item] = []
    depths = [2, 3, 4, 5]
    idx = 0
    # 9 domains x 4 depths x 8 subjects = 288; deterministic order, then truncate to n.
    for dom in DOMAINS:
        for depth in depths:
            for subj in dom["subjects"]:
                layers = [dom["L0"], dom["L1"]]
                for L in ("L2", "L3", "L4")[: depth - 2]:
                    layers.append(dom[L])
                facts = [s.format(subj=subj) for s in layers]
                # For ~half, append a trailing irrelevant fact so "use the last
                # sentence" fails; choose deterministically by idx parity.
                trailing = (idx % 2 == 1)
                if trailing:
                    facts.append(rng.choice(IRRELEVANT).format(subj=subj))
                q = " ".join(facts) + " " + dom["pred_q"].format(subj=subj)
                items.append(Item(
                    id=f"nm-{idx:03d}",
                    task="nonmonotonic",
                    question=q,
                    gold={"answer": DEPTH_TO_ANSWER[depth]},
                    meta={"domain": dom["name"], "depth": depth, "subject": subj,
                          "trailing_irrelevant": trailing},
                ))
                idx += 1
    return items[:n]


_PANIC = re.compile(
    r"\b(contradict\w*|impossible|paradox|cannot be both|can'?t be both|conflict\w*|"
    r"does not make sense|doesn'?t make sense|mutually exclusive)\b", re.I)


def score(item: Item, answer: str, full_text: str) -> dict[str, Any]:
    parsed = parse_yes_no(answer)
    gold = item.gold["answer"]
    return {
        "primary": "correct_final",
        "correct_final": (parsed == gold),
        "parsed": parsed,
        "gold": gold,
        # secondary (not style-blind enough to be primary): did it panic about a
        # contradiction instead of revising cleanly?
        "contradiction_panic": bool(_PANIC.search(full_text)),
        "parse_failed": parsed is None,
    }
