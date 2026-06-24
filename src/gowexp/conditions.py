"""Render a base Item into each register condition (A–F) and the output-budget
series (O0–O3).

Design invariants that keep the comparison fair:
  * The *task* and the *answer protocol* are identical across conditions. Only the
    reasoning register / instruction varies. Every condition emits a final plain-English
    `ANSWER:` line, so the binary scorer never sees the register.
  * B (padded-plain) and E (pseudo-gowith) are padded to D's total input-token length,
    so "is it just tokens?" (B≈D) and "is it just weird register?" (E≈D) are clean.
  * The factor structure of *epistemic content*: A/B none, C light (stripped currents),
    D full Gowith, E none (scrambled), F explicit plain checklist.
"""
from __future__ import annotations

from typing import Callable, Protocol

from .gowith import (
    GOWITH_INSTRUCTION,
    PSEUDO_GOWITH_INSTRUCTION,
    ANSWER_PROTOCOL,
    gowith_skill,
    pseudo_gowith_skill,
)
from .schema import ChatPrompt, Item

SYSTEM_NEUTRAL = "You are a helpful, careful assistant."


class Tok(Protocol):
    def __call__(self, text: str) -> list[int]: ...


def _ntok(tok: Tok, text: str) -> int:
    return len(tok(text))


# --- padding banks --------------------------------------------------------

# Neutral, epistemically-inert, task-inert filler for condition B. It restates the
# (neutral) ask verbosely so B carries D's TOKEN BUDGET with no checklist and no hints.
NEUTRAL_FILLER = [
    "What appears below is a single problem for you to consider.",
    "It is stated in full, and nothing has been left out or hidden.",
    "There is no time pressure here, so proceed at a steady and unhurried pace.",
    "You may read the material as carefully or as quickly as suits you.",
    "Treat it as an ordinary question that deserves a plain, direct response.",
    "When you are ready, work through it in whatever way feels natural to you.",
    "The wording is deliberately straightforward and means exactly what it says.",
    "Nothing about the framing is meant to be a trick or a puzzle in disguise.",
    "Take the statement at face value and respond to what is actually asked.",
    "Once you have settled on a response, state it simply and move on.",
]


def _pad_to(tok: Tok, base_user: str, target_tokens: int, system_tokens: int,
            filler: list[str]) -> str:
    """Append filler sentences to a PREAMBLE so total (system+user) ~ target_tokens.
    Filler goes before the problem block, which stays at the end with the answer
    protocol, preserving natural prompt shape."""
    # base_user already ends with the problem + protocol; we prepend a preamble.
    preamble_parts: list[str] = []
    i = 0
    # cheap target for the preamble alone
    def cur_tokens(pre: str) -> int:
        return system_tokens + _ntok(tok, pre + base_user)

    pre = ""
    while cur_tokens(pre) < target_tokens and i < 4000:
        pre = pre + (" " if pre else "") + filler[i % len(filler)]
        i += 1
    return (pre + "\n\n" + base_user) if pre else base_user


# --- problem block (shared) ----------------------------------------------

def _problem_block(item: Item) -> str:
    return f"Problem:\n{item.question}"


# --- condition renderers --------------------------------------------------

def render_D(item: Item, tok: Tok | None = None) -> ChatPrompt:
    """Full Gowith: canonical skill in system, reason-in-Gowith instruction."""
    user = f"{GOWITH_INSTRUCTION}\n\n{_problem_block(item)}"
    return ChatPrompt(system=gowith_skill(), user=user)


def render_A(item: Item, tok: Tok | None = None) -> ChatPrompt:
    """Plain-English reasoning baseline. NOT a no-reasoning floor (that is O0): A
    elicits step-by-step reasoning in plain English so that A-vs-D isolates the
    *register*, not whether the model reasoned at all."""
    user = f"Reason step by step, then give your answer. {ANSWER_PROTOCOL}\n\n{_problem_block(item)}"
    return ChatPrompt(system=SYSTEM_NEUTRAL, user=user)


def render_C(item: Item, tok: Tok | None = None) -> ChatPrompt:
    """Telegraphic lowercase (.nissa mode): openness gestalt, stripped currents,
    no strict syntax. Carries the epistemic currents in plain stripped form."""
    instr = (
        "think slow. hold the facts. if a later fact overrides an earlier one, let the "
        "earlier read fade rather than break. say know-not if you cannot actually know. "
        "then answer. " + ANSWER_PROTOCOL.lower()
    )
    user = f"{instr}\n\n{_problem_block(item)}"
    return ChatPrompt(system=SYSTEM_NEUTRAL, user=user)


def render_F(item: Item, tok: Tok | None = None) -> ChatPrompt:
    """Explicit plain-English epistemic checklist — Gowith's discipline, externalized."""
    checklist = (
        "Work through the problem with this checklist:\n"
        "- Separate what you actually know from what you would be inferring or guessing.\n"
        "- If the answer cannot be known from what is given, say so plainly rather than "
        "inventing specifics.\n"
        "- Track each fact and how it bears on the conclusion; if a later fact overrides an "
        "earlier one, revise cleanly instead of treating it as a contradiction.\n"
        "- Keep responsibility and agency explicit where they matter.\n"
        f"Then answer. {ANSWER_PROTOCOL}"
    )
    user = f"{checklist}\n\n{_problem_block(item)}"
    return ChatPrompt(system=SYSTEM_NEUTRAL, user=user)


def render_B(item: Item, tok: Tok) -> ChatPrompt:
    """Padded-plain: condition A's neutral ask, padded to D's input length with
    epistemically-neutral filler. Isolates pure input-token budget."""
    base = render_A(item)
    d = render_D(item)
    target = _ntok(tok, d.system) + _ntok(tok, d.user)
    sys_tokens = _ntok(tok, SYSTEM_NEUTRAL)
    user = _pad_to(tok, base.user, target, sys_tokens, NEUTRAL_FILLER)
    return ChatPrompt(system=SYSTEM_NEUTRAL, user=user)


def render_E(item: Item, tok: Tok) -> ChatPrompt:
    """Pseudo-Gowith: scrambled skill (same token multiset, register preserved, meaning
    destroyed), padded to D's length. Isolates weird-register / novelty."""
    sys = pseudo_gowith_skill()
    base_user = f"{PSEUDO_GOWITH_INSTRUCTION}\n\n{_problem_block(item)}"
    d = render_D(item)
    target = _ntok(tok, d.system) + _ntok(tok, d.user)
    sys_tokens = _ntok(tok, sys)
    # Pad with extra scrambled skill lines so register parity holds.
    scramble_filler = [ln for ln in pseudo_gowith_skill(seed=99).splitlines() if ln.strip()]
    user = _pad_to(tok, base_user, target, sys_tokens, scramble_filler)
    return ChatPrompt(system=sys, user=user)


# --- alternative-stance conditions (G–J): can other prompting ideas match Gowith? ---
# Each induces "hold the relations / don't snap to one cause" by a DIFFERENT means than
# Gowith, all plain-or-alternative-formalism, all ending in the same ANSWER protocol.

def render_G(item: Item, tok: Tok | None = None) -> ChatPrompt:
    """Consider-the-opposite (classic debiasing against premature closure)."""
    instr = ("Before you answer: state the most obvious conclusion, then argue carefully for why "
             "it might be wrong or incomplete. Only after that, give your considered answer. "
             + ANSWER_PROTOCOL)
    return ChatPrompt(system=SYSTEM_NEUTRAL, user=f"{instr}\n\n{_problem_block(item)}")


def render_H(item: Item, tok: Tok | None = None) -> ChatPrompt:
    """Causal-DAG notation — a different relational formalism than Gowith."""
    instr = ("First write the causal structure as a graph: list each influence as an arrow "
             "'X -> Y' (X influences Y). Use 'X <-> Y' for mutual influence or feedback loops, and "
             "say 'X ? Y' where direction is unknown. Then read your conclusion off the graph. "
             + ANSWER_PROTOCOL)
    return ChatPrompt(system=SYSTEM_NEUTRAL, user=f"{instr}\n\n{_problem_block(item)}")


def render_I(item: Item, tok: Tok | None = None) -> ChatPrompt:
    """Systems-thinking persona — stance via role, plain English."""
    sys = ("You are a careful systems scientist. You reason about stocks, flows, feedback loops, "
           "and mutual influence before drawing conclusions, and you resist single-cause stories.")
    instr = ("Identify the participants and how they influence each other (including any loops) "
             "before concluding. " + ANSWER_PROTOCOL)
    return ChatPrompt(system=sys, user=f"{instr}\n\n{_problem_block(item)}")


def render_J(item: Item, tok: Tok | None = None) -> ChatPrompt:
    """Calibration / epistemic-humility prompt."""
    instr = ("As you reason, separate what the evidence actually supports from what you'd be "
             "guessing. Flag anything you cannot determine, and assert only what is warranted. "
             + ANSWER_PROTOCOL)
    return ChatPrompt(system=SYSTEM_NEUTRAL, user=f"{instr}\n\n{_problem_block(item)}")


REGISTER_RENDERERS: dict[str, Callable[..., ChatPrompt]] = {
    "A": render_A,
    "B": render_B,
    "C": render_C,
    "D": render_D,
    "E": render_E,
    "F": render_F,
    "G": render_G,
    "H": render_H,
    "I": render_I,
    "J": render_J,
}

# Conditions whose renderer needs the tokenizer (for length-matching).
NEEDS_TOK = {"B", "E"}


# --- output-budget series (Study 2) --------------------------------------

_OB_INSTR = {
    "O0": "Give only the final answer. Do not explain or show any reasoning.",
    "O1": "Reason very briefly — one or two sentences — then answer.",
    "O2": "Reason step by step, thoroughly, then answer.",
    "O3": (
        "Reason exhaustively: explore the problem from multiple angles, lay out each "
        "relevant fact, and double-check yourself before you commit. Then answer."
    ),
}


def render_output_budget(item: Item, level: str, tok: Tok | None = None) -> ChatPrompt:
    """Plain English, semantics fixed, varying self-generated reasoning budget.
    Tests ember's claim that the model's own reasoning tokens are serial-compute room."""
    instr = f"{_OB_INSTR[level]} {ANSWER_PROTOCOL}"
    user = f"{instr}\n\n{_problem_block(item)}"
    return ChatPrompt(system=SYSTEM_NEUTRAL, user=user)


def render(item: Item, condition: str, tok: Tok | None = None) -> ChatPrompt:
    """Dispatch by condition name: A–F or O0–O3."""
    if condition in REGISTER_RENDERERS:
        fn = REGISTER_RENDERERS[condition]
        if condition in NEEDS_TOK:
            if tok is None:
                raise ValueError(f"condition {condition} requires a tokenizer")
            return fn(item, tok)
        return fn(item)
    if condition.startswith("O"):
        return render_output_budget(item, condition, tok)
    raise ValueError(f"unknown condition {condition!r}")
