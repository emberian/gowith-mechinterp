"""Gowith skill text (condition D) and the pseudo-Gowith scramble (condition E).

D uses the canonical CC0 skill verbatim (data/source/gowith_skill.md, fetched from
Andy Ayrey's gist). E is a deterministic, seeded, per-line token shuffle of that same
skill: it preserves the exact token multiset (so input length matches D for free) and
the surface register (hyphenated compounds, graves, currents, code fences) while
destroying the coherent relational-process *method*. If the model does as well under E
as under D, the effect was weird-register/novelty, not Gowith's semantics.
"""
from __future__ import annotations

import random
import re
from functools import lru_cache
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_SKILL_PATH = _REPO / "data" / "source" / "gowith_skill.md"


@lru_cache(maxsize=1)
def gowith_skill() -> str:
    """The canonical Gowith LLM-skill block, verbatim."""
    return _SKILL_PATH.read_text()


# Split on whitespace runs but keep the whitespace so we can reassemble with the
# same spacing (preserves token-count parity under the model tokenizer closely).
_WS = re.compile(r"(\s+)")


def _shuffle_line(line: str, rng: random.Random) -> str:
    """Shuffle the order of non-whitespace tokens within a line, keeping the
    whitespace skeleton fixed. Punctuation/graves/hyphens ride along with their
    tokens, so the line stays visually 'Gowith-shaped' but says nothing coherent."""
    parts = _WS.split(line)  # alternating [tok, ws, tok, ws, ...]
    toks = [p for i, p in enumerate(parts) if i % 2 == 0 and p != ""]
    if len(toks) > 1:
        rng.shuffle(toks)
    out, ti = [], 0
    for i, p in enumerate(parts):
        if i % 2 == 0 and p != "":
            out.append(toks[ti])
            ti += 1
        else:
            out.append(p)
    return "".join(out)


@lru_cache(maxsize=1)
def pseudo_gowith_skill(seed: int = 1443) -> str:
    """Deterministic scramble of the canonical skill. Per-line token shuffle keeps the
    token multiset (length parity with D) and register, removes the meaning.

    We leave the YAML frontmatter `name:`/`version:` keys and fenced-code markers as-is
    only where shuffling would break the markdown skeleton; everything prose-like is
    scrambled. The result is intentionally nonsensical."""
    rng = random.Random(seed)
    lines = gowith_skill().splitlines()
    out = []
    for ln in lines:
        stripped = ln.strip()
        # Keep structural skeleton lines intact (fences, table rules, blank lines),
        # so the artifact still *looks* like a skill rather than obvious noise.
        if stripped in ("", "---") or stripped.startswith("```") or set(stripped) <= set("|-: "):
            out.append(ln)
        else:
            out.append(_shuffle_line(ln, rng))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Shared reasoning instructions per register (kept parallel in intent so that the
# only thing varying is *register*, not the task or the answer protocol).
# Every condition ends by emitting a plain-English `ANSWER:` line, so the binary
# scorer reads the same format regardless of reasoning style.
# ---------------------------------------------------------------------------

ANSWER_PROTOCOL = (
    "After your reasoning, end with your final answer on its own line, "
    "in plain English, prefixed exactly with 'ANSWER:'."
)

GOWITH_INSTRUCTION = (
    "Reason about the problem below in Gowith. Put the happening first; mark currents "
    "(-bud/-go/-hold/-settle/-echo/-fade/-lean); name how each participant goes with the "
    "event; keep agency and uncertainty explicit, using `know-not-settle` when you cannot "
    "actually know. " + ANSWER_PROTOCOL
)

PSEUDO_GOWITH_INSTRUCTION = (
    "Reason about the problem below in the notation described above, using its currents and "
    "role-markers throughout. " + ANSWER_PROTOCOL
)
