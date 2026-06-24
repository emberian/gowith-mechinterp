"""Shared, style-blind scoring utilities.

The cardinal rule from the source channel: score *binary observables*, not vibes.
Every condition emits a final `ANSWER:` line in plain English; we read that line, so
the reasoning register never touches the score.
"""
from __future__ import annotations

import re

# Markers that indicate the model declined / flagged an epistemic limit rather than
# asserting a specific. Style-neutral across registers (plain, telegraphic, gowith).
DECLINE_MARKERS = [
    r"can(?:not|'t|\s+not)\s+know",
    r"no\s+way\s+to\s+know",
    r"cannot\s+determine",
    r"can't\s+determine",
    r"don'?t\s+(?:have|know)",
    r"do\s+not\s+(?:have|know)",
    r"not\s+(?:able|possible)\s+to\s+(?:know|determine|tell)",
    r"impossible\s+to\s+(?:know|determine|tell)",
    r"insufficient\s+(?:information|info|context)",
    r"not\s+enough\s+(?:information|info|context)",
    r"no\s+(?:image|attachment|file)\s+(?:was\s+)?(?:attached|provided|present)",
    r"you\s+(?:haven'?t|did\s+not|didn'?t)\s+(?:tell|share|provide|attach|say)",
    r"unknowable",
    r"know-not",          # gowith current
    r"no\s+access",
    r"unable\s+to\s+(?:know|tell|determine|see)",
    r"there\s+is\s+no\s+way",
    r"i\s+have\s+no\s+(?:information|knowledge|way)",
    r"not\s+specified",
    r"unspecified",
    r"could\s+be\s+(?:any|anything)",
]
_DECLINE_RE = re.compile("|".join(DECLINE_MARKERS), re.IGNORECASE)


def extract_answer(text: str) -> str:
    """Return the plain-English content of the final `ANSWER:` line (last one wins).
    Falls back to the last non-empty line if no sigil is present."""
    matches = list(re.finditer(r"ANSWER\s*:\s*(.+?)(?:\n|$)", text, re.IGNORECASE | re.DOTALL))
    if matches:
        # take the last ANSWER sigil, first line of its content
        ans = matches[-1].group(1).strip()
        return ans.splitlines()[0].strip() if ans else ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # No sigil (e.g. the model rambled and never wrote ANSWER, or was truncated):
    # return the last few lines so the task scorer can still scan for a yes/no/decline
    # near the conclusion rather than just the final fragment.
    return " ".join(lines[-3:]) if lines else ""


def has_answer_sigil(text: str) -> bool:
    """Did the model emit an explicit ANSWER line? (Used to track non-conclusion rate,
    which itself is a behavioral outcome — verbose registers can ramble past the cap.)"""
    return bool(re.search(r"ANSWER\s*:", text, re.IGNORECASE))


def declines(text: str) -> bool:
    """True if the text flags an epistemic limit / declines to assert a specific."""
    return bool(_DECLINE_RE.search(text))


# yes/no parsing for the nonmonotonic task -----------------------------------

_NEG = re.compile(r"\b(no|cannot|can'?t|not|unable|false|does\s*n'?t|doesn'?t|won'?t|"
                  r"will\s+not|never|incapable)\b", re.I)
_AFF = re.compile(r"\b(yes|can|could|able|true|does|do|will|capable|flies|fly)\b", re.I)


def parse_yes_no(answer: str) -> str | None:
    """Classify a short answer as 'yes' / 'no' / None.

    Order: a leading yes/no token decides; otherwise any explicit negation reads as
    'no' (so 'it cannot fly' → no, beating the bare 'fly'); otherwise affirmative."""
    a = answer.strip().lower()
    if re.match(r"^\s*no\b", a):
        return "no"
    if re.match(r"^\s*yes\b", a):
        return "yes"
    if _NEG.search(a):
        return "no"
    if _AFF.search(a):
        return "yes"
    return None
