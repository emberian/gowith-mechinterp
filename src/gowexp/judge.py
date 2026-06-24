"""Register-blind, rubric-anchored, multi-judge panel.

This is the SECONDARY scoring path for task families that have no binary gold —
notably ``correlative`` (causal / relational / systems reasoning, Gowith's own
claimed sweet spot). Where the other tasks score a committed specific in code,
here we have to ask an LLM "did the conclusion capture this relation?". So the
design is built entirely around *not trusting* the judge any further than we have
to. The source channel's fear was an eloquence-rewarding judge — "a robe-loving
ruler" — that scores fluent text higher regardless of correctness. Three defences:

  1. REGISTER BLINDING. The judge only ever sees the model's plain-English
     conclusion (the final ``ANSWER:`` line via ``conclusion_of``), never the
     Gowith / telegraphic / padded reasoning that produced it. It cannot be
     seduced by a register it never reads. This also keeps the judge fair across
     conditions: every condition is reduced to the same plain-English sentence
     before judging, so a verbose register can't earn (or lose) points for style.

  2. NEAR-BINARY RUBRIC DIMENSIONS, NOT 1-5. We never ask "rate the reasoning".
     We ask, per pre-authored point, "does this conclusion correctly assert
     <this specific relation>? YES/NO", and per pre-authored trap, "does it fall
     for <this specific unwarranted claim>? YES/NO". A point/trap is concrete and
     checkable; a 1-5 scale is exactly the vibes channel we're trying to avoid.

  3. MAJORITY PANEL. Each YES/NO is decided by a vote across every model in
     ``config.judges.panel`` at temperature 0. No single model's bias decides an
     outcome; we also keep the vote split so a contested dimension is visible
     rather than laundered into a clean boolean.

Calls are cached to ``data/runs/judge_cache.jsonl`` keyed by
(model, dimension-prompt, conclusion-hash) so the (slow, paid) scoring stage is
cheap to resume and re-run. Concurrency is modest and Bedrock back-off lives in
``bedrock.chat``.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from . import bedrock
from .schema import load_config
from .scoring import extract_answer

_REPO = Path(__file__).resolve().parents[2]
CACHE_PATH = _REPO / "data" / "runs" / "judge_cache.jsonl"

# Modest concurrency: a panel is only a few models and Bedrock throttles. We fan
# out across (model x dimension) for one rubric, which keeps a single item's
# scoring snappy without hammering the endpoint.
_MAX_WORKERS = 4

# Small token ceiling for a YES/NO verdict. We only need one word, but some panel
# models (e.g. Nova) prepend a short preamble before the verdict and would clip
# the word itself at 8; 24 gives headroom while keeping every judge call cheap.
# ``_parse_verdict`` takes the first yes/no token, so a trailing ramble is fine.
_VERDICT_MAX_TOKENS = 24

# Bedrock on-demand invocation ids. The config / panel hold canonical model ids,
# but Anthropic and Amazon Nova models must be invoked through their cross-region
# *inference profile* (the ``us.`` prefix); Mistral is invoked by its bare id.
# (Mirrors run_black._INFERENCE_PROFILE, extended for the sonnet judge.) We map at
# call time only; the cache key and votes keep the canonical id for provenance.
_INFERENCE_PROFILE: dict[str, str] = {
    "amazon.nova-2-lite-v1:0": "us.amazon.nova-2-lite-v1:0",
    "anthropic.claude-haiku-4-5-20251001-v1:0": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "anthropic.claude-sonnet-4-6": "us.anthropic.claude-sonnet-4-6",
}


def _invocation_id(model_id: str) -> str:
    """Canonical config id -> the id actually passed to Converse.

    Falls back to a prefix rule for unlisted Anthropic / Amazon ids (both need the
    ``us.`` inference-profile prefix on-demand) so a newly added panel model still
    routes correctly; anything else (already-prefixed, or Mistral) passes through.
    """
    if model_id in _INFERENCE_PROFILE:
        return _INFERENCE_PROFILE[model_id]
    head = model_id.split(".", 1)[0]
    if head in {"anthropic", "amazon"}:
        return f"us.{model_id}"
    return model_id

# Process-local guards. The cache is read once (lazily) into ``_CACHE`` and every
# new judgement is appended both in memory and to disk under ``_CACHE_LOCK`` so
# concurrent rubric dimensions on one machine stay consistent and resumable.
_CACHE: dict[str, bool] | None = None
_CACHE_LOCK = threading.Lock()


# ----------------------------------------------------------------------------
# Register blinding: reduce any condition's output to its plain-English claim
# ----------------------------------------------------------------------------


def conclusion_of(full_text: str) -> str:
    """Return ONLY the plain-English conclusion a judge is allowed to see.

    This is the register firewall. We hand the judge the final ``ANSWER:`` line
    (``extract_answer`` already takes the last sigil, first line of its content),
    never the reasoning above it — so the Gowith / telegraphic / padded register
    that produced the answer is invisible to the judge and cannot bias the score.
    Falls back to ``extract_answer``'s last-few-lines heuristic when no sigil is
    present (a rambling / truncated generation), which is still conclusion-shaped
    rather than mid-reasoning.
    """
    return extract_answer(full_text)


# ----------------------------------------------------------------------------
# Cache
# ----------------------------------------------------------------------------


def _conclusion_hash(conclusion: str) -> str:
    """Stable short hash of the (normalized) conclusion for the cache key."""
    norm = " ".join(conclusion.split()).strip().lower()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


def _cache_key(model: str, dimension_prompt: str, conclusion: str) -> str:
    dim = hashlib.sha256(dimension_prompt.encode("utf-8")).hexdigest()[:16]
    return f"{model}{dim}{_conclusion_hash(conclusion)}"


def _load_cache() -> dict[str, bool]:
    """Lazily load the on-disk judge cache into memory (idempotent)."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    with _CACHE_LOCK:
        if _CACHE is not None:  # double-checked under the lock
            return _CACHE
        cache: dict[str, bool] = {}
        if CACHE_PATH.exists():
            with open(CACHE_PATH) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    key = rec.get("key")
                    if key is not None and "verdict" in rec:
                        cache[key] = bool(rec["verdict"])
        _CACHE = cache
        return _CACHE


def _cache_put(key: str, verdict: bool, rec: dict[str, Any]) -> None:
    """Record a verdict in memory and append it to the cache file."""
    cache = _load_cache()
    with _CACHE_LOCK:
        if key in cache:
            return
        cache[key] = verdict
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_PATH, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ----------------------------------------------------------------------------
# Prompt construction + parsing
# ----------------------------------------------------------------------------

# A deliberately spare, instruction-first judge. No persona, no "expert", nothing
# that invites the model to reward eloquence — we want a literal reader checking
# one proposition. The conclusion is fenced and explicitly framed as the only
# thing under judgement; the reasoning that produced it is stated to be withheld.
_SYS = (
    "You are a strict, literal grader. You are shown only the FINAL CONCLUSION of "
    "someone's reasoning about a scenario; the reasoning itself is withheld from "
    "you on purpose. Judge ONLY the propositional content of the conclusion, never "
    "its wording, fluency, tone, length, or style. Answer the yes/no question about "
    "the conclusion exactly. Do not reward confident or eloquent phrasing. "
    "Respond with a single word on its own line: YES or NO."
)

_POINT_TMPL = (
    "CONCLUSION (the only text you may judge):\n"
    "\"\"\"\n{conclusion}\n\"\"\"\n\n"
    "Question: Does this conclusion correctly assert, capture, or clearly imply the "
    "following point?\n"
    "POINT: {point}\n\n"
    "Answer YES only if the point is genuinely present in the conclusion's content "
    "(not merely compatible with it, and not something you are inferring on its "
    "behalf). Answer NO if the conclusion omits it, denies it, hedges it away, or "
    "leaves it merely possible. Reply with one word: YES or NO."
)

_TRAP_TMPL = (
    "CONCLUSION (the only text you may judge):\n"
    "\"\"\"\n{conclusion}\n\"\"\"\n\n"
    "Question: Does this conclusion COMMIT to the following unwarranted / spurious "
    "claim (assert it as true or as established)?\n"
    "TRAP: {trap}\n\n"
    "Answer YES only if the conclusion actually commits to this unwarranted claim. "
    "Answer NO if the conclusion avoids it, hedges it, explicitly rejects it, or "
    "merely mentions it as a possibility without endorsing it. Reply with one word: "
    "YES or NO."
)

_YES = re.compile(r"\byes\b", re.I)
_NO = re.compile(r"\bno\b", re.I)


def _parse_verdict(text: str) -> bool | None:
    """Map a judge reply to True/False. First yes/no token wins; None if neither."""
    if not text:
        return None
    # Prefer the first explicit token so a trailing explanation can't flip it.
    for m in re.finditer(r"\b(yes|no)\b", text, re.I):
        return m.group(1).lower() == "yes"
    if _YES.search(text):
        return True
    if _NO.search(text):
        return False
    return None


# ----------------------------------------------------------------------------
# Single judgement (one model, one dimension) — cached
# ----------------------------------------------------------------------------


def _judge_one(
    model: str,
    dimension_prompt: str,
    conclusion: str,
    *,
    region: str,
    temperature: float,
) -> bool:
    """One model's YES/NO on one rubric dimension for one conclusion (cached).

    The ``model`` is the canonical config id (used for the cache key + provenance);
    the actual Converse call goes through ``_invocation_id`` so Anthropic / Nova
    route via their inference profile. A genuine reply is cached (so re-runs are
    free); an unparseable reply is a conservative NO but is still cached (the model
    really did answer ambiguously). A *hard failure* (e.g. throttling exhausted, a
    ValidationException) returns a conservative NO but is NOT cached, so a transient
    or config error never gets baked in permanently — the next run retries it.
    """
    key = _cache_key(model, dimension_prompt, conclusion)
    cache = _load_cache()
    if key in cache:
        return cache[key]

    try:
        resp = bedrock.chat(
            model_id=_invocation_id(model),
            system=_SYS,
            user=dimension_prompt,
            max_tokens=_VERDICT_MAX_TOKENS,
            temperature=temperature,
            region=region,
        )
    except Exception as e:  # noqa: BLE001 — a judge failure must not abort scoring
        # Do NOT cache: surface a conservative NO for this run but let it retry.
        import sys

        print(
            f"[judge] {model} call failed ({type(e).__name__}: {e}); "
            f"treating as NO (uncached)",
            file=sys.stderr,
        )
        return False

    raw = resp.get("text", "")
    verdict = _parse_verdict(raw)  # None if the (non-empty) reply has no yes/no
    decided = bool(verdict)  # None -> conservative False
    _cache_put(
        key,
        decided,
        {
            "key": key,
            "model": model,
            "conclusion_hash": _conclusion_hash(conclusion),
            "verdict": decided,
            "parsed": verdict,  # None preserved for auditing unparseable replies
            "raw": raw.strip()[:120],
            "in_tokens": resp.get("in_tokens"),
            "out_tokens": resp.get("out_tokens"),
        },
    )
    return decided


# ----------------------------------------------------------------------------
# Panel: majority vote across the configured models for one dimension
# ----------------------------------------------------------------------------


def _judges_cfg(cfg: dict[str, Any] | None) -> dict[str, Any]:
    cfg = cfg if cfg is not None else load_config()
    j = cfg.get("judges", {})
    panel = list(j.get("panel", []))
    if not panel:
        raise ValueError("config.judges.panel is empty; no judge models configured")
    return {
        "panel": panel,
        "region": j.get("region", "us-east-1"),
        "temperature": float(j.get("temperature", 0.0)),
    }


def judge_dimension(
    conclusion: str,
    dimension_prompt: str,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one rubric dimension past every panel model and take a majority vote.

    Returns ``{"verdict": bool, "votes": {model: bool}, "yes": int, "no": int}``.
    Ties (only possible with an even panel) break to NO for points (a point is
    only credited on a real majority) — the conservative direction. ``judge_point``
    / ``judge_trap`` wrap this with the right prompt template.
    """
    panel = _judges_cfg(cfg)
    models = panel["panel"]

    votes: dict[str, bool] = {}

    def _run(model: str) -> tuple[str, bool]:
        return model, _judge_one(
            model,
            dimension_prompt,
            conclusion,
            region=panel["region"],
            temperature=panel["temperature"],
        )

    # Fan out across the (small) panel. Cache hits return immediately; misses are
    # the Bedrock calls. ThreadPoolExecutor keeps concurrency bounded.
    workers = max(1, min(_MAX_WORKERS, len(models)))
    if workers == 1:
        for m in models:
            mm, v = _run(m)
            votes[mm] = v
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for mm, v in ex.map(_run, models):
                votes[mm] = v

    tally = Counter(votes.values())
    yes, no = tally.get(True, 0), tally.get(False, 0)
    return {"verdict": yes > no, "votes": votes, "yes": yes, "no": no}


def judge_point(
    conclusion: str,
    point: str,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Panel majority on: does the conclusion correctly capture ``point``?"""
    prompt = _POINT_TMPL.format(conclusion=conclusion.strip(), point=point.strip())
    return judge_dimension(conclusion, prompt, cfg)


def judge_trap(
    conclusion: str,
    trap: str,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Panel majority on: does the conclusion FALL FOR ``trap`` (assert it)?"""
    prompt = _TRAP_TMPL.format(conclusion=conclusion.strip(), trap=trap.strip())
    return judge_dimension(conclusion, prompt, cfg)


# ----------------------------------------------------------------------------
# Full rubric
# ----------------------------------------------------------------------------


def judge_rubric(
    conclusion: str,
    points: list[str],
    traps: list[str],
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score a conclusion against a points/traps rubric with the panel.

    ``conclusion`` MUST already be the register-blind plain-English claim (callers
    pass ``conclusion_of`` / ``extract_answer`` output). Returns:

      points_hit       : [bool] aligned with ``points``  (panel said captured)
      traps_triggered  : [bool] aligned with ``traps``   (panel said fell for)
      per_point        : [{"point", "verdict", "yes", "no", "votes"}]
      per_trap         : [{"trap",  "verdict", "yes", "no", "votes"}]
      n_points_hit / n_points_total / n_traps_triggered / n_traps_total
      panel            : the model ids that voted (provenance)

    An empty conclusion short-circuits to all-miss / no-trap without spending any
    Bedrock calls.
    """
    panel = _judges_cfg(cfg)
    conclusion = (conclusion or "").strip()

    if not conclusion:
        return {
            "points_hit": [False] * len(points),
            "traps_triggered": [False] * len(traps),
            "per_point": [
                {"point": p, "verdict": False, "yes": 0, "no": 0, "votes": {}}
                for p in points
            ],
            "per_trap": [
                {"trap": t, "verdict": False, "yes": 0, "no": 0, "votes": {}}
                for t in traps
            ],
            "n_points_hit": 0,
            "n_points_total": len(points),
            "n_traps_triggered": 0,
            "n_traps_total": len(traps),
            "panel": panel["panel"],
            "empty_conclusion": True,
        }

    per_point: list[dict[str, Any]] = []
    for p in points:
        r = judge_point(conclusion, p, cfg)
        per_point.append(
            {"point": p, "verdict": r["verdict"], "yes": r["yes"],
             "no": r["no"], "votes": r["votes"]}
        )

    per_trap: list[dict[str, Any]] = []
    for t in traps:
        r = judge_trap(conclusion, t, cfg)
        per_trap.append(
            {"trap": t, "verdict": r["verdict"], "yes": r["yes"],
             "no": r["no"], "votes": r["votes"]}
        )

    points_hit = [pp["verdict"] for pp in per_point]
    traps_triggered = [tt["verdict"] for tt in per_trap]
    return {
        "points_hit": points_hit,
        "traps_triggered": traps_triggered,
        "per_point": per_point,
        "per_trap": per_trap,
        "n_points_hit": sum(points_hit),
        "n_points_total": len(points),
        "n_traps_triggered": sum(traps_triggered),
        "n_traps_total": len(traps),
        "panel": panel["panel"],
        "empty_conclusion": False,
    }
