"""Fold all the extra result-streams into report/data/results.json under `extras`.

Adds: the 13-model cross-family sweep (reasoning-immunity), the G–J alternative-stance
conditions, and the trajectory dynamics (per-token PR/L0/novelty, Lyapunov, attention
range incl. the surface-rhythm controls). Idempotent; run after gowexp.analyze.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from .scoring import extract_answer, parse_yes_no
from .schema import Item, read_jsonl
from .tasks import REGISTRY

_REPO = Path(__file__).resolve().parents[2]
RUNS = _REPO / "data" / "runs"
REPORT = _REPO / "report" / "data" / "results.json"
REASONING = {"deepseek.r1-v1:0", "mistral.magistral-small-2509"}


def _items() -> dict[str, Item]:
    return {d["id"]: Item.from_dict(d) for d in read_jsonl(_REPO / "data" / "items.jsonl")}


def _headline(it: Item, sc: dict):
    t = it.task
    if t == "nonmonotonic":
        return float(sc["correct_final"])
    if t == "observable":
        return float(sc["accuracy"])
    if t == "agency":
        return float(sc["role_accuracy"])
    if t == "epistemic" and sc.get("knowable") is False:
        return float(not sc["confabulated"])
    return None


def cross_family_sweep(items) -> dict:
    """13-model A/D/E sweep on binary tasks -> per-model D-A + reasoning split."""
    rows = [json.loads(l) for l in open(RUNS / "black" / "generations.jsonl")
            if json.loads(l).get("decode") == "sample" and json.loads(l)["condition"] in "ADE"]
    cell = defaultdict(list)
    for r in rows:
        it = items.get(r["item_id"])
        if it is None or it.task == "correlative":
            continue
        h = _headline(it, REGISTRY[it.task].score(it, extract_answer(r["text"]), r["text"]))
        if h is not None:
            cell[(r["model"], r["condition"])].append(h)
    models = sorted({m for m, _ in cell})
    per_model = {}
    for m in models:
        a = np.mean(cell[(m, "A")]) if cell[(m, "A")] else None
        d = np.mean(cell[(m, "D")]) if cell[(m, "D")] else None
        e = np.mean(cell[(m, "E")]) if cell[(m, "E")] else None
        if a is None or d is None:
            continue
        per_model[m] = {"A": float(a), "D": float(d), "E": float(e) if e is not None else None,
                        "D_minus_A": float(d - a), "reasoning": m in REASONING}
    r_da = [v["D_minus_A"] for v in per_model.values() if v["reasoning"]]
    n_da = [v["D_minus_A"] for v in per_model.values() if not v["reasoning"]]
    return {"n_models": len(per_model), "per_model": per_model,
            "mean_D_minus_A_reasoning": float(np.mean(r_da)) if r_da else None,
            "mean_D_minus_A_nonreasoning": float(np.mean(n_da)) if n_da else None,
            "interpretation": ("Reasoning-trained models are immune to the Gowith tax "
                               "(mean D-A ~0) while non-reasoning models pay it regardless of "
                               "scale; capability-via-reasoning protects, capability-via-scale "
                               "does not.")}


_KW = re.compile(r"feedback|mutual|loop|both .* (?:influence|affect)|reinforc|each other", re.I)


def alt_conditions(items) -> dict:
    """G-J alternative-stance conditions: can other prompts match Gowith? (Gemma)."""
    f = RUNS / "alt" / "generations.jsonl"
    if not f.exists():
        return {}
    rows = list(read_jsonl(f))
    nm = defaultdict(list)
    corr = defaultdict(list)
    kw = defaultdict(list)
    for r in rows:
        it = items.get(r["item_id"])
        if it is None:
            continue
        if it.task == "nonmonotonic":
            nm[r["condition"]].append(parse_yes_no(extract_answer(r["text"])) == it.gold["answer"])
        elif it.task == "correlative":
            sc = REGISTRY["correlative"].score(it, extract_answer(r["text"]), r["text"])
            if sc.get("rubric_score") is not None:
                corr[r["condition"]].append(sc["rubric_score"])
            kw[r["condition"]].append(bool(_KW.search(extract_answer(r["text"]).lower())))
    labels = {"A": "plain", "D": "Gowith", "G": "consider-opposite", "H": "causal-DAG",
              "I": "systems-persona", "J": "calibration"}
    out = {}
    for c in labels:
        out[c] = {"label": labels[c],
                  "nonmono_acc": float(np.mean(nm[c])) if nm[c] else None,
                  "correlative_rubric": float(np.mean(corr[c])) if corr[c] else None,
                  "keyword_rate": float(np.mean(kw[c])) if kw[c] else None}
    return {"conditions": out,
            "interpretation": ("A plain systems-scientist persona (I) matches/beats Gowith on the "
                               "goopy task with less crisp-task tax — the one upside is the stance, "
                               "reachable without the conlang. High-correlative conditions also use "
                               "the most rubric keywords (partial keyword-circularity).")}


def trajectory() -> dict:
    """Per-token dynamics, Lyapunov, attention-range (+ surface-rhythm controls)."""
    out = {}
    for name, fn in [("pertoken", "pertoken.json"), ("lyapunov", "lyapunov.json"),
                     ("attn_range", "attn_range.json")]:
        p = RUNS / "white" / fn
        if p.exists():
            out[name] = json.loads(p.read_text()).get("agg", json.loads(p.read_text()))
    out["interpretation"] = (
        "Gowith templates LOCAL token dynamics — lower effective dimensionality "
        "(participation ratio), sparser per-token features, locally contractive (Lyapunov) — "
        "and shows more long-range attention mass. BUT: the long-range effect is largely an "
        "input-length confound (padded-plain B reaches as far), and surface-rhythm controls "
        "(hyphen-rhythm Y, pig-latin P) reproduce part of the long-range push with short plain "
        "inputs and no relational grammar — while word-per-line (W) KILLS it. So the signature "
        "is a generic 'connected-rhythm templating' effect, partly reproduced by orthogonal "
        "surface manipulations, not a Gowith-specific circuit. All n=8-12; suggestive not proven.")
    return out


def main() -> None:
    items = _items()
    R = json.loads(REPORT.read_text())
    R["extras"] = {
        "cross_family_13": cross_family_sweep(items),
        "alt_conditions": alt_conditions(items),
        "trajectory": trajectory(),
        "caveats": {
            "opus_not_tested": ("The model that made the original 'felt easier' claim (Claude "
                                "Opus 4.8) could not be tested: no Bedrock access, no API credits. "
                                "Our reasoning-immunity finding predicts it would NOT be taxed."),
            "white_box_one_model": ("Mechanistic data is from ONE model (Gemma-3-12B-it, "
                                    "non-reasoning). The reasoning-model mechanistic comparison "
                                    "(R1-Distill-Qwen-14B) is future work — the GPU box "
                                    "auto-terminated on its cost-guard before it ran."),
        },
    }
    def _finite(o):
        if isinstance(o, float):
            return o if np.isfinite(o) else None
        if isinstance(o, dict):
            return {k: _finite(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_finite(v) for v in o]
        return o

    REPORT.write_text(json.dumps(_finite(R), indent=1))
    cf = R["extras"]["cross_family_13"]
    print(f"folded extras into results.json")
    print(f"  cross-family: {cf['n_models']} models, "
          f"reasoning D-A={cf['mean_D_minus_A_reasoning']:+.3f} vs "
          f"non-reasoning {cf['mean_D_minus_A_nonreasoning']:+.3f}")
    print(f"  alt-conditions + trajectory + caveats folded")


if __name__ == "__main__":
    main()
