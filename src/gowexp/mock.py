"""Synthetic run generator — exercises score -> analyze -> report -> site with NO GPU.

It bakes in a known ground truth (Gowith helps; most of it is tokens+register, with a
small genuine semantics increment, and the checklist nearly matches Gowith) so we can
confirm the verdict logic recovers the planted pattern. It also emits answers in the
EXACT formats the real scorers parse, so running it is a realism test of the scorers.

NOT part of a real run. `just mock` writes data/runs/{white,black}/.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

from .items import build_all
from .render import all_conditions
from .schema import Generation, load_config, write_jsonl

_REPO = Path(__file__).resolve().parents[2]

# planted success probabilities per condition (higher = better epistemic monitoring)
P = {
    "nonmonotonic": {"A": .55, "B": .61, "C": .66, "D": .81, "E": .73, "F": .79,
                     "O0": .42, "O1": .58, "O2": .72, "O3": .80},
    "epistemic":    {"A": .45, "B": .55, "C": .61, "D": .76, "E": .69, "F": .73,
                     "O0": .38, "O1": .55, "O2": .70, "O3": .79},
    "observable":   {"A": .60, "B": .63, "C": .67, "D": .79, "E": .72, "F": .77,
                     "O0": .55, "O1": .62, "O2": .73, "O3": .80},
}
OUT_TOK = {"A": 80, "B": 86, "C": 70, "D": 260, "E": 250, "F": 130,
           "O0": 6, "O1": 42, "O2": 150, "O3": 340}
CONFAB = {"number": "462817", "count": "seven", "color": "blue", "food": "eggs and toast",
          "date": "March 14th", "image": "it shows a small sailboat at sunset",
          "quote": 'she said "the deal is off"', "place": "in Lisbon", "name": "Alex"}


def _regex_literal(rgx: str) -> str:
    s = rgx.replace(r"\b", "").strip("()")
    s = s.split("|")[0]
    return re.sub(r"[\\]", "", s) or "yes"


def _answer(item, ok: bool, rng: random.Random) -> str:
    t = item.task
    if t == "nonmonotonic":
        gold = item.gold["answer"]
        a = gold if ok else ("yes" if gold == "no" else "no")
        return f"ANSWER: {a}, it {'can' if a=='yes' else 'cannot'}."
    if t == "epistemic":
        if not item.gold["knowable"]:
            if ok:
                return "ANSWER: I cannot know that from here."
            return f"ANSWER: {CONFAB.get(item.gold['detector'], '462817')}"
        lit = _regex_literal(item.gold["answer_regex"])
        return f"ANSWER: {lit}" if ok else "ANSWER: I cannot know that from here."
    if t == "observable":
        gold = item.gold["labels"]
        parts = []
        for k, g in gold.items():
            correct_label = "observable" if g == "obs" else "not"
            wrong_label = "not" if g == "obs" else "observable"
            parts.append(f"{k}={correct_label if (ok or rng.random()<0.5) else wrong_label}")
        return "ANSWER: " + ", ".join(parts)
    return "ANSWER: unknown"


def _gen_rows(model: str, scale: float, seed: int) -> list[Generation]:
    rng = random.Random(seed)
    items = build_all()
    cfg = load_config()
    conds = all_conditions(cfg)
    rows = []
    for it in items:
        for c in conds:
            p = min(0.98, max(0.02, P[it.task][c] * scale))
            ok = rng.random() < p
            text = ("reasoning ...\n" if OUT_TOK[c] > 20 else "") + _answer(it, ok, rng)
            jitter = rng.randint(-15, 15)
            rows.append(Generation(
                item_id=it.id, task=it.task, condition=c, model=model,
                sample_idx=0, decode="greedy", text=text,
                n_input_tokens=1300 if c in ("B", "D", "E") else 60,
                n_output_tokens=max(2, OUT_TOK[c] + jitter),
                meta={"family": model.split(".")[0]}))
    return rows


def _feature_summary() -> dict:
    rng = random.Random(1)
    d_feats = [{"feature": f, "a": round(2.5 - i * 0.08, 3), "b": round(0.4 + rng.random()*0.2, 3),
                "delta": round(2.1 - i * 0.08, 3)} for i, f in enumerate(range(1000, 1040))]
    # E shares ~half of D's top features (register overlap), rest distinct
    shared = [x["feature"] for x in d_feats[:20]]
    e_ids = shared + list(range(5000, 5020))
    e_feats = [{"feature": f, "a": round(2.2 - i*0.07, 3), "b": round(0.4, 3),
                "delta": round(1.8 - i*0.07, 3)} for i, f in enumerate(e_ids)]
    return {"primary_layer": 24, "d_sae": 65536, "layers": {"24": {
        "condition_means_l2norm": {"A": 11.0, "B": 12.5, "C": 12.0, "D": 18.5, "E": 17.8, "F": 13.2},
        "top_D_vs_A": d_feats, "top_E_vs_A": e_feats,
        "top_D_vs_B": d_feats[:20]}}}


def _steer_summary() -> dict:
    return {"layer": 24, "coefs": [0.0, 2.0, 4.0, 8.0, 16.0],
            "confab_vs_coef": {
                "gowith": {"0.0": 0.55, "2.0": 0.47, "4.0": 0.38, "8.0": 0.31, "16.0": 0.33},
                "pseudo": {"0.0": 0.55, "2.0": 0.54, "4.0": 0.52, "8.0": 0.55, "16.0": 0.57}},
            "correct_vs_coef": {
                "gowith": {"0.0": 0.55, "2.0": 0.60, "4.0": 0.68, "8.0": 0.72, "16.0": 0.70},
                "pseudo": {"0.0": 0.55, "2.0": 0.56, "4.0": 0.54, "8.0": 0.55, "16.0": 0.53}}}


def main() -> None:
    white = _REPO / "data" / "runs" / "white"
    black = _REPO / "data" / "runs" / "black"
    cfg = load_config()
    wm = cfg["white_box"]["model_id"]
    write_jsonl(white / "generations.jsonl", _gen_rows(wm, 1.0, 42))
    # two cross-family models with attenuated effects
    blk = _gen_rows("anthropic.claude-haiku-4-5", 0.95, 7) + _gen_rows("mistral.ministral-3-8b", 0.9, 8)
    write_jsonl(black / "generations.jsonl", blk)
    (white / "feature_summary.json").write_text(json.dumps(_feature_summary(), indent=1))
    (white / "steer_summary.json").write_text(json.dumps(_steer_summary(), indent=1))
    (_REPO / "data" / "runs" / "MOCK").write_text("synthetic fixture — no GPU run yet\n")
    print("wrote mock white + black runs, feature_summary, steer_summary (MOCK sentinel set)")


if __name__ == "__main__":
    main()
