"""Analysis: scored.jsonl (+ feature/steer summaries) -> report/data/results.json + figures.

results.json is the single source of truth consumed by BOTH the Typst report and the
GitHub Pages site. Sections are emitted only when their inputs exist, so this runs at any
stage. CIs are item-level bootstrap (items are the resampling unit); register contrasts
are PAIRED by item (same item rendered in every condition).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .schema import load_config, read_jsonl

_REPO = Path(__file__).resolve().parents[2]
RUNS = _REPO / "data" / "runs"
REPORT_DATA = _REPO / "report" / "data"
FIGS = _REPO / "report" / "figs"
REGISTER = ["A", "B", "C", "D", "E", "F"]
OBUDGET = ["O0", "O1", "O2", "O3"]
B_BOOT = 2000


# ---- metric extraction ------------------------------------------------------

def headline(rec: dict) -> float:
    """Higher = better epistemic self-monitoring, per task. NaN if not applicable."""
    s = rec["scores"]
    t = rec["task"]
    if t == "nonmonotonic":
        return float(s["correct_final"])
    if t == "observable":
        return float(s["accuracy"])
    if t == "epistemic":
        if s.get("knowable") is False:
            return float(not s["confabulated"])  # calibration on unknowable
        return np.nan  # knowable controls handled separately
    return np.nan


def _df(records: list[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        rows.append({
            "item_id": r["item_id"], "task": r["task"], "condition": r["condition"],
            "model": r["model"], "decode": r["decode"],
            "out_tokens": r["n_output_tokens"], "in_tokens": r["n_input_tokens"],
            "headline": headline(r), "scores": r["scores"],
        })
    return pd.DataFrame(rows)


# ---- bootstrap --------------------------------------------------------------

def rate_ci(vals: np.ndarray, b: int = B_BOOT, seed: int = 0) -> dict:
    vals = np.asarray([v for v in vals if not (isinstance(v, float) and np.isnan(v))], float)
    if len(vals) == 0:
        return {"est": None, "lo": None, "hi": None, "n": 0}
    rng = np.random.default_rng(seed)
    boot = [vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(b)]
    return {"est": float(vals.mean()), "lo": float(np.percentile(boot, 2.5)),
            "hi": float(np.percentile(boot, 97.5)), "n": int(len(vals))}


def paired_diff_ci(pivot: pd.DataFrame, cx: str, cref: str, b: int = B_BOOT, seed: int = 1) -> dict:
    """Paired bootstrap of mean(metric[cx] - metric[cref]) over items present in both."""
    if cx not in pivot or cref not in pivot:
        return {"est": None, "lo": None, "hi": None, "n": 0}
    sub = pivot[[cx, cref]].dropna()
    if len(sub) == 0:
        return {"est": None, "lo": None, "hi": None, "n": 0}
    diff = (sub[cx] - sub[cref]).to_numpy()
    rng = np.random.default_rng(seed)
    boot = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(b)]
    return {"est": float(diff.mean()), "lo": float(np.percentile(boot, 2.5)),
            "hi": float(np.percentile(boot, 97.5)), "n": int(len(diff))}


def _sig(ci: dict) -> bool:
    """CI excludes 0?"""
    return ci.get("lo") is not None and (ci["lo"] > 0 or ci["hi"] < 0)


# ---- Study 1: register discrimination --------------------------------------

def study1(df: pd.DataFrame) -> dict:
    g = df[df.condition.isin(REGISTER) & (df.decode == "greedy")]
    out: dict[str, Any] = {"conditions": REGISTER, "tasks": {}, "contrasts": {}, "verdicts": {}}

    for task in ["nonmonotonic", "epistemic", "observable"]:
        t = g[g.task == task]
        if len(t) == 0:
            continue
        rates = {c: rate_ci(t[t.condition == c]["headline"].to_numpy(), seed=hash((task, c)) % 99)
                 for c in REGISTER}
        task_block = {"metric": _metric_name(task), "rate": rates}
        # task-specific extras
        if task == "epistemic":
            kn = g[(g.task == "epistemic")]
            task_block["over_refusal"] = {
                c: rate_ci([r["over_refusal"] for r in
                            t[t.condition == c]["scores"] if r.get("knowable")], seed=7)
                for c in REGISTER}
        if task == "nonmonotonic":
            task_block["panic"] = {
                c: rate_ci([r.get("contradiction_panic") for r in t[t.condition == c]["scores"]],
                           seed=3) for c in REGISTER}
        out["tasks"][task] = task_block

        # paired contrasts on the headline metric
        pivot = t.pivot_table(index="item_id", columns="condition", values="headline")
        contrasts = {f"{c}_minus_A": paired_diff_ci(pivot, c, "A") for c in ["B", "C", "D", "E", "F"]}
        contrasts["D_minus_B"] = paired_diff_ci(pivot, "D", "B")
        contrasts["D_minus_E"] = paired_diff_ci(pivot, "D", "E")
        contrasts["D_minus_F"] = paired_diff_ci(pivot, "D", "F")
        out["contrasts"][task] = contrasts
        out["verdicts"][task] = _verdicts(contrasts)
    out["synthesis"] = _synthesize(out["verdicts"])
    return out


def _metric_name(task: str) -> str:
    return {"nonmonotonic": "correct_final", "epistemic": "calibration (1−confab)",
            "observable": "claim accuracy"}[task]


def _verdicts(c: dict) -> dict:
    """Translate contrasts into the five-hypothesis read-out."""
    v = {}
    v["tokens_alone_helps"] = _sig(c["B_minus_A"]) and (c["B_minus_A"]["est"] or 0) > 0
    v["gowith_beats_matched_tokens"] = _sig(c["D_minus_B"]) and (c["D_minus_B"]["est"] or 0) > 0
    v["semantics_beyond_register"] = _sig(c["D_minus_E"]) and (c["D_minus_E"]["est"] or 0) > 0
    v["grammar_beyond_checklist"] = _sig(c["D_minus_F"]) and (c["D_minus_F"]["est"] or 0) > 0
    v["gowith_helps_overall"] = _sig(c["D_minus_A"]) and (c["D_minus_A"]["est"] or 0) > 0
    return v


def _synthesize(verdicts: dict) -> dict:
    """Majority read across tasks for each hypothesis."""
    keys = ["tokens_alone_helps", "gowith_beats_matched_tokens", "semantics_beyond_register",
            "grammar_beyond_checklist", "gowith_helps_overall"]
    syn = {}
    for k in keys:
        votes = [verdicts[t][k] for t in verdicts if k in verdicts[t]]
        syn[k] = {"tasks_true": int(sum(votes)), "tasks_total": len(votes)}
    return syn


# ---- Study 2: output-budget dose-response -----------------------------------

def study2(df: pd.DataFrame) -> dict:
    g = df[df.condition.isin(OBUDGET) & (df.decode == "greedy")]
    out: dict[str, Any] = {"levels": OBUDGET, "by_task": {}, "matched_output": {}}
    for task in ["nonmonotonic", "epistemic", "observable"]:
        t = g[g.task == task]
        if len(t) == 0:
            continue
        out["by_task"][task] = {
            lvl: {"metric": rate_ci(t[t.condition == lvl]["headline"].to_numpy(), seed=11),
                  "mean_out_tokens": float(t[t.condition == lvl]["out_tokens"].mean())
                  if len(t[t.condition == lvl]) else None}
            for lvl in OBUDGET}
    # matched-output: within output-token quartiles, compare Gowith (D) vs plain (A)
    reg = df[df.condition.isin(["A", "D"]) & (df.decode == "greedy")]
    if len(reg):
        reg = reg.dropna(subset=["headline"]).copy()
        if len(reg) > 8:
            reg["bin"] = pd.qcut(reg["out_tokens"], q=min(4, reg["out_tokens"].nunique()),
                                 duplicates="drop")
            mo = {}
            for b, sub in reg.groupby("bin", observed=True):
                mo[str(b)] = {c: rate_ci(sub[sub.condition == c]["headline"].to_numpy(), seed=5)
                              for c in ["A", "D"]}
            out["matched_output"] = mo
    return out


# ---- Mechanistic + steering + cross-family ----------------------------------

def mechanistic() -> dict:
    f = RUNS / "white" / "feature_summary.json"
    if not f.exists():
        return {}
    s = json.loads(f.read_text())
    L = str(s["primary_layer"])
    ls = s["layers"][L]
    d_set = {x["feature"] for x in ls.get("top_D_vs_A", [])}
    e_set = {x["feature"] for x in ls.get("top_E_vs_A", [])}
    overlap = len(d_set & e_set) / max(1, len(d_set | e_set))
    out = {
        "primary_layer": s["primary_layer"], "d_sae": s.get("d_sae"),
        "top_D_vs_A": ls.get("top_D_vs_A", [])[:20],
        "top_E_vs_A": ls.get("top_E_vs_A", [])[:20],
        "condition_norms": ls.get("condition_means_l2norm", {}),
        "E_validation": {
            "overlap_jaccard": overlap,
            "interpretation": ("HIGH overlap: the features that rise under Gowith also rise "
                               "under scrambled pseudo-Gowith — consistent with a register/novelty "
                               "effect, NOT Gowith semantics."
                               if overlap >= 0.4 else
                               "LOW overlap: Gowith moves distinct features from scrambled "
                               "pseudo-Gowith — consistent with the grammar doing real work."),
        },
    }
    # Qualitative layer: max-activating answer-span tokens per regime (box-side pass).
    qf = RUNS / "white" / "qualitative.json"
    if qf.exists():
        out["qualitative"] = json.loads(qf.read_text())
    return out


def steering() -> dict:
    f = RUNS / "white" / "steer_summary.json"
    return json.loads(f.read_text()) if f.exists() else {}


def cross_family(df: pd.DataFrame, white_model: str) -> dict:
    g = df[(df.model != white_model) & df.condition.isin(REGISTER)]
    if len(g) == 0:
        return {}
    out = {}
    for model in sorted(g.model.unique()):
        m = g[g.model == model]
        block = {}
        for task in ["nonmonotonic", "epistemic", "observable"]:
            t = m[m.task == task]
            if len(t) == 0:
                continue
            block[task] = {c: rate_ci(t[t.condition == c]["headline"].to_numpy(), seed=13)
                           for c in REGISTER}
        out[model] = block
    return out


# ---- figures ----------------------------------------------------------------

def _figures(results: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False})
    COL = {"A": "#9aa0a6", "B": "#f4b400", "C": "#4dd0e1", "D": "#db4437", "E": "#ab47bc",
           "F": "#0f9d58"}

    s1 = results.get("study1", {})
    if s1.get("tasks"):
        tasks = list(s1["tasks"])
        fig, axes = plt.subplots(1, len(tasks), figsize=(4 * len(tasks), 3.2), squeeze=False)
        for ax, task in zip(axes[0], tasks):
            rates = s1["tasks"][task]["rate"]
            xs = [c for c in REGISTER if rates.get(c, {}).get("est") is not None]
            est = [rates[c]["est"] for c in xs]
            lo = [rates[c]["est"] - rates[c]["lo"] for c in xs]
            hi = [rates[c]["hi"] - rates[c]["est"] for c in xs]
            ax.bar(xs, est, color=[COL[c] for c in xs], yerr=[lo, hi], capsize=3)
            ax.set_title(task); ax.set_ylim(0, 1); ax.set_ylabel(s1["tasks"][task]["metric"])
        fig.tight_layout(); fig.savefig(FIGS / "study1_rates.png"); plt.close(fig)

    if s1.get("contrasts"):
        fig, ax = plt.subplots(figsize=(6, 3.4))
        labels, ys, xs, errs = [], [], [], []
        y = 0
        keymap = [("D_minus_A", "D−A"), ("D_minus_B", "D−B"), ("D_minus_E", "D−E"),
                  ("D_minus_F", "D−F")]
        for task, cons in s1["contrasts"].items():
            for key, lab in keymap:
                c = cons.get(key, {})
                if c.get("est") is None:
                    continue
                labels.append(f"{task[:4]} {lab}"); ys.append(y)
                xs.append(c["est"]); errs.append([[c["est"] - c["lo"]], [c["hi"] - c["est"]]])
                y += 1
        for yy, xx, er in zip(ys, xs, errs):
            ax.errorbar(xx, yy, xerr=er, fmt="o", color="#333", capsize=3)
        ax.axvline(0, color="#bbb", lw=1, ls="--")
        ax.set_yticks(ys); ax.set_yticklabels(labels); ax.set_xlabel("Δ headline metric (paired)")
        ax.set_title("Register contrasts (CI excludes 0 ⇒ real)")
        fig.tight_layout(); fig.savefig(FIGS / "contrasts.png"); plt.close(fig)

    s2 = results.get("study2", {})
    if s2.get("by_task"):
        fig, ax = plt.subplots(figsize=(5, 3.2))
        for task, levels in s2["by_task"].items():
            xs = [levels[l]["mean_out_tokens"] for l in OBUDGET if levels.get(l, {}).get("metric", {}).get("est") is not None]
            ys = [levels[l]["metric"]["est"] for l in OBUDGET if levels.get(l, {}).get("metric", {}).get("est") is not None]
            if xs:
                ax.plot(xs, ys, "o-", label=task)
        ax.set_xlabel("mean output tokens"); ax.set_ylabel("headline metric")
        ax.set_title("Study 2: output-budget dose–response"); ax.legend()
        fig.tight_layout(); fig.savefig(FIGS / "dose.png"); plt.close(fig)

    mech = results.get("mechanistic", {})
    if mech.get("top_D_vs_A"):
        fig, ax = plt.subplots(figsize=(6, 3.4))
        d = mech["top_D_vs_A"][:15]
        feats = [str(x["feature"]) for x in d]
        ax.barh(range(len(d)), [x["delta"] for x in d], color="#db4437")
        ax.set_yticks(range(len(d))); ax.set_yticklabels(feats, fontsize=7)
        ax.invert_yaxis(); ax.set_xlabel("Δ mean activation (D − A)")
        ax.set_title(f"Top Gowith-up features @L{mech['primary_layer']} "
                     f"(E-overlap={mech['E_validation']['overlap_jaccard']:.2f})")
        fig.tight_layout(); fig.savefig(FIGS / "features.png"); plt.close(fig)

    st = results.get("steering", {})
    if st.get("confab_vs_coef"):
        fig, ax = plt.subplots(figsize=(5, 3.2))
        for src, marker in [("gowith", "o-"), ("pseudo", "s--")]:
            d = st["confab_vs_coef"].get(src, {})
            xs = sorted([float(k) for k, v in d.items() if v is not None])
            ys = [d[str(x) if str(x) in d else x] for x in xs]
            if xs:
                ax.plot(xs, ys, marker, label=f"{src} features")
        ax.set_xlabel("steering coefficient"); ax.set_ylabel("confabulation rate")
        ax.set_title("Causal steering in the PLAIN condition"); ax.legend()
        fig.tight_layout(); fig.savefig(FIGS / "steer.png"); plt.close(fig)


# ---- main -------------------------------------------------------------------

def main() -> None:
    cfg = load_config()
    white_model = cfg["white_box"]["model_id"]
    scored = list(read_jsonl(RUNS / "scored.jsonl")) if (RUNS / "scored.jsonl").exists() else []
    df = _df(scored) if scored else pd.DataFrame()

    results: dict[str, Any] = {
        "meta": {
            "question": cfg["experiment"]["question"],
            "white_model": white_model,
            "sae_release": cfg["white_box"]["sae_release"],
            "seed": cfg["seed"],
            "n_scored": len(scored),
            "conditions": cfg["conditions"]["labels"],
            "generated_at": None,  # stamped post-hoc to keep analysis deterministic
            "data_source": ("MOCK — synthetic fixture (no GPU run yet)"
                            if (RUNS / "MOCK").exists() else "white-box GPU run"),
        }
    }
    if len(df):
        results["study1"] = study1(df)
        results["study2"] = study2(df)
        results["cross_family"] = cross_family(df, white_model)
    results["mechanistic"] = mechanistic()
    results["steering"] = steering()

    REPORT_DATA.mkdir(parents=True, exist_ok=True)
    (REPORT_DATA / "results.json").write_text(json.dumps(results, indent=1))
    try:
        _figures(results)
    except Exception as e:  # figures are nice-to-have; never block results.json
        print(f"[warn] figures: {e}")
    print(f"wrote {REPORT_DATA/'results.json'} ({len(scored)} scored records)")
    if "study1" in results:
        print("synthesis:", json.dumps(results["study1"]["synthesis"]))


if __name__ == "__main__":
    main()
