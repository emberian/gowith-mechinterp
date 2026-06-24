"""CLI: inspect one SAE feature from our captured activations.

    python -m gowexp.feature 542          # primary layer
    python -m gowexp.feature 542 --layer 31

Prints the Neuronpedia dashboard URL + per-condition / per-task mean activation of that
feature across our 3600 generations (from the committed sae_means_L*.npz), so the channel
can answer "where does Gowith engage feature N?" without a GPU.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .explore import np_url
from .schema import load_config

_REPO = Path(__file__).resolve().parents[2]
RUNS = _REPO / "data" / "runs" / "white"


def main(argv: list[str] | None = None) -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("feature", type=int)
    ap.add_argument("--layer", type=int, default=cfg["white_box"]["sae_primary_layer"])
    a = ap.parse_args(argv)
    f, layer = a.feature, a.layer

    npz = RUNS / f"sae_means_L{layer}.npz"
    ri = RUNS / "row_index.json"
    if not (npz.exists() and ri.exists()):
        raise SystemExit(f"missing {npz} or {ri}; fetch the run data first")
    means = np.load(npz)["means"][:, f].astype(np.float32)
    rows = json.loads(ri.read_text())

    print(f"\nfeature #{f} @ layer {layer}")
    print(f"neuronpedia: {np_url(layer, f)}\n")

    by_cond: dict[str, list[float]] = defaultdict(list)
    by_task: dict[str, list[float]] = defaultdict(list)
    by_ct: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        v = float(means[r["row"]])
        by_cond[r["condition"]].append(v)
        by_task[r["task"]].append(v)
        by_ct[(r["task"], r["condition"])].append(v)

    def line(d, keys):
        return "  ".join(f"{k}={np.mean(d[k]):.2f}" for k in keys if d.get(k))

    print("mean activation by CONDITION:")
    print("  " + line(by_cond, ["A", "B", "C", "D", "E", "F", "O0", "O1", "O2", "O3"]))
    print("\nmean activation by TASK:")
    print("  " + line(by_task, ["nonmonotonic", "epistemic", "observable", "agency", "correlative"]))
    print("\nGowith (D) minus plain (A), per task:")
    for t in ["nonmonotonic", "epistemic", "observable", "agency", "correlative"]:
        d, aa = by_ct.get((t, "D")), by_ct.get((t, "A"))
        if d and aa:
            print(f"  {t:14} {np.mean(d) - np.mean(aa):+.2f}")
    print()


if __name__ == "__main__":
    main()
