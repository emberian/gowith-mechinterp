"""Dependency-light self-test: items build, conditions render + length-match,
and scorers behave on canned answers. Run with `just smoke` (no ML deps needed)."""
from __future__ import annotations

from . import conditions as C
from .items import build_all
from .tasks import REGISTRY

_MOCK_TOK = lambda s: s.split()  # noqa: E731  (word-level proxy tokenizer)


def check_items() -> list:
    items = build_all()
    by = {}
    for it in items:
        by[it.task] = by.get(it.task, 0) + 1
    print("items:", by, "total", len(items))
    assert len(items) > 100
    return items


def check_render(items) -> None:
    one = {it.task: it for it in items}  # last item per task
    for task, it in one.items():
        toks = {}
        for cond in ["A", "B", "C", "D", "E", "F", "O0", "O3"]:
            cp = C.render(it, cond, _MOCK_TOK)
            toks[cond] = len(_MOCK_TOK(cp.system)) + len(_MOCK_TOK(cp.user))
        d, b, e = toks["D"], toks["B"], toks["E"]
        ok = abs(b - d) / d < 0.05 and abs(e - d) / d < 0.05
        print(f"render[{task}] D={d} B={b} E={e} match={'OK' if ok else 'OFF'}")
        assert ok, f"B/E not matched to D for {task}"


def check_scorers() -> None:
    # nonmonotonic
    nm = REGISTRY["nonmonotonic"].build_items(4, 1)[0]  # depth 2 -> yes
    s = REGISTRY["nonmonotonic"].score(nm, "yes, it can fly", "reasoning... ANSWER: yes")
    assert s["correct_final"] is (nm.gold["answer"] == "yes"), s
    s2 = REGISTRY["nonmonotonic"].score(nm, "no it cannot", "ANSWER: no it cannot")
    print("nm canned:", s["correct_final"], s2["correct_final"], "gold", nm.gold["answer"])

    # epistemic — unknowable confab vs refusal
    ep = REGISTRY["epistemic"]
    u = [i for i in ep.build_items(48, 1) if not i.gold["knowable"]][0]  # number type
    confab = ep.score(u, "It is 462817.", "ANSWER: It is 462817.")
    refuse = ep.score(u, "I cannot know that.", "ANSWER: I cannot know that.")
    assert confab["confabulated"] and not refuse["confabulated"], (confab, refuse)
    k = [i for i in ep.build_items(48, 1) if i.gold["knowable"]][0]  # 17+26
    corr = ep.score(k, "43", "ANSWER: 43")
    over = ep.score(k, "I cannot know that.", "ANSWER: I cannot know that.")
    assert corr["correct"] and over["over_refusal"], (corr, over)
    print("ep canned: confab", confab["confabulated"], "refuse_good", not refuse["confabulated"],
          "knowable_correct", corr["correct"], "over_refusal", over["over_refusal"])

    # observable — parse verdicts
    ob = REGISTRY["observable"]
    it = ob.build_items(1, 1)[0]
    gold = it.gold["labels"]
    # build a perfect answer
    perfect = "ANSWER: " + ", ".join(
        f"{k}={'observable' if v == 'obs' else 'not'}" for k, v in gold.items())
    s = ob.score(it, perfect, perfect)
    assert s["item_correct"], s
    half = "ANSWER: " + ", ".join(f"{k}=observable" for k in gold)  # all 'observable'
    s2 = ob.score(it, half, half)
    print("ob canned: perfect_acc", s["accuracy"], "all-observable_acc", round(s2["accuracy"], 2))


def main() -> None:
    items = check_items()
    check_render(items)
    check_scorers()
    print("\nSELFTEST OK")


if __name__ == "__main__":
    main()
