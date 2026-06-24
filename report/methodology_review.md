# Independent methodology review (GPT-5.5 via codex), 2026-06-24

Adversarial red-team of this experiment's design, scoring, and conclusions, run with `codex exec` against the repo. The README headline and limitations were revised in response. Verbatim below.

Bottom line: the crisp-task degradation is real in the recorded metrics, but it is not cleanly attributable to “Gowith grammar.” The positive “tool” claim is weak: it is a small, n=40, judge-scored correlative effect with CI crossing zero, matched by pseudo-Gowith, and entangled with answer length/rubric wording.

**A. Confounds**
- **CRITICAL:** A-vs-D is not a clean register contrast. A is plain step-by-step ([conditions.py](/Users/ember/dev/gowexp/src/gowexp/conditions.py:88)); D is a long system skill plus an instruction to use currents, agency, and `know-not-settle` ([conditions.py](/Users/ember/dev/gowexp/src/gowexp/conditions.py:82)). That tests a prompt package, not “register.”
- **MAJOR:** B is not a fair “tokens only” control. Its filler says “nothing hidden,” “not a trick,” “steady pace,” etc. ([conditions.py](/Users/ember/dev/gowexp/src/gowexp/conditions.py:41)). Also B pads the user prompt; D/E put a huge skill in system. Same approximate token count, different instruction channel and semantics.
- **MAJOR:** E is not a clean weird-register control. It is one line-shuffled scramble, preserves Gowith lexemes/markers, and produces very different output behavior. In raw white generations, mean output tokens were A 384, B 188, D 221, E 500. That alone can shift scores.
- **MAJOR:** Agency is ceilinged. A/B/C/F are all 1.0; D is 0.877 ([results.json](/Users/ember/dev/gowexp/report/data/results.json:228)). This task can show D breaking an easy parser/task, not Gowith “home turf” benefit.
- **MAJOR:** Correlative is explicitly not binary-scored, despite README saying tasks are “all binary-scored” ([README.md](/Users/ember/dev/gowexp/README.md:64), [correlative.py](/Users/ember/dev/gowexp/src/gowexp/tasks/correlative.py:3)). It is a rubric/LLM-judge task in the exact domain the prompt teaches.

**B. Scoring Validity**
- **CRITICAL:** The nonmonotonic scorer is not style-blind to hedging. `parse_yes_no` maps any non-leading `not` to “no” ([scoring.py](/Users/ember/dev/gowexp/src/gowexp/scoring.py:67)). D had 30 gold-yes items parsed as no, versus 3 for A/B/C/E and 6 for F. Answers like “can, but not certain” are exactly the Gowith-induced style and get penalized.
- **MAJOR:** `extract_answer` only keeps the first line after `ANSWER:` and falls back to the last three lines if no sigil exists ([scoring.py](/Users/ember/dev/gowexp/src/gowexp/scoring.py:39)). That can drop multi-line conclusions and leaks register/reasoning into scoring on truncation/no-sigil cases.
- **MAJOR:** The judge firewall is weaker than claimed. It hides reasoning, but not final-answer length, lexical overlap, or rubric-shaped wording. D/E correlative conclusions are longer than A/B and more likely to say “feedback loop,” “mutual,” etc. The judge only checks whether the conclusion asserts prewritten points/traps ([judge.py](/Users/ember/dev/gowexp/src/gowexp/judge.py:186)).
- **MAJOR:** Correlative traps are asymmetric. Example `co-000`: A’s “algae bloom caused cooling” gets 0 points but 0 traps, because traps target the reporter’s opposite one-way claim. Wrong one-arrow answers can escape penalty.

**C. Statistical Issues**
- **MAJOR:** No multiple-comparison correction across 5 tasks and many contrasts. The code treats “CI excludes 0” as decisive per contrast ([analyze.py](/Users/ember/dev/gowexp/src/gowexp/analyze.py:100)).
- **MAJOR:** The correlative headline is underpowered and honestly non-significant: D-A = +0.125, CI [-0.05, +0.30], n=40 ([results.json](/Users/ember/dev/gowexp/report/data/results.json:525)). D-E = +0.0125, CI [-0.1125, +0.15], so notation-specific benefit is unsupported ([results.json](/Users/ember/dev/gowexp/report/data/results.json:549)).
- **MAJOR:** The output-budget claim is not “flat-to-negative.” Correlative goes O0 0.05, O1 0.1625, O2 0.55, O3 0.3875 ([results.json](/Users/ember/dev/gowexp/report/data/results.json:800)). Also 59 white-box generations hit 1024 tokens, mostly O2/O3 correlative/nonmonotonic, despite config saying it “must NOT clip” ([experiment.yaml](/Users/ember/dev/gowexp/config/experiment.yaml:24)).
- **MINOR:** Rate-CI seeds use Python `hash((task,c))`, which is process-randomized unless hash seeding is fixed ([analyze.py](/Users/ember/dev/gowexp/src/gowexp/analyze.py:115)).

**D. Alternative Explanations**
- D hurts crisp tasks because the model is spending effort obeying an unfamiliar conlang/system skill, not because relational grammar taxes reasoning.
- The nonmonotonic tax is partly a parser/hedging artifact.
- Correlative “help” may be rubric-keyword prompting: D/E push feedback/mutual-causation language, and the judge rewards those propositions.
- B’s failure does not refute token budget; it refutes this specific filler-padding recipe.
- E≈D on correlative suggests the effect is not Gowith notation. It may be “relational keyword soup plus longer conclusion.”

**E. Data Support**
- Supported: D is worse on nonmonotonic, agency, observable. Nonmonotonic A 0.944 vs D 0.618 ([results.json](/Users/ember/dev/gowexp/report/data/results.json:29)); agency A 1.0 vs D 0.877 ([results.json](/Users/ember/dev/gowexp/report/data/results.json:228)).
- Not supported: “Gowith helps overall.” The synthesis says `gowith_helps_overall` is 0/5 tasks ([results.json](/Users/ember/dev/gowexp/report/data/results.json:617)).
- Not supported: “in-domain benefit.” The in-domain mean D-A is 0.0009, basically zero ([results.json](/Users/ember/dev/gowexp/report/data/results.json:631)).
- Not supported: mechanistic specificity. E-overlap is 0.404 and explicitly interpreted as “HIGH overlap... register/novelty effect, NOT Gowith semantics” ([results.json](/Users/ember/dev/gowexp/report/data/results.json:1158)), yet the report later calls it “low E-overlap” and claims causal specificity ([report.typ](/Users/ember/dev/gowexp/report/report.typ:722)).
- Mechanistic labels are deflationary: top D features include “base64 encoding,” “sentence structure and grammar,” “commands for actions,” “counting words,” “questions and responses,” and “array or object access” ([np_labels.json](/Users/ember/dev/gowexp/data/runs/np_labels.json:32)).
- Steering is oversold. Gowith confabulation goes 0.094 → 0.057 → 0.094 → 0.075 → 0.208, not monotone improvement; pseudo at 16 is better than Gowith (0.038 vs 0.208) ([results.json](/Users/ember/dev/gowexp/report/data/results.json:3063)).

**F. Fix Before Publishing**
The single necessary caveat: “This run shows a D-prompt-package degradation on several easy/templated tasks and a non-significant correlative rubric lift matched by pseudo-Gowith. It does not isolate Gowith grammar, does not refute output-token budget, and does not establish mechanistic causal specificity.”

Before making the stronger claim, rerun with a style-robust parser, multiple pseudo controls, no truncation, harder non-ceiling agency items, and correlative scoring that is independent of LLM-rubric keyword matching.
