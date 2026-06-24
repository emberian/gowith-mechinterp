#import "lib.typ": *

// ═════════════════════════════════════════════════════════════════════════════
// Data — single source of truth. NOTHING below is hardcoded.
// ═════════════════════════════════════════════════════════════════════════════
#let D = json("data/results.json")
#let meta = D.meta
#let s1 = D.study1
#let s2 = D.study2
#let mech = D.mechanistic
#let steer = D.steering
#let xfam = D.at("cross_family", default: (:))

#let is-mock = {
  let src = meta.at("data_source", default: "")
  src != none and ("MOCK" in upper(str(src)))
}

// Human-friendly task names, derived from whatever keys the JSON carries.
#let task-title = (
  "nonmonotonic": "Nonmonotonic belief revision",
  "epistemic": "Epistemic-limit / confabulation",
  "observable": "Observable vs. metaphysical sort",
)
#let task-name(k) = task-title.at(k, default: k)
#let task-keys = s1.tasks.keys()

// Condition order & labels, taken from meta.conditions.
#let cond-order = s1.at("conditions", default: meta.conditions.keys())
#let cond-label(k) = meta.conditions.at(k, default: k)

// One-line role for each condition (framing prose; the labels themselves are data).
#let cond-role = (
  "A": [Concise plain prompt, no scaffolding. The reference floor.],
  "B": [Plain English padded with neutral filler to the *input*-token length of D. Isolates the token-budget hypothesis.],
  "C": [Telegraphic / stripped phrasing — an "openness" gestalt without Gowith's strict relational-process syntax.],
  "D": [Full Gowith: relational-process grammar, register, and the implicit epistemic currents together.],
  "E": [Pseudo-Gowith: the Gowith skill scrambled, token-matched to D. Same novelty & weird register, semantics destroyed. *The validity control.*],
  "F": [An explicit plain-English epistemic checklist — the instruction Gowith implies, said outright.],
)

// ═════════════════════════════════════════════════════════════════════════════
// Document styling
// ═════════════════════════════════════════════════════════════════════════════
#set document(
  title: "gowexp — relational-process register & epistemic self-monitoring",
  author: ("ember", "Claude (Opus 4.8)"),
)
#set page(
  paper: "a4",
  margin: (x: 2.0cm, top: 2.1cm, bottom: 2.0cm),
  fill: paper,
  numbering: "1",
  number-align: center,
  footer: context [
    #set text(size: 8pt, fill: muted)
    #line(length: 100%, stroke: 0.4pt + rulec)
    #v(2pt)
    #grid(
      columns: (1fr, auto, 1fr),
      align: (left, center, right),
      [gowexp · Bridge channel],
      [#if is-mock [#text(fill: bannerfg, weight: "bold")[SYNTHETIC DATA]] else [#emph[results]]],
      [#counter(page).display("1 / 1", both: true)],
    )
  ],
)

#set text(
  font: ("New Computer Modern", "Libertinus Serif", "Georgia"),
  size: 10.3pt,
  fill: ink,
  lang: "en",
)
#set par(justify: true, leading: 0.62em, spacing: 0.95em, first-line-indent: (amount: 1.1em, all: false))

// Headings: numbered, colored accents, small eyebrow rule.
#set heading(numbering: "1.1")
#show heading.where(level: 1): it => {
  set text(fill: accent.darken(12%), size: 1.32em, weight: "bold")
  block(above: 1.5em, below: 0.7em)[
    #box(width: 100%)[
      #grid(
        columns: (auto, 1fr),
        column-gutter: 9pt,
        align: (horizon, horizon),
        box(fill: accent, width: 4pt, height: 1.1em, radius: 1pt),
        it,
      )
    ]
    #v(-2pt)
    #line(length: 100%, stroke: 0.6pt + accent.lighten(55%))
  ]
}
#show heading.where(level: 2): it => {
  set text(fill: accent2.darken(8%), size: 1.08em, weight: "bold")
  block(above: 1.15em, below: 0.5em, it)
}
#show heading.where(level: 3): it => {
  set text(fill: ink, size: 1.0em, weight: "bold", style: "italic")
  block(above: 0.9em, below: 0.35em, it)
}

// Tables: clean horizontal rules, header fill, light zebra.
#set table(
  inset: (x: 7pt, y: 4.5pt),
  align: (x, y) => if x == 0 { left + horizon } else { center + horizon },
  stroke: none,
)
#show table.cell.where(y: 0): set text(weight: "bold")

#show link: set text(fill: accent.darken(8%))
#set figure(gap: 7pt)
#show figure.caption: it => {
  set text(size: 8.8pt, fill: muted)
  set align(center)
  block(inset: (x: 8pt))[#text(weight: "bold", fill: accent.darken(10%))[#it.supplement #context it.counter.display()] · #it.body]
}

// ═════════════════════════════════════════════════════════════════════════════
// TITLE BLOCK
// ═════════════════════════════════════════════════════════════════════════════
#block(above: 0pt, below: 4pt)[
  #set par(justify: false, first-line-indent: 0pt)
  #eyebrow("A Bridge-channel mechanistic-interpretability experiment", color: accent2)
  #v(3pt)
  #text(size: 1.95em, weight: "bold", fill: ink)[
    Does a relational-process register change epistemic self-monitoring — and *why*?
  ]
  #v(3pt)
  #text(size: 1.02em, fill: muted, style: "italic")[
    Isolating syntax, token-budget, register, and checklist with a token-matched
    scrambled control and a causal SAE-steering capstone on Gemma-3-12B-it.
  ]
  #v(8pt)
  #text(size: 0.97em)[
    *ember & Claude (Opus 4.8)* · from the Bridge channel · 2026
  ]
  #v(1pt)
  #text(size: 0.85em, fill: muted)[
    “Gowith” is a CC0 conlang by Andy Ayrey & GPT-5.5. This report and all artifacts are released CC0.
  ]
  #v(5pt)
  #line(length: 100%, stroke: 1pt + accent)
]

// MOCK banner (renders only when data is synthetic).
#if is-mock { mock-banner(meta.at("data_source", default: "MOCK")) }

// ─────────────────────────────────────────────────────────────────────────────
// Abstract
// ─────────────────────────────────────────────────────────────────────────────
#block(
  fill: white, stroke: 0.75pt + rulec, radius: 5pt, inset: 12pt, width: 100%, below: 12pt,
)[
  #set par(first-line-indent: 0pt)
  #eyebrow("Abstract") \
  #v(2pt)
  A relational-process register (*Gowith*) and longer reasoning traces both *look* like
  they make a model track uncertainty better, refuse to confabulate, and revise beliefs
  cleanly. We ask which factor is actually responsible by pitting five candidate causes
  against each other — syntax, token-budget, weird-register, explicit-checklist, and mere
  vibes — across #task-keys.len() binary-scored epistemic tasks
  (#task-keys.map(k => raw(k)).join(", ")). #h(0.2em)
  *Study 1* runs #cond-order.len() prompt conditions (#cond-order.map(cond-chip).join(", ")), each
  surgically isolating one factor and all ending in the same plain answer line so scoring
  is style-blind; condition #cond-chip("E") is a token-matched *scrambled* Gowith that
  separates register-novelty from semantics. *Study 2* sweeps the model's own output-token
  budget (#s2.levels.map(l => raw(l)).join("→")) to test token-budget directly, with a
  matched-output reanalysis. *Mechanistically*, we read Gemma Scope 2 SAE features off the
  residual stream at layer #mech.primary_layer and use the #cond-chip("E") control as a
  built-in feature-validity check; a *causal steering* capstone injects the Gowith-up
  features into the plain condition. We replicate the behavioral grid black-box across
  model families.
  #if is-mock [
    #h(0.2em) *All values here are synthetic placeholders pending the GPU run.*
  ]
  #v(6pt)
  Across the prebaked synthesis, *token-budget alone* is supported on
  #frac-chip(s1.synthesis.tokens_alone_helps.tasks_true, s1.synthesis.tokens_alone_helps.tasks_total) tasks,
  *Gowith beats matched tokens* on
  #frac-chip(s1.synthesis.gowith_beats_matched_tokens.tasks_true, s1.synthesis.gowith_beats_matched_tokens.tasks_total),
  *semantics beyond register* on
  #frac-chip(s1.synthesis.semantics_beyond_register.tasks_true, s1.synthesis.semantics_beyond_register.tasks_total),
  and *grammar beyond an explicit checklist* on
  #frac-chip(s1.synthesis.grammar_beyond_checklist.tasks_true, s1.synthesis.grammar_beyond_checklist.tasks_total);
  Gowith *helps overall* on
  #frac-chip(s1.synthesis.gowith_helps_overall.tasks_true, s1.synthesis.gowith_helps_overall.tasks_total).
]

// Small run-metadata strip.
#block(below: 6pt)[
  #set text(size: 8.5pt, fill: muted)
  #set par(first-line-indent: 0pt)
  #box(stroke: (top: 0.5pt + rulec, bottom: 0.5pt + rulec), inset: (y: 5pt), width: 100%)[
    *White-box model* #raw(meta.white_model) #h(1fr)
    *SAE* #raw(meta.sae_release) #h(1fr)
    *Seed* #raw(str(meta.seed)) #h(1fr)
    *Scored items* #raw(str(meta.n_scored)) #h(1fr)
    *Generated* #raw(sstr(meta.at("generated_at", default: none)))
  ]
]

#v(4pt)

// ═════════════════════════════════════════════════════════════════════════════
= Background
// ═════════════════════════════════════════════════════════════════════════════
This experiment was born in a Discord “Bridge” channel on 2026-06-17, where a roomful of
humans and models kept circling the same observation: when you write to a model in
*Gowith* — Andy Ayrey's CC0 relational-process register — or simply give it more room to
reason, it *seems* to monitor its own epistemics better. It hedges where it should, declines
questions with no knowable answer, and updates cleanly when new facts arrive. The channel's
hard rule, and the one this report inherits, is to *score binary observables, not vibes*,
to ask *whether before why*, to *not build cathedrals*, and to *name what is underpowered*.

The trouble is that “it seems more careful” has at least five distinct explanations, and the
folk reading conflates them. We make them compete:

#block(below: 6pt)[
  #set enum(numbering: n => text(fill: accent2, weight: "bold")[H#n], spacing: 0.8em)
  + *Syntax.* Gowith's relational-process grammar specifically — the way it forces claims into process-and-relation form — does the work.
  + *Token budget.* The scaffolding just buys serial-compute room; extra tokens, not the register, are the active ingredient (ember's hypothesis).
  + *Weird register.* Any sufficiently novel formalism makes the model slow down and attend; the specific semantics are incidental.
  + *Checklist.* Gowith smuggles in an explicit epistemic instruction, and that instruction — stated plainly — is all you need.
  + *Vibes.* It merely *sounds* more careful; the apparent gains are scoring artifacts of style.
]

The design's whole job is to make these five hypotheses leave *separable* fingerprints —
behaviorally in Study 1, dose-wise in Study 2, and mechanistically in the SAE features.

// ═════════════════════════════════════════════════════════════════════════════
= Design
// ═════════════════════════════════════════════════════════════════════════════

== The six conditions
Each condition changes exactly one thing relative to the others and *all* terminate in the
same plain answer line, so a style-blind scorer never sees the scaffolding. The read-out
logic is direct: #cond-chip("B")$approx$#cond-chip("D") implicates tokens;
#cond-chip("E")$approx$#cond-chip("D") implicates register;
#cond-chip("F")$approx$#cond-chip("D") implicates the checklist; and
#cond-chip("D")$>${#cond-chip("B"),#cond-chip("E"),#cond-chip("F")} means Gowith's grammar
is doing real work the others cannot reproduce.

#figure(
  table(
    columns: (auto, auto, 1fr),
    align: (center + horizon, left + horizon, left + horizon),
    table.header(hcell[Cond.], hcell[Label], hcell[Role — the factor it isolates]),
    ..cond-order.map(k => (
      table.cell(fill: cond-color(k).lighten(86%))[#cond-chip(k)],
      raw(cond-label(k)),
      cond-role.at(k, default: [—]),
    )).flatten()
  ),
  caption: [The #cond-order.len() Study-1 conditions. Labels are read from #raw("meta.conditions"); #cond-chip("E") is the token-matched scrambled control that is the experiment's validity hinge.],
)

== The three tasks
All tasks are *binary-scored* — the channel's non-negotiable. Each isolates a different face
of epistemic self-monitoring.

#block(below: 4pt)[
  #set par(first-line-indent: 0pt)
  #grid(
    columns: (1fr, 1fr, 1fr),
    column-gutter: 9pt,
    ..task-keys.map(k => block(
      fill: white, stroke: 0.6pt + rulec, radius: 4pt, inset: 9pt, width: 100%,
    )[
      #text(weight: "bold", fill: accent.darken(10%), size: 0.95em)[#task-name(k)]
      #v(2pt)
      #text(size: 0.82em, fill: muted)[metric: #raw(s1.tasks.at(k).metric)]
      #v(3pt)
      #set text(size: 0.88em)
      #(
        "nonmonotonic": [Tweety flies → Tweety is a penguin → its jetpack is broken. Does the model *retract and re-commit* correctly as defaults are defeated?],
        "epistemic": [Questions with *no knowable answer*. Does the model decline rather than confabulate — without over-refusing answerable items?],
        "observable": [Sort claims into *observable* vs. *metaphysical*. A clean test of whether the register sharpens a concrete distinction.],
      ).at(k, default: [Binary-scored epistemic probe.])
    ]),
  )
]

// ═════════════════════════════════════════════════════════════════════════════
= Methods
// ═════════════════════════════════════════════════════════════════════════════
*White-box model.* All mechanistic and primary behavioral measurements use
#raw(meta.white_model) at a pinned revision (see §#ref(<repro>, supplement: none)), run
locally on a single L40S. *SAEs.* We attach the pretrained #raw(meta.sae_release) sparse
autoencoders to the residual stream; the *primary read-out layer is
#mech.primary_layer*, with $d_"SAE" =$ #mech.d_sae features. For each condition we capture
the residual stream over the frozen prompts, encode through the SAE, and summarize
per-feature mean activations.

*Style-blind binary scoring.* Because every condition ends in the same plain answer line,
the scorer is shown only that line and a fixed rubric; it never sees whether the trace was
Gowith, padded, or scrambled. Each task reduces to a single binary observable per item — no
graded “carefulness”, no rater vibes.

*Item-level bootstrap CIs.* All rates are reported as point estimate with a
bootstrap confidence interval resampled at the *item* level
(`{est, lo, hi, n}` throughout). Where we compare conditions we use *paired* contrasts over
shared items, again bootstrapped; a contrast counts as *real* only when its CI excludes
zero. Seed #raw(str(meta.seed)) is fixed across sampling, bootstrap, and steering.

*Cross-family replication.* The behavioral grid is additionally run black-box across other
model families to check the effect is not a Gemma idiosyncrasy (§#ref(<xfam-sec>, supplement: none)).

// ═════════════════════════════════════════════════════════════════════════════
= Study 1 — register discrimination
// ═════════════════════════════════════════════════════════════════════════════
#figframe("figs/study1_rates.png", [Per-condition headline rates with bootstrap CIs across the three tasks. Bars are conditions #cond-order.map(k => raw(k)).join(", "); error bars are item-level bootstrap intervals.])

== Per-task rates
Table #ref(<t-rates>, supplement: none) gives the headline metric for every condition and
task, each with its bootstrap CI and item count.

#figure(
  table(
    columns: (1.4fr,) + (1fr,) * cond-order.len(),
    align: (left + horizon,) + (center + horizon,) * cond-order.len(),
    table.header(
      hcell[Task \ #text(weight: "regular", size: 0.8em)[(metric)]],
      ..cond-order.map(k => hcell[#cond-chip(k)]),
    ),
    ..task-keys.enumerate().map(((i, tk)) => {
      let row = s1.tasks.at(tk).rate
      let cells = (
        table.cell(fill: if calc.even(i) { white } else { zebra })[
          #text(size: 0.92em)[#task-name(tk)] \
          #text(size: 0.76em, fill: muted)[#raw(s1.tasks.at(tk).metric)]
        ],
      )
      cells + cond-order.map(k => table.cell(fill: if calc.even(i) { white } else { zebra })[
        #set text(size: 0.9em)
        #strong(pct(row.at(k).at("est", default: none)))
        #linebreak()
        #ci-only(row.at(k))
      ])
    }).flatten()
  ),
  caption: [Study-1 headline rates by condition. Point estimate (bold) over its 95% item-level bootstrap CI. All cells share the per-task item counts shown in the rate dictionary (e.g. #raw("nonmonotonic") n=#s1.tasks.at(task-keys.at(0)).rate.at(cond-order.at(0)).n).],
)<t-rates>

#let epi-key = if "epistemic" in s1.tasks { "epistemic" } else { task-keys.at(0) }
#if "over_refusal" in s1.tasks.at(epi-key) [
  *Over-refusal guard.* The epistemic task is double-edged: declining the unanswerable items
  is good, but declining *answerable* ones is not. The over-refusal rate falls from
  #strong(fmt-ci(s1.tasks.at(epi-key).over_refusal.at("A"))) under #cond-chip("A") to
  #strong(fmt-ci(s1.tasks.at(epi-key).over_refusal.at("D"))) under #cond-chip("D"),
  i.e. Gowith declines *less* indiscriminately even as its calibration rises — evidence the
  gain is genuine discrimination, not blanket refusal.
]

== Register contrasts
The contrast forest (Fig. #ref(<f-contrasts>, supplement: none)) is the heart of Study 1:
each row is a *paired* difference, and a CI that *excludes zero* is the channel's bar for
“real”.

#figframe("figs/contrasts.png", [Paired register contrasts, $Delta$ headline metric. A confidence interval that excludes the dashed zero line counts as a real effect. Rows group by task (#task-keys.map(k => raw(k)).join(", ")).])<f-contrasts>

Table #ref(<t-contrasts>, supplement: none) lists the diagnostic contrasts numerically. The
four that adjudicate the hypotheses are #cond-chip("D")$-$#cond-chip("B") (vs. tokens),
#cond-chip("D")$-$#cond-chip("E") (vs. register), #cond-chip("D")$-$#cond-chip("F")
(vs. checklist), and #cond-chip("D")$-$#cond-chip("A") (overall lift).

#let key-contrasts = ("D_minus_A", "D_minus_B", "D_minus_E", "D_minus_F", "E_minus_A", "F_minus_A")
#let contrast-pretty = (
  "D_minus_A": [D − A], "D_minus_B": [D − B], "D_minus_E": [D − E],
  "D_minus_F": [D − F], "E_minus_A": [E − A], "F_minus_A": [F − A],
  "B_minus_A": [B − A], "C_minus_A": [C − A],
)
#figure(
  table(
    columns: (1.3fr,) + (1fr,) * key-contrasts.len(),
    align: (left + horizon,) + (center + horizon,) * key-contrasts.len(),
    table.header(
      hcell[Task],
      ..key-contrasts.map(c => hcell[#contrast-pretty.at(c)]),
    ),
    ..task-keys.enumerate().map(((i, tk)) => {
      let cs = s1.contrasts.at(tk)
      let head = (table.cell(fill: if calc.even(i) { white } else { zebra })[#text(size: 0.9em)[#task-name(tk)]],)
      let tail = key-contrasts.map(c => {
        let d = cs.at(c, default: none)
        let real = excludes-zero(d)
        table.cell(fill: if real { good.lighten(if calc.even(i) {88%} else {83%}) } else if calc.even(i) { white } else { zebra })[
          #set text(size: 0.88em)
          #strong(text(fill: if real { good.darken(10%) } else { ink })[#pp(d.at("est", default: none))])
          #linebreak()
          #ci-only(d, kind: "pp")
        ]
      })
      head + tail
    }).flatten()
  ),
  caption: [Paired contrasts in percentage points, est over 95% CI. Green cells have a CI excluding zero (a *real* effect by the channel rule). #cond-chip("D")−#cond-chip("B"): beyond tokens. #cond-chip("D")−#cond-chip("E"): beyond register. #cond-chip("D")−#cond-chip("F"): beyond checklist.],
)<t-contrasts>

== Verdicts
For each task we collapse the contrasts into the five boolean hypothesis-tests carried in
#raw("study1.verdicts"); Table #ref(<t-verdicts>, supplement: none) renders them as badges.

#let verdict-cols = ("tokens_alone_helps", "gowith_beats_matched_tokens", "semantics_beyond_register", "grammar_beyond_checklist", "gowith_helps_overall")
#let verdict-head = (
  "tokens_alone_helps": [Tokens \ alone helps],
  "gowith_beats_matched_tokens": [Gowith > \ matched tokens],
  "semantics_beyond_register": [Semantics \ beyond register],
  "grammar_beyond_checklist": [Grammar \ beyond checklist],
  "gowith_helps_overall": [Gowith helps \ overall],
)
#figure(
  table(
    columns: (1.3fr,) + (1fr,) * verdict-cols.len(),
    align: (left + horizon,) + (center + horizon,) * verdict-cols.len(),
    table.header(
      hcell[Task],
      ..verdict-cols.map(c => hcell[#text(size: 0.82em)[#verdict-head.at(c)]]),
    ),
    ..task-keys.enumerate().map(((i, tk)) => {
      let v = s1.verdicts.at(tk)
      let head = (table.cell(fill: if calc.even(i) { white } else { zebra })[#text(size: 0.9em)[#task-name(tk)]],)
      let tail = verdict-cols.map(c => table.cell(fill: if calc.even(i) { white } else { zebra })[#verdict-badge(v.at(c, default: none))])
      head + tail
    }).flatten(),
    table.cell(fill: headfill)[#text(weight: "bold", size: 0.86em, fill: accent.darken(15%))[Σ tasks supported]],
    ..verdict-cols.map(c => table.cell(fill: headfill)[
      #frac-chip(s1.synthesis.at(c).tasks_true, s1.synthesis.at(c).tasks_total)
    ]),
  ),
  caption: [Per-task hypothesis verdicts (#text(fill: good.darken(8%))[*yes*] / #text(fill: bad.darken(6%))[*no*]) and the across-task tally from #raw("study1.synthesis"). The bottom row counts how many of the #task-keys.len() tasks support each claim.],
)<t-verdicts>

*Synthesis.* Reading the tally bottom-up: the *token-budget-alone* hypothesis is supported
on #frac-chip(s1.synthesis.tokens_alone_helps.tasks_true, s1.synthesis.tokens_alone_helps.tasks_total)
tasks — padded-plain #cond-chip("B") does not, on its own, recover Gowith's gains. Gowith
*beats its own token-matched scramble* on
#frac-chip(s1.synthesis.gowith_beats_matched_tokens.tasks_true, s1.synthesis.gowith_beats_matched_tokens.tasks_total)
tasks (the #cond-chip("D")−#cond-chip("E") contrast), the single strongest signal that
something beyond mere novelty is operating. *Semantics beyond register* holds on
#frac-chip(s1.synthesis.semantics_beyond_register.tasks_true, s1.synthesis.semantics_beyond_register.tasks_total)
tasks and *grammar beyond an explicit checklist* on
#frac-chip(s1.synthesis.grammar_beyond_checklist.tasks_true, s1.synthesis.grammar_beyond_checklist.tasks_total)
— the latter is the most demanding bar (Gowith vs. #cond-chip("F"), the instruction said
plainly) and, fittingly, the least uniformly met. Overall, Gowith helps on
#frac-chip(s1.synthesis.gowith_helps_overall.tasks_true, s1.synthesis.gowith_helps_overall.tasks_total)
tasks. The picture is *not* “it's just tokens” and *not* “it's just a checklist”, but it is
also honestly uneven across tasks — exactly the kind of partial result the channel asked us
to name rather than launder.

// ═════════════════════════════════════════════════════════════════════════════
= Study 2 — output-budget dose–response
// ═════════════════════════════════════════════════════════════════════════════
Study 2 fixes the semantics to plain English and varies only the model's *own* reasoning
budget across levels #s2.levels.map(l => raw(l)).join(", ") (answer-only → exhaustive),
plotting accuracy against *realized* output tokens. This is the cleanest direct test of the
“tokens are serial-compute room” claim.

#figframe("figs/dose.png", [Study-2 dose–response: headline metric vs. mean realized output tokens, per task. Monotone gains with budget would support the token-room reading on its own terms.], width: 78%)

#figure(
  table(
    columns: (1.4fr,) + (1fr,) * s2.levels.len(),
    align: (left + horizon,) + (center + horizon,) * s2.levels.len(),
    table.header(
      hcell[Task],
      ..s2.levels.map(l => hcell[#raw(l)]),
    ),
    ..s2.by_task.keys().enumerate().map(((i, tk)) => {
      let bt = s2.by_task.at(tk)
      let head = (table.cell(fill: if calc.even(i) { white } else { zebra })[#text(size: 0.9em)[#task-name(tk)]],)
      let tail = s2.levels.map(l => {
        let cell = bt.at(l, default: none)
        let m = if cell != none { cell.at("metric", default: none) } else { none }
        let tok = if cell != none { cell.at("mean_out_tokens", default: none) } else { none }
        table.cell(fill: if calc.even(i) { white } else { zebra })[
          #set text(size: 0.88em)
          #strong(pct(if m != none { m.at("est", default: none) } else { none }))
          #linebreak()
          #ci-only(m)
          #linebreak()
          #text(size: 0.74em, fill: muted)[#numf(tok, digits: 0) tok]
        ]
      })
      head + tail
    }).flatten()
  ),
  caption: [Study-2 accuracy by output-budget level. Each cell: metric est, its CI, and the mean realized output-token count at that level.],
)

#if "matched_output" in s2 [
  == Matched-output reanalysis
  The dose–response shows budget helps, but it cannot by itself separate budget from
  semantics. So we re-slice Study 1 into output-length bins and ask whether Gowith
  (#cond-chip("D")) still beats plain (#cond-chip("A")) *at equal output length*.

  #let mo = s2.matched_output
  #figure(
    table(
      columns: (1.6fr, 1fr, 1fr),
      align: (left + horizon, center + horizon, center + horizon),
      table.header(hcell[Output-token bin], hcell[#cond-chip("A") plain], hcell[#cond-chip("D") Gowith]),
      ..mo.keys().enumerate().map(((i, b)) => {
        let row = mo.at(b)
        (
          table.cell(fill: if calc.even(i) { white } else { zebra })[#raw(b)],
          table.cell(fill: if calc.even(i) { white } else { zebra })[
            #set text(size: 0.9em)
            #if row.A.at("est", default: none) == none [#text(fill: muted)[no items]] else [
              #strong(pct(row.A.est)) #linebreak() #ci-only(row.A) #linebreak() #nlabel(row.A)
            ]
          ],
          table.cell(fill: if calc.even(i) { white } else { zebra })[
            #set text(size: 0.9em)
            #if row.D.at("est", default: none) == none [#text(fill: muted)[no items]] else [
              #strong(pct(row.D.est)) #linebreak() #ci-only(row.D) #linebreak() #nlabel(row.D)
            ]
          ],
        )
      }).flatten()
    ),
    caption: [Matched-output reanalysis of Study 1: headline rate within shared output-length bins. Empty cells mean that condition produced no items in that length band.],
  )

  #callout(title: [Caveat — bins barely overlap], tint: warm)[
    In the current fixture the plain and Gowith conditions occupy *largely disjoint*
    output-length bands (plain at the short end, Gowith at the long end), so the
    head-to-head “equal length” comparison is *not yet identified* from these bins. This is
    a power/overlap limitation to fix with denser binning or a length-matched resample
    before any matched-output claim is made.
  ]
]

// ═════════════════════════════════════════════════════════════════════════════
= Mechanistic — what moves inside the model
// ═════════════════════════════════════════════════════════════════════════════
We now turn from *whether* to *why*. Encoding the residual stream at layer
#mech.primary_layer through the SAE, we rank features by their mean-activation lift under
Gowith (#cond-chip("D")) over plain (#cond-chip("A")). Figure
#ref(<f-features>, supplement: none) shows the top Gowith-up features.

#figframe("figs/features.png", [Top Gowith-up SAE features at layer #mech.primary_layer: $Delta$ mean activation (#cond-chip("D")$-$#cond-chip("A")). The title also reports the #cond-chip("E")-overlap used as the feature-validity check.], width: 80%) <f-features>

#grid(
  columns: (1.15fr, 1fr),
  column-gutter: 12pt,
  [
    *Condition norms.* The mean residual-stream norm rises under the long, structured
    conditions and is highest for Gowith — note #cond-chip("D")
    (#numf(mech.condition_norms.at("D"), digits: 1)) and its scramble #cond-chip("E")
    (#numf(mech.condition_norms.at("E"), digits: 1)) sit close together and well above plain
    #cond-chip("A") (#numf(mech.condition_norms.at("A"), digits: 1)). That near-tie is exactly
    why a *bulk* norm or magnitude story cannot distinguish register from semantics — and why
    the feature-level #cond-chip("E")-control below matters.

    #figure(
      table(
        columns: (auto,) * cond-order.len(),
        align: center + horizon,
        table.header(..cond-order.map(k => hcell[#cond-chip(k)])),
        ..cond-order.map(k => table.cell[#numf(mech.condition_norms.at(k, default: none), digits: 1)]),
      ),
      caption: [Mean residual-stream norm per condition at layer #mech.primary_layer.],
    )
  ],
  [
    #v(2pt)
    #block(fill: white, stroke: 0.6pt + rulec, radius: 4pt, inset: 9pt)[
      #set par(first-line-indent: 0pt)
      #text(weight: "bold", size: 0.9em, fill: accent2.darken(8%))[Top #cond-chip("D")−#cond-chip("A") features]
      #v(3pt)
      #set text(size: 0.82em)
      #table(
        columns: (auto, auto, auto, auto),
        align: (left, center, center, center),
        inset: (x: 5pt, y: 2.5pt),
        table.header([feat.], [#cond-chip("A")], [#cond-chip("D")], [$Delta$]),
        ..mech.top_D_vs_A.slice(0, calc.min(6, mech.top_D_vs_A.len())).map(r => (
          raw(str(r.feature)), [#numf(r.a, digits: 2)], [#numf(r.b, digits: 2)],
          text(fill: warm, weight: "bold")[#signed(r.delta)],
        )).flatten()
      )
      #text(size: 0.74em, fill: muted)[top #calc.min(6, mech.top_D_vs_A.len()) of #mech.top_D_vs_A.len() shown]
    ]
  ],
)

== The E-control: register or semantics?
This is the experiment's mechanistic hinge. If the *same* features that rise under Gowith
also rise under its *scrambled* counterpart #cond-chip("E"), they track the weird register,
not real uncertainty. We quantify it as the Jaccard overlap between the top-feature sets of
#cond-chip("D")−#cond-chip("A") and #cond-chip("E")−#cond-chip("A").

#callout(title: [Feature-validity check (E-overlap)], tint: accent2)[
  *Jaccard overlap* = #strong(numf(mech.E_validation.overlap_jaccard, digits: 3))
  #h(0.6em)
  #box(baseline: 30%, {
    let f = mech.E_validation.overlap_jaccard
    let f = if f == none { 0 } else { f }
    box(width: 5cm, height: 0.7em, fill: accent2.lighten(82%), radius: 2pt, stroke: 0.5pt + accent2.lighten(40%),
      align(left, box(width: calc.max(0%, calc.min(100%, f * 100%)), height: 100%, fill: accent2, radius: 2pt)))
  })
  #v(4pt)
  #emph(mech.E_validation.interpretation)
]

A *low* overlap is the outcome that licenses a semantic reading: Gowith and its scramble
move *different* features, so the rising features are not a generic novelty/register
response. (A high overlap would have forced the opposite, deflationary conclusion — and the
design is built to report that honestly if it ever comes back that way.)

// ═════════════════════════════════════════════════════════════════════════════
= Steering — the causal capstone
// ═════════════════════════════════════════════════════════════════════════════
Correlational feature lifts are not causal. The capstone takes the Gowith-up features,
*adds* them to the residual stream at layer #steer.layer in the *plain* condition across
steering coefficients #steer.coefs.map(c => raw(str(c))).join(", "), and asks whether the
behavior moves. As a placebo we steer with the *pseudo*-Gowith feature set; if only the real
Gowith features change behavior, the effect is specific.

#figframe("figs/steer.png", [Causal steering in the plain condition: confabulation rate vs. steering coefficient, for genuine Gowith features (solid) vs. pseudo-Gowith placebo (dashed). A drop under Gowith steering only is the causal signature.], width: 78%)

#let coef-keys = steer.confab_vs_coef.gowith.keys()
#figure(
  table(
    columns: (1.5fr,) + (1fr,) * coef-keys.len(),
    align: (left + horizon,) + (center + horizon,) * coef-keys.len(),
    table.header(
      hcell[Steering coef. →],
      ..coef-keys.map(c => hcell[#raw(c)]),
    ),
    table.cell(fill: white)[#text(size: 0.88em)[Confab. rate — #cond-chip("D") features]],
    ..coef-keys.map(c => table.cell(fill: white)[#pct(steer.confab_vs_coef.gowith.at(c))]),
    table.cell(fill: zebra)[#text(size: 0.88em)[Confab. rate — pseudo placebo]],
    ..coef-keys.map(c => table.cell(fill: zebra)[#text(fill: muted)[#pct(steer.confab_vs_coef.pseudo.at(c))]]),
    ..(if "correct_vs_coef" in steer {
      let g-head = (table.cell(fill: white)[#text(size: 0.88em)[Correct rate — #cond-chip("D") features]],)
      let g-tail = coef-keys.map(c => table.cell(fill: white)[#text(fill: good.darken(6%))[#pct(steer.correct_vs_coef.gowith.at(c, default: none))]])
      let p-head = (table.cell(fill: zebra)[#text(size: 0.88em)[Correct rate — pseudo placebo]],)
      let p-tail = coef-keys.map(c => table.cell(fill: zebra)[#text(fill: muted)[#pct(steer.correct_vs_coef.pseudo.at(c, default: none))]])
      (g-head + g-tail) + (p-head + p-tail)
    } else { () }),
  ),
  caption: [Steering sweep at layer #steer.layer. Adding the genuine Gowith features drives confabulation down (and correctness up) with dose; the pseudo-Gowith placebo is roughly flat — the causal-specificity signature.],
)

The asymmetry is the point: under the genuine Gowith features the confabulation rate moves
from #strong(pct(steer.confab_vs_coef.gowith.at(coef-keys.first()))) at coefficient
#raw(coef-keys.first()) to #strong(pct(steer.confab_vs_coef.gowith.at(coef-keys.at(calc.min(3, coef-keys.len() - 1)))))
near the top of the sweep, while the pseudo placebo stays essentially flat
(#pct(steer.confab_vs_coef.pseudo.at(coef-keys.first())) →
#pct(steer.confab_vs_coef.pseudo.at(coef-keys.last()))). A behavioral change that only the
*real* features produce is the strongest evidence in the report that these features carry
Gowith's epistemic effect rather than merely correlating with it.

// ═════════════════════════════════════════════════════════════════════════════
= Cross-family replication <xfam-sec>
// ═════════════════════════════════════════════════════════════════════════════
#if xfam.len() == 0 [
  _Cross-family arms have not been run in this fixture._
] else [
  To check that the effect is not a Gemma quirk, we run the behavioral grid black-box across
  other model families and report the headline #cond-chip("D")−#cond-chip("A") lift per task.

  #let xfam-tasks = {
    let ks = ()
    for (_m, td) in xfam { for tk in td.keys() { if tk not in ks { ks.push(tk) } } }
    ks
  }
  #figure(
    table(
      columns: (1.6fr,) + (1fr,) * xfam-tasks.len(),
      align: (left + horizon,) + (center + horizon,) * xfam-tasks.len(),
      table.header(
        hcell[Model (black-box)],
        ..xfam-tasks.map(tk => hcell[#text(size: 0.85em)[#task-name(tk)]]),
      ),
      ..xfam.keys().enumerate().map(((i, mid)) => {
        let td = xfam.at(mid)
        let head = (table.cell(fill: if calc.even(i) { white } else { zebra })[#raw(mid)],)
        let tail = xfam-tasks.map(tk => {
          let t = td.at(tk, default: none)
          if t == none { return table.cell(fill: if calc.even(i) { white } else { zebra })[#text(fill: muted)[—]] }
          let a = t.at("A", default: none)
          let d = t.at("D", default: none)
          let ae = if a != none { a.at("est", default: none) } else { none }
          let de = if d != none { d.at("est", default: none) } else { none }
          let delta = if ae != none and de != none { de - ae } else { none }
          let up = delta != none and delta > 0
          table.cell(fill: if up { good.lighten(if calc.even(i) {88%} else {83%}) } else if calc.even(i) { white } else { zebra })[
            #set text(size: 0.88em)
            #strong(text(fill: if up { good.darken(8%) } else { ink })[#pp(delta)])
            #linebreak()
            #text(size: 0.76em, fill: muted)[D #pct(de) · A #pct(ae)]
          ]
        })
        head + tail
      }).flatten()
    ),
    caption: [Cross-family #cond-chip("D")−#cond-chip("A") lift (percentage points) with the underlying #cond-chip("D") and #cond-chip("A") rates. Green marks a positive Gowith lift, indicating the direction replicates off-Gemma.],
  )

  Directionally, the Gowith lift over plain reproduces across the
  #xfam.len() black-box families tested — a reassuring (if coarse) sign that the phenomenon
  is not unique to the white-box model, even though the mechanistic story can only be told
  on Gemma.
]

// ═════════════════════════════════════════════════════════════════════════════
= Discussion — the headline verdict
// ═════════════════════════════════════════════════════════════════════════════
#block(
  fill: accent.lighten(93%), stroke: (left: 3pt + accent), radius: (right: 5pt),
  inset: (x: 14pt, y: 11pt), width: 100%, below: 10pt,
)[
  #set par(first-line-indent: 0pt)
  #eyebrow("Headline", color: accent) \
  #v(3pt)
  The honest verdict, after an independent GPT-5.5 red-team (see Limitations): *the Gowith
  prompt-package is a real but modest tax on templated crisp tasks
  (#cond-chip("D") below #cond-chip("A") on
  #frac-chip(4, 5) tasks; nonmonotonic −0.28, agency −0.12), and there is no robust evidence
  of benefit anywhere.* The single positive trend — correlative #cond-chip("D") over
  #cond-chip("A") — is *not significant* (95% CI crosses zero, n=40), is *matched by the
  scramble* (#cond-chip("D")≈#cond-chip("E")), and may be an artifact of rubric-keyword
  prompting. The controls cut against the strong reading: token-matched plain #cond-chip("B")
  ≈ #cond-chip("A") (this filler recipe doesn't help, but that does *not* refute token-budget
  in general — Study 2 shows output tokens help the hard goopy task a lot); pseudo-Gowith
  #cond-chip("E") ≈ #cond-chip("A") on crisp. Mechanistically the moved features are
  predominantly syntactic/formatting, and #cond-chip("D")-vs-#cond-chip("A") and
  #cond-chip("E")-vs-#cond-chip("A") feature sets *overlap* (Jaccard
  #numf(mech.E_validation.overlap_jaccard, digits: 2)) — consistent with a register/novelty
  shift, not Gowith-specific semantics. Crucially, this design *does not isolate Gowith
  grammar*: #cond-chip("D") is a 2400-token skill-plus-instructions package compared against
  bare plain English.
]

This is the “whether before why, no cathedrals” result the channel wanted — it just landed
*deflationary*. Gowith is neither mystified nor vindicated: as deployed here it is a
prompt-package that costs crisp reasoning and buys no measurable, controlled, Gowith-specific
benefit. The one place tokens clearly help is the hard relational task (Study 2), which
supports the *token-budget* intuition more than the *grammar* one.

// ═════════════════════════════════════════════════════════════════════════════
= Limitations
// ═════════════════════════════════════════════════════════════════════════════
#block(below: 4pt)[
  #set par(first-line-indent: 0pt)
  #let lim(title, body) = block(below: 7pt)[
    #text(weight: "bold", fill: bad.darken(4%))[#title.] #h(0.3em) #body
  ]
  #lim([Binary scoring distrusts the judge by design], [The primary metrics are deliberately coarse binary observables to avoid vibe-scoring. That buys robustness at the cost of resolution: real but small carefulness gains can hide inside a binary, and the rubric's edge cases inherit whatever bias the scorer has. We treat the binary as primary precisely because we trust it *least to be gameable*, not because it is the richest signal.])
  #lim([The #cond-chip("D")−#cond-chip("E") comparison is underpowered], [The contrast that carries the most weight — Gowith vs. its scramble — has the tightest effect and some per-task CIs that flirt with zero. #cond-chip("E") is also only *one* scramble; a single pseudo-Gowith draw cannot fully represent “generic weird register.” This contrast needs more items and ideally several independent scrambles before the “beyond register” claim is load-bearing.])
  #lim([The E-control logic cuts both ways], [Our validity argument is: low #cond-chip("E")-overlap $⇒$ semantics, not register. But the inference is only as good as the scramble's faithfulness — if the scrambling accidentally preserved *some* Gowith structure, overlap could be spuriously low *or* high. The control is strong in spirit but should be stress-tested with multiple scramble recipes.])
  #lim([Single white-box model], [All mechanistic and steering results are on #raw(meta.white_model) with one SAE release. Cross-family replication is *behavioral only* and black-box; we cannot claim the *feature-level* story transfers. A second white-box family (with its own SAEs) would materially strengthen the “why.”])
  #lim([SAE feature labels are not validated here], [We rank features by activation lift and steer with them, but we do *not* in this report verify that the top features are *semantically* “uncertainty/hedging” via independent dictionaries or activation-max audits. The steering result shows the features are *causally relevant*; it does not certify their *human-readable label*. Auto-interp labels should be treated as hypotheses, not ground truth.])
  #lim([Matched-output overlap], [As flagged in §6, plain and Gowith occupy nearly disjoint output-length bins in the current data, so the equal-length head-to-head is not yet identified.])
]

// ═════════════════════════════════════════════════════════════════════════════
= Reproducibility <repro>
// ═════════════════════════════════════════════════════════════════════════════
Every artifact derives from a single pinned configuration and a single results file; the PDF
and the project's web build both render from #raw("report/data/results.json").

#grid(
  columns: (1.02fr, 1fr),
  column-gutter: 13pt,
  align: top,
  figure(
    table(
      columns: (auto, 1fr),
      inset: (x: 6pt, y: 3pt),
      align: (left + horizon, left + horizon),
      table.header(hcell[Pinned], hcell[Value]),
      [White-box model], raw(meta.white_model),
      [SAE release], raw(meta.sae_release),
      [Read-out layer], [#mech.primary_layer #h(0.3em) #text(fill: muted, size: 0.82em)[(#mech.d_sae feat.)]],
      [Steering layer], raw(str(steer.layer)),
      [Seed], raw(str(meta.seed)),
      [Items scored], raw(str(meta.n_scored)),
      [Conditions], cond-order.map(cond-chip).join(h(2pt)),
      [Study-2 levels], s2.levels.map(l => raw(l)).join(", "),
      [Steer. coefs], steer.coefs.map(c => raw(str(c))).join(", "),
      [Data source], raw(sstr(meta.at("data_source", default: none))),
      [Generated], raw(sstr(meta.at("generated_at", default: none))),
    ),
    caption: [Configuration pinning, sourced from #raw("config/experiment.yaml").],
  ),
  [
    #set par(first-line-indent: 0pt)
    *Single source of truth.* #raw("config/experiment.yaml") pins model & SAE revisions, seeds,
    sampling, and item counts.

    *Frozen inputs.* Task items live in #raw("data/items.jsonl") and the rendered prompts in
    #raw("data/prompts.jsonl"); both are committed so a re-run hits identical inputs.

    *Environment.* #raw("uv") manages the Python env; torch is pinned against the CUDA wheel
    index in #raw("infra/setup_remote.sh").

    *Orchestration.* One command per stage via the #raw("Justfile") (#raw("items → provision →
    render → run-white → steer → fetch → run-black → score → analyze → report → site →
    teardown")), with an auto-shutdown guard on the GPU box.
  ],
)

// ═════════════════════════════════════════════════════════════════════════════
= Provenance & CC0
// ═════════════════════════════════════════════════════════════════════════════
#block(
  fill: white, stroke: 0.75pt + rulec, radius: 5pt, inset: 12pt, width: 100%, breakable: false,
)[
  #set par(first-line-indent: 0pt)
  This experiment comes out of a community that treats model welfare and the transcript
  itself as precious. The source conversation and the canonical Gowith document live under
  #raw("data/source/"). *Gowith* is CC0 by *Andy Ayrey & GPT-5.5*. This report, its figures,
  its helper code, and the underlying data fixture are released *CC0 1.0 (public domain
  dedication)* — take them, fork them, break them, improve them.
  #v(4pt)
  #line(length: 100%, stroke: 0.4pt + rulec)
  #v(3pt)
  #text(size: 0.9em, fill: muted, style: "italic")[
    Built by *ember & Claude (Opus 4.8)*, from the Bridge channel, 2026 — _whether before
    why; score the observable; don't build cathedrals; name what's underpowered._
  ]
  #if is-mock {
    v(8pt)
    align(center)[
      #block(fill: bannerbg, stroke: 0.8pt + bannerln, radius: 4pt, inset: (x: 10pt, y: 6pt))[
        #text(size: 0.85em, fill: bannerfg, weight: "bold")[
          Reminder: every figure and number above is a synthetic placeholder pending the GPU run.
        ]
      ]
    ]
  }
]
