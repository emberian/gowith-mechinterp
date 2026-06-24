// lib.typ — helpers & styling for the gowexp report.
// All numbers come from data/results.json; these helpers only format & guard.

// ─────────────────────────────────────────────────────────────────────────────
// Palette — colored section accents, condition tints, semantic states.
// ─────────────────────────────────────────────────────────────────────────────
#let ink      = rgb("#1d2530")     // body text
#let muted    = rgb("#5b6675")     // secondary text
#let accent   = rgb("#2f6f8f")     // primary section accent (slate teal)
#let accent2  = rgb("#9a4f8a")     // secondary accent (register / mechanistic)
#let warm     = rgb("#c0563b")     // Gowith / steering "up"
#let good     = rgb("#2f8f5b")     // true / supported
#let bad      = rgb("#b23b54")     // false / not supported
#let paper    = rgb("#fbfaf7")     // page background tint
#let rulec    = rgb("#dcd7cc")     // hairline rules
#let headfill = rgb("#eef3f5")     // table header fill
#let zebra    = rgb("#f6f4ef")     // table zebra fill
#let bannerbg = rgb("#fff4e0")     // mock-data banner background
#let bannerfg = rgb("#8a4b00")     // mock-data banner foreground
#let bannerln = rgb("#e0a14b")     // mock-data banner border

// Per-condition tints (A–F), echoing the bar-chart palette where sensible.
#let cond-color(k) = (
  "A": rgb("#7a8593"),
  "B": rgb("#d9a01a"),
  "C": rgb("#2bb6c8"),
  "D": warm,
  "E": accent2,
  "F": good,
).at(k, default: muted)

// ─────────────────────────────────────────────────────────────────────────────
// Numeric formatting
// ─────────────────────────────────────────────────────────────────────────────

// Stringify anything, mapping none -> em-dash (str() rejects none).
#let sstr(x) = if x == none { "—" } else { str(x) }

// Round to `digits` decimals, return a string (handles none).
#let numf(x, digits: 3) = {
  if x == none { return "—" }
  if type(x) == str { return x }
  str(calc.round(x, digits: digits))
}

// Percent with one decimal, e.g. 0.7894 -> "78.9%". Guards none.
#let pct(x, digits: 1) = {
  if x == none { return "—" }
  str(calc.round(x * 100, digits: digits)) + "%"
}

// Signed percentage-point delta, e.g. 0.0463 -> "+4.6 pp". Guards none.
#let pp(x, digits: 1) = {
  if x == none { return "—" }
  let v = calc.round(x * 100, digits: digits)
  (if v > 0 { "+" } else if v == 0 { "±" } else { "" }) + str(v) + " pp"
}

// Plain signed number (for activation deltas etc.).
#let signed(x, digits: 2) = {
  if x == none { return "—" }
  let v = calc.round(x, digits: digits)
  (if v > 0 { "+" } else { "" }) + str(v)
}

// "est [lo, hi]" from a {est,lo,hi} dict, as a point-estimate percent with a
// bracketed CI. Guards none dicts and none fields. `as` ∈ "pct" | "pp" | "raw".
#let fmt-ci(d, kind: "pct", digits: 1) = {
  if d == none { return "—" }
  let e = d.at("est", default: none)
  let lo = d.at("lo", default: none)
  let hi = d.at("hi", default: none)
  if e == none { return "—" }
  let f = if kind == "pct" { pct } else if kind == "pp" { pp } else { (v) => numf(v, digits: digits) }
  let body = f(e)
  if lo == none or hi == none {
    body
  } else {
    box[#body #text(size: 0.82em, fill: muted)[[#f(lo), #f(hi)]]]
  }
}

// Just the CI bracket (no point estimate), for compact cells.
#let ci-only(d, kind: "pct") = {
  if d == none { return "—" }
  let lo = d.at("lo", default: none)
  let hi = d.at("hi", default: none)
  if lo == none or hi == none { return "—" }
  let f = if kind == "pct" { pct } else if kind == "pp" { pp } else { numf }
  text(size: 0.85em, fill: muted)[[#f(lo), #f(hi)]]
}

// Whether a contrast's CI excludes zero (→ "real" under the channel's rule).
#let excludes-zero(d) = {
  if d == none { return false }
  let lo = d.at("lo", default: none)
  let hi = d.at("hi", default: none)
  if lo == none or hi == none { return false }
  (lo > 0 and hi > 0) or (lo < 0 and hi < 0)
}

// n-of-items subscript.
#let nlabel(d) = {
  if d == none { return "" }
  let n = d.at("n", default: none)
  if n == none { return "" }
  text(size: 0.78em, fill: muted)[n=#n]
}

// ─────────────────────────────────────────────────────────────────────────────
// Badges & callouts
// ─────────────────────────────────────────────────────────────────────────────

// A yes/no verdict pill. true -> green "yes", false -> red "no", none -> grey "—".
#let verdict-badge(b) = {
  let (fill, fg, label) = if b == true {
    (good.lighten(78%), good.darken(8%), "yes")
  } else if b == false {
    (bad.lighten(80%), bad.darken(6%), "no")
  } else {
    (luma(238), muted, "—")
  }
  box(
    fill: fill,
    inset: (x: 6pt, y: 2pt),
    radius: 3pt,
    stroke: 0.5pt + fg.lighten(40%),
    text(size: 0.82em, weight: "bold", fill: fg, label),
  )
}

// A "k / m tasks" fraction chip, tinted by how strong the support is.
#let frac-chip(k, m) = {
  let frac = if m == 0 { 0 } else { k / m }
  let (fill, fg) = if frac >= 0.999 {
    (good.lighten(80%), good.darken(8%))
  } else if frac <= 0.001 {
    (bad.lighten(82%), bad.darken(6%))
  } else {
    (rgb("#fff1d6"), bannerfg)
  }
  box(
    fill: fill, inset: (x: 7pt, y: 2pt), radius: 3pt,
    stroke: 0.5pt + fg.lighten(45%),
    text(size: 0.88em, weight: "bold", fill: fg)[#k / #m],
  )
}

// Condition code chip (A–F) tinted by condition.
#let cond-chip(k) = {
  let c = cond-color(k)
  box(
    fill: c.lighten(78%), inset: (x: 5pt, y: 1pt), radius: 3pt,
    stroke: 0.5pt + c.lighten(30%),
    text(weight: "bold", fill: c.darken(12%), size: 0.9em, k),
  )
}

// The synthetic-data banner. Renders only when `is_mock` is true.
#let mock-banner(source) = {
  block(
    width: 100%,
    fill: bannerbg,
    stroke: 1.2pt + bannerln,
    radius: 6pt,
    inset: (x: 14pt, y: 11pt),
    above: 6pt, below: 14pt,
  )[
    #set text(fill: bannerfg)
    #grid(
      columns: (auto, 1fr),
      column-gutter: 11pt,
      align: (horizon, left),
      text(size: 1.5em)[⚠],
      [
        #text(weight: "bold", size: 1.02em)[SYNTHETIC PLACEHOLDER DATA — NOT A REAL RUN.]
        #linebreak()
        #text(size: 0.93em)[
          Every number, figure, and verdict below is generated from a mock fixture
          (#raw(source)). The GPU run has not happened yet. These pages exist to
          validate the pipeline and the report layout; do not cite any value as a finding.
        ]
      ],
    )
  ]
}

// A soft callout box for notes / interpretation strings.
#let callout(title: none, body, tint: accent) = {
  block(
    width: 100%,
    fill: tint.lighten(90%),
    stroke: (left: 2.5pt + tint),
    radius: (right: 4pt),
    inset: (x: 12pt, y: 9pt),
    above: 9pt, below: 9pt,
  )[
    #if title != none [
      #text(weight: "bold", fill: tint.darken(12%), size: 0.95em, title)
      #linebreak()
    ]
    #set text(size: 0.95em)
    #body
  ]
}

// Section divider: an accent rule + small-caps eyebrow used in heading show rule.
#let eyebrow(txt, color: accent) = {
  text(fill: color, weight: "bold", size: 0.78em, tracking: 1.2pt)[#upper(txt)]
}

// ─────────────────────────────────────────────────────────────────────────────
// Table scaffolding
// ─────────────────────────────────────────────────────────────────────────────

// Shared cell styling: header cells filled, body cells with subtle padding.
#let hcell(body) = table.cell(fill: headfill)[#text(weight: "bold", size: 0.9em, fill: accent.darken(18%))[#body]]

// A figure with a numbered, styled caption.
#let figframe(path, caption, width: 100%) = figure(
  block(
    stroke: 0.75pt + rulec,
    radius: 5pt,
    inset: 7pt,
    fill: white,
    image(path, width: width),
  ),
  caption: caption,
)
