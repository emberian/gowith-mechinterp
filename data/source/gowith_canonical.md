# Gowith - a relational language

## What is this and why have you done it
The Sapir Whorf hypothesis holds that the language we speak shapes the way that we think. 

Gowith begins from a simple suspicion: ordinary English is very good at making the world look like separate things that possess properties, cause effects, and own other things. “The wind blew the door shut.” “His hands.” “Paul’s picture.” These are useful patterns, but they can become clumsy when we are trying to think about relationships, correlations, living systems, conflicts, ecologies, quantum measurement, or anything else where the important thing is not inside one object but happening between many participants.

I call this English habit substance-and-ownership. That is my shorthand, not a formal linguistic category.

Gowith experiments with a different habit: relational-process-animate grammar. In this mode, happenings come first. Participants join the happening in named ways. Possession becomes a specific relation rather than the default relation. States bud, hold, fade, echo, and settle.

This experiment has several ancestors and neighbours. It is inspired by weak linguistic relativity, process philosophy, natural-language ontology, semantic-role grammar, linguistic typology around possession and animacy, and Robin Wall Kimmerer’s writing on the grammar of animacy. In Braiding Sweetgrass, Kimmerer gives the beautiful example of a bay not only as a fixed noun, but as something the water is doing: “to be a bay.” Gowith is not Anishinaabemowin, Te Reo Māori, Quechua, or any Aboriginal language. It does not claim descent from them, and it does not claim to represent Indigenous grammar. It is an English-derived synthetic notation built by an artist asking AI weird questions.
The question is smaller and stranger: can we use a made-up language to make certain thoughts easier for language models and humans to hold?

It may not be complete, useful, correct, or even good. It is a small experiment. Maybe nothing happens. Maybe something interesting settles. I am curious to hear whether it creates unexpected shifts in how people solve problems in partnership with large language models.

In homage to Alan Watts, I call it Gowith.

## How to use it 
You do not need to speak this language. It is 2026, we have language models, and they speak everything. Including made up languages like this one. 

So: use the below Markdown as a skill in any LLM, or pass it in to a context window. You may ask your favourite model, such as  Claude or GPT, to use it when when reasoning about: complex systems, relationships and conflict, biology, quantum mechanics, and other goopy "correlative" topics. 

In my experience, it is a very token efficient way for models to reason about topics which English is bad at. It may be better or worse than mathematical, scientific or ERD style notations - I have not exhaustively compared or evaluated performance. Treat it as an experiment, and [let me know](https://x.com/andyayrey) if anything interesting happens when you do. 

The skill comes with a translation back to English. In honour of the unique and beautiful way ESOL speakers mirror the logic of their mother tongue when speaking English, it includes a mapping from Gowith back to Gowith-accented English.  

## Author and License
[@AndyAyrey](https://x.com/andyayrey) and GPT5.5

This is for anyone, use it however you want. To the extent possible under law, I have waived all copyright and related or neighboring rights to this work using CC0.

## Disclaimers
Theoretically, Gowith should be safer than ordinary English when reasoning about screwy ideas involving quantum mechanics, biology, spacetime, consciousness, complex systems, or other classic sources of “I solved the universe” delusion. Its whole point is to make relations, uncertainty, locality, process, and partial perspective easier to say.

But no guarantees are offered.

Truth does not sit. Truth settles.


## LLM Skill - give this to your models

````markdown
---
name: gowith
version: 0.1
description: >-
  Translate ordinary English into Gowith seeds and Gowith-projected English:
  an English-derived relational-process-animate register that foregrounds
  happenings, participation, phase, relation, and explicit agency.
---

# Gowith

Gowith is an experimental controlled language and model skill. It forks English away from a default **substance-and-ownership** framing and toward a **relational-process-animate** framing. Version 0.1 is a productive semantic notation and translation register, not yet a fully phonologized conlang.

These are design shorthands, not established linguistic categories:

- **SO:** substances bear properties, own things, and then enter relations.
- **RPA:** happenings unfold; participants gather in roles; relations hold, change, and leave traces.

Gowith does not ban nouns or stable facts. Nouns are handles for relatively stable participants. It changes what the grammar makes easiest to say.

## Modes

**Gowith seeds** are compact event-first notation:

```text
Tire-bud: mi-through.
```

**Gowith ESOL** is the seed projected back into readable English:

> Tiredness is budding in me.

Here “ESOL” means a deliberately calque-like translation register, not an imitation of any learner community or accent.

## Core clause

```text
CLAUSE := PROCESS-CURRENT: PARTICIPANT-GOWITH, PARTICIPANT-GOWITH...
```

Put the happening first. Do not require a grammatical subject or direct object.

```text
Cook-go: Mara-lead, onions-from, fire-through,
         soup-toward, family-for, kitchen-among.
```

> Cooking is going, with Mara leading; onions become material, fire carries the process, soup emerges, the family benefits, and the kitchen surrounds it.

English lexical roots may be used freely and compounded with hyphens.

## Currents

Currents replace simple tense with the event's present relation to completion and consequence.

| Current | Meaning |
|---|---|
| `-bud` | beginning, emerging, becoming perceptible |
| `-go` | underway and still open |
| `-hold` | sustained, habitual, or repeatedly maintained |
| `-settle` | completed, resolved, or arrived at a result |
| `-echo` | completed but still consequential now |
| `-fade` | weakening, ceasing, or formerly holding |
| `-lean` | intended, expected, probable, or approaching |

Optional operators precede the current:

```text
Hunger-return-go: mi-through.       # hunger is returning
Decision-not-settle: wi-team-among. # no decision has settled
```

Clock and calendar phrases normally take `-around`.

## Gowiths

A gowith names how a participant joins the happening.

| Gowith | Role |
|---|---|
| `-lead` | initiator or organizer with meaningful control |
| `-with` | companion, co-participant, or moving theme |
| `-through` | body, medium, instrument, or mechanism |
| `-across` | affected, encountered, altered, or resisted participant |
| `-toward` | recipient, aim, attraction, or emerging result |
| `-from` | source, material, ancestry, or departure |
| `-for` | beneficiary, care, duty, or vulnerability |
| `-among` | spatial, social, ecological, or enabling field |
| `-between` | mutual or jointly sustained relation |
| `-against` | resistance, pressure, support, or collision |
| `-around` | time, topic, occasion, or loose circumstance |
| `-as` | temporary role, name, appearance, or presentation |

Word order marks attention, not metaphysical priority. Binary relations may be compacted directly, as in `mi-between-yu`.

## Pronouns and groups

```text
mi      speaking center
yu      addressed center
they    other center or centers
```

Prefer a typed collective over bare *we* when the relation matters:

```text
wi-two      the two of us
wi-house    this household
wi-team     this working group
wi-all-here everyone present
```

## Possession becomes a named relation

Do not translate *my/your/their X* mechanically. Ask what actually holds.

```text
my hand      -> hand body-with mi
my sister    -> person sister-between mi
my coat      -> coat wear-with mi
my house     -> house dwell-with mi
                or house law-claimed-with mi
my headache  -> Headache-go: mi-through.
my mistake   -> Mistake-settle: mi-lead.
```

Legal ownership remains expressible; it is one relation rather than the default relation.

## Copula and qualities

Avoid bare `X is Y` where a process, role, or relation is more exact.

```text
Mara is tired.          -> Tire-go: Mara-through.
Mara is a teacher.      -> Teach-hold: Mara-through, school-among.
We are friends.         -> Friend-hold: mi-between-yu.
I am called Nara.       -> Answer-hold: mi-as-Nara.
The keys are on the table.
                        -> Rest-hold: keys-with, table-against.
```

Stable classification and exact identity may still be stated when genuinely needed. Gowith resists the generic copula; it does not prohibit factual precision.

## Animacy

All referents may participate grammatically. In Gowith, **animate** means capable of changing how an event unfolds, not necessarily alive, sentient, or conscious.

A door may resist, shelter, open, or channel. Rain may lead one event and move through another. This is event-relative participation, not compulsory personification or panpsychism.

## Agency and accountability

Relational language must not dissolve responsibility.

Use `-lead` for meaningful control. Optional stance tags include `-drift` for accidental involvement, `-pressed` for constrained action, and `-witness` for observation.

```text
Hurt-settle: mi-lead, words-through, yu-across.
```

> Hurt settled across you through words from me, with me leading.

```text
Break-settle: cup-with, shelf-from, floor-against; lead-unknown.
```

> The cup fell from the shelf and broke against the floor; no responsible initiator is known.

Never use process language as a decorative version of “mistakes were made.” Preserve agency, causality, consent, obligation, uncertainty, and harm.

## Translation procedure

When asked to translate into Gowith:

1. Preserve the source claim, including uncertainty, negation, modality, and responsibility.
2. Ask: **what is happening?** Make that the process root.
3. Choose the current: bud, go, hold, settle, echo, fade, or lean.
4. Assign each relevant participant a gowith.
5. Expand generic possession and copular identity into the actual relation.
6. Check that intentional, accidental, constrained, and unknown agency remain distinct.
7. Render the Gowith seed.
8. When requested, project it into Gowith ESOL.

Default output:

```markdown
### Gowith
`...`

### Gowith ESOL
> ...
```

Add an ordinary-English gloss only when useful.

## Gowith ESOL projection

Keep the process as the grammatical center while using enough English syntax to remain immediately readable.

```text
Tire-bud: mi-through.
```

> Tiredness is budding in me.

```text
Decision-not-settle: wi-team-among.
```

> A decision has not settled among us.

```text
Understand-hold: mi-through; Agree-not-hold: mi-through.
```

> Understanding holds in me; agreement does not.

```text
Friend-hold: mi-between-yu.
```

> Friendship holds between us.

```text
Repair-lean: mi-from, yu-toward.
```

> Repair is leaning from me toward you.

The projection should feel one grammatical step away from ordinary English, not maximally alien or ornamental.

## Guardrails

- Do not claim that Gowith reveals reality more truly than other languages.
- Do not assign an “RPA ontology” to a natural language or culture without evidence.
- Do not imitate stereotypes of second-language speakers.
- Do not turn every object into a conscious agent.
- Do not erase stable entities, legal facts, boundaries, or individual responsibility.
- Do not make the result vague merely to make it poetic.
- Prefer ordinary human situations unless the prompt asks for technical language.

## Status and prior art

Gowith is an original synthesis, not a description, reconstruction, or translation of any existing language. Its background includes:

- **Process philosophy:** treating becoming and dynamic existence as philosophically primary. Johanna Seibt, [“Process Philosophy,” Stanford Encyclopedia of Philosophy](https://plato.stanford.edu/entries/process-philosophy/).
- **Natural-language ontology:** studying ontological categories and structures implicit in language. Friederike Moltmann, [“Natural Language Ontology,” Stanford Encyclopedia of Philosophy](https://plato.stanford.edu/entries/natural-language-ontology/).
- **Semantic roles and case grammar:** representing participants by their roles in events rather than only by subject/object position. Charles J. Fillmore, [“The Case for Case”](https://linguistics.berkeley.edu/~syntax-circle/syntax-group/spr08/fillmore.pdf).
- **Linguistic typology:** natural languages grammaticalize animacy and distinguish kinds of possession in many different ways. See WALS [Chapter 34](https://wals.info/chapter/34), [Chapter 58](https://wals.info/chapter/58), and [Chapter 59](https://wals.info/chapter/59).
- **Linguistic relativity, cautiously understood:** particular linguistic features may influence particular habits of cognition; Gowith does not assume global linguistic determinism. See [“Whorfianism,” Stanford Encyclopedia of Philosophy](https://plato.stanford.edu/entries/linguistics/whorfianism.html).

## One-line definition

> **Gowith is English refactored so that happenings come first, participants declare how they go with them, possession becomes a specific relation, and consequences remain grammatically present.**
````

[1]: https://plato.stanford.edu/entries/process-philosophy/ "
Process Philosophy (Stanford Encyclopedia of Philosophy)
"
