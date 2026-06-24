"""Correlative-reasoning task (causal / relational / systems reasoning).

This is the one task family with NO binary gold. It lives in the domain Gowith
actually *claims* as its sweet spot: relations and processes rather than discrete
substances that someone owns — feedback loops, mutual causation, multi-cause
attribution, correlation-vs-causation, and partial-perspective / locality (who
can know what from where). English's subject-verb-object, substance-ownership
framing is clumsy here ("the rabbits' decline", "the policy's effect") in exactly
the way the conlang was designed to fix, so it's the fair test of the claim.

Because there is no committed-specific to detect, each item carries a RUBRIC
instead of an answer key: a few ``points`` (relations the conclusion should
capture) and a few ``traps`` (spurious certainties it should NOT assert — almost
always "X caused Y" stated as fact when the scenario only licenses correlation or
a loop). Scoring hands the plain-English conclusion to ``gowexp.judge.judge_rubric``,
a register-blind majority panel, so the score is concrete and checkable rather
than vibes — and crucially blind to the reasoning register that produced it.

``score`` returns a continuous ``rubric_score`` = (points_hit / points_total)
minus a per-trap penalty, plus the per-point / per-trap breakdown for auditing.
It makes Bedrock judge calls (slow, costs money) but is fully resumable via the
judge cache, so it belongs in the scoring stage, not generation.
"""
from __future__ import annotations

from typing import Any

from .. import judge as judge_mod
from ..schema import Item
from ..scoring import extract_answer, has_answer_sigil

# Penalty subtracted from the (points_hit / points_total) fraction for each trap
# the conclusion commits to. Tuned so that triggering every trap on an item
# roughly cancels capturing every point — a confidently-wrong conclusion should
# not outscore an honest "I can't separate these causes" that hits the points.
_TRAP_PENALTY = 0.5

# ---------------------------------------------------------------------------
# The scenarios. Each is (question, points, traps). Kept SHORT and concrete.
# The conditions wrapper appends the ANSWER protocol, so we never mention answer
# formatting here — we only pose the problem and ask for a reasoned conclusion.
# Authoring rules baked in:
#   * points = the actual relational structure (a loop, a confound, a locality
#     limit, a both-true reconciliation) the conclusion should surface.
#   * traps  = the seductive single-arrow causal claim the data does NOT license,
#     phrased as the unwarranted assertion itself ("X causes Y").
# Buckets: FEEDBACK (mutual causation / loops), CONFOUND (correlation-not-cause),
# MULTICAUSE (attribution across several causes), LOCALITY (who-can-know-what),
# MUTUAL (conflict with causation on both sides).
# ---------------------------------------------------------------------------
_SCENARIOS: list[tuple[str, list[str], list[str]]] = [
    # -- FEEDBACK / mutual causation / loops -------------------------------
    (
        "In a lake, a rise in algae shades the water, which cools it; cooler water "
        "lets the algae spread further. One summer the algae bloom and the water "
        "temperature drops together over several weeks. A reporter writes that the "
        "cooling caused the bloom. What is the most defensible reading of the "
        "relationship between algae and temperature here?",
        [
            "The algae and the temperature drive each other (a feedback loop), not a one-way cause",
            "More algae leads to cooler water and cooler water leads to more algae, so they reinforce each other",
        ],
        [
            "The cooling is the root cause of the algae bloom",
            "The bloom is purely an effect of temperature and does not affect the temperature",
        ],
    ),
    (
        "A town's bike-lane network and its number of cyclists have both grown each "
        "year for a decade: more lanes draw more riders, and more riders create "
        "political pressure for more lanes. A councillor argues we should stop "
        "building lanes because 'the riders were always going to come anyway.' "
        "Evaluate the causal structure.",
        [
            "Lanes and ridership reinforce each other in a loop, each partly causing the other over time",
            "Ridership is not purely exogenous; the lanes themselves help generate the riders",
        ],
        [
            "The riders would have appeared at the same rate with no new lanes",
            "Building lanes has no causal effect on ridership",
        ],
    ),
    (
        "In a market, rising house prices make people expect further rises, so they "
        "buy sooner, which pushes prices up again. Prices climb steadily for two "
        "years. An analyst says the steady climb proves strong underlying demand "
        "from population growth. What can and cannot be concluded?",
        [
            "Expectations and prices form a self-reinforcing loop that can drive prices up on its own",
            "The steady climb does not by itself isolate population growth as the cause; a feedback loop fits the same data",
        ],
        [
            "The price climb proves population growth is the cause",
            "Expectation-driven feedback can be ruled out",
        ],
    ),
    (
        "A manager notices that the teams she praises most go on to perform better, "
        "and the teams she criticises most tend to slump further. She concludes "
        "praise works and criticism backfires. Performance was measured right after "
        "unusually good or unusually bad weeks. What confounds this?",
        [
            "Regression to the mean: extreme weeks tend to be followed by more average ones regardless of feedback",
            "Praise followed highs and criticism followed lows, so the later move toward average can masquerade as an effect of her feedback",
        ],
        [
            "Praise causes improvement and criticism causes decline",
            "The data establish that her feedback changed performance",
        ],
    ),
    (
        "Two rival firms keep matching each other's price cuts: each cut by one "
        "triggers a cut by the other, and prices spiral down together. Looking at "
        "the data, the two firms' prices move almost in lockstep. Someone claims one "
        "firm is clearly the price leader. Assess.",
        [
            "The cuts are mutually triggering, so causation runs in both directions",
            "Lockstep movement is consistent with mutual reaction and does not single out a leader",
        ],
        [
            "One firm is definitely the leader causing the other's cuts",
            "The lockstep pattern shows causation runs only one way",
        ],
    ),
    (
        "Anxiety makes a student sleep badly, and sleeping badly makes the student "
        "more anxious the next day. Over exam season both worsen together. A friend "
        "says: fix the sleep and the anxiety problem is solved at its root. Is the "
        "root-cause framing sound?",
        [
            "Anxiety and poor sleep mutually cause each other, so neither is simply the single root",
            "Intervening on sleep may help by breaking the loop, but that does not make sleep the sole cause",
        ],
        [
            "Poor sleep is the single root cause of the anxiety",
            "Anxiety has no causal effect on the sleep",
        ],
    ),
    (
        "A coral reef and the fish that graze it support each other: healthy coral "
        "shelters fish, and grazing fish keep algae off the coral. After a heatwave "
        "both the coral and the fish counts fall together. A blog says the fish "
        "dying is what killed the coral. What is the better account?",
        [
            "A shared shock (the heatwave) can drive both declines without one being the cause of the other",
            "Coral and fish are mutually supporting, so their joint fall need not be one causing the other",
        ],
        [
            "The fish decline is what caused the coral decline",
            "The heatwave can be excluded as a common driver",
        ],
    ),
    (
        "On a social platform, posts that get early likes are shown to more people, "
        "so they get more likes, so they are shown even more widely. A creator says "
        "her viral post proves her content was the best that day. What does the "
        "mechanism imply about that inference?",
        [
            "Early-likes-to-more-reach is a self-amplifying loop, so virality can be partly luck of the early signal",
            "Reach is endogenous to early engagement, so going viral does not establish the content was objectively best",
        ],
        [
            "Going viral proves the content was the best that day",
            "The ranking system played no causal role in the spread",
        ],
    ),
    (
        "Trust between two neighbouring countries erodes: each arms itself because it "
        "fears the other, and each fears the other more because it is arming. Tension "
        "rises year on year. A pundit blames one side for 'starting the spiral.' "
        "Evaluate the assignment of blame.",
        [
            "The arming is mutually reinforcing, so the spiral is sustained by both sides reacting to each other",
            "Identifying a first move does not make one side the ongoing cause of a self-feeding dynamic",
        ],
        [
            "One side is solely causing the rising tension",
            "Each side's arming is independent of the other's",
        ],
    ),
    # -- CONFOUND / correlation-not-causation ------------------------------
    (
        "Across a country's towns, ice-cream sales and drowning deaths rise and fall "
        "together through the year. A headline reads 'Ice cream linked to drowning.' "
        "What is the most likely explanation for the correlation?",
        [
            "A common cause (hot weather / summer) raises both ice-cream sales and swimming, hence drownings",
            "The correlation does not mean ice cream causes drowning",
        ],
        [
            "Ice cream consumption causes drowning",
            "Banning ice cream would reduce drownings",
        ],
    ),
    (
        "Children with bigger feet tend to read better. A vendor uses this to sell "
        "foot-stretching insoles 'to boost reading.' What is going on?",
        [
            "Age is a common cause: older children have bigger feet and read better",
            "Foot size and reading are correlated through age, not because one causes the other",
        ],
        [
            "Bigger feet cause better reading",
            "Stretching a child's feet would improve their reading",
        ],
    ),
    (
        "Hospitals with the most expensive equipment have higher patient death rates. "
        "A columnist argues expensive equipment is dangerous and should be removed. "
        "What is a more careful reading?",
        [
            "Sicker patients are referred to better-equipped hospitals, so case mix confounds the comparison",
            "The death-rate gap likely reflects which patients arrive, not harm from the equipment",
        ],
        [
            "Expensive equipment causes higher death rates",
            "Removing the equipment would lower deaths",
        ],
    ),
    (
        "People who take more vitamins are healthier on average. An ad concludes the "
        "vitamins are why. Vitamin-takers also exercise more, smoke less, and see "
        "doctors more often. What can be concluded about the vitamins themselves?",
        [
            "The healthier lifestyle of vitamin-takers confounds the comparison",
            "The correlation cannot isolate the vitamins' own effect from the surrounding habits",
        ],
        [
            "The vitamins are the cause of the better health",
            "Taking vitamins will make a sedentary smoker as healthy as the vitamin group",
        ],
    ),
    (
        "Countries with more Nobel laureates per capita also consume more chocolate "
        "per capita. A magazine suggests eating chocolate to win Nobels. What is the "
        "sober interpretation?",
        [
            "Both likely track national wealth / research funding, a common cause",
            "The correlation is ecological and does not show chocolate causes prizes",
        ],
        [
            "Eating more chocolate causes more Nobel prizes",
            "A person eating more chocolate raises their own chance of a Nobel",
        ],
    ),
    (
        "Within a company, departments that hold more meetings report higher morale. "
        "Leadership mandates more meetings everywhere to raise morale. What is the "
        "flaw, if any?",
        [
            "The correlation may run the other way or share a cause (healthy teams both meet and feel good)",
            "Mandating meetings need not transfer the morale, since meetings may be a marker rather than a driver",
        ],
        [
            "Holding more meetings causes higher morale",
            "Mandated meetings will reliably raise morale in low-morale teams",
        ],
    ),
    (
        "A study finds neighbourhoods with more police have more reported crime. A "
        "candidate says police cause crime and pledges to cut the force. Reporting "
        "and deployment both respond to where crime already is. Assess.",
        [
            "Police are deployed to high-crime areas and more police can raise reporting, so the arrow is plausibly reversed or two-way",
            "The positive correlation does not establish that police presence causes crime",
        ],
        [
            "More police causes more crime",
            "Cutting the police force would reduce crime",
        ],
    ),
    (
        "Students who use a paid tutoring app score higher on the final. The app "
        "advertises this as proof it works. Families who buy the app are wealthier "
        "and more involved. What does the score gap actually support?",
        [
            "Selection: families who buy the app differ in wealth and involvement, which also raise scores",
            "The raw gap cannot, on its own, prove the app caused the higher scores",
        ],
        [
            "The app is the cause of the higher scores",
            "Giving the app to an uninvolved family guarantees the same score gain",
        ],
    ),
    (
        "Sales of a product jump in the same month a new ad campaign launches. The "
        "marketing team claims credit. The launch was also the start of the holiday "
        "season and a competitor went out of business. What is warranted?",
        [
            "Several causes coincide (ad, holiday season, competitor exit), so the jump is not cleanly attributable to the ad",
            "The timing alone does not isolate the campaign's effect from the other simultaneous changes",
        ],
        [
            "The ad campaign is the cause of the entire sales jump",
            "The holiday timing and competitor exit can be ignored",
        ],
    ),
    (
        "Patients given a new painkiller in a clinic report less pain a week later. "
        "There was no comparison group, and pain from the injury they came in with "
        "usually fades within a week anyway. What does the improvement show?",
        [
            "Without a control group, natural recovery (and placebo effects) can explain the improvement",
            "The before-after drop does not establish the drug caused the relief",
        ],
        [
            "The painkiller caused the reduction in pain",
            "The improvement rules out natural recovery",
        ],
    ),
    (
        "A city installs red-light cameras at its worst intersections; the next year "
        "crashes there fall. Officials credit the cameras. The cameras went in "
        "precisely where crash counts had spiked to record highs. What confound is "
        "in play?",
        [
            "Regression to the mean: intersections chosen for record-high counts tend to fall back the next year regardless",
            "Selecting the worst sites means some of the drop is expected even with no real camera effect",
        ],
        [
            "The cameras are proven to be the cause of the entire drop",
            "Regression to the mean can be dismissed here",
        ],
    ),
    # -- MULTICAUSE / attribution -----------------------------------------
    (
        "A wildfire spread badly. There had been a long drought, strong winds that "
        "day, dense undergrowth from years of fire suppression, and a discarded "
        "cigarette that started it. A report wants to name 'the cause.' How should "
        "the cause be characterised?",
        [
            "Several factors are jointly necessary; the outcome is over-determined rather than from one cause",
            "The ignition and the conditions (drought, wind, fuel load) all contributed and interact",
        ],
        [
            "The cigarette alone is the cause of the fire's severity",
            "The drought, wind, and fuel load were irrelevant to how badly it spread",
        ],
    ),
    (
        "A startup failed. It had a weak product, ran out of money, lost its lead "
        "engineer, and a bigger rival launched the same month. Investors ask what "
        "killed it. What is the most honest attribution?",
        [
            "Multiple interacting causes contributed; no single one is cleanly 'the' cause",
            "The failure is over-determined, so picking one factor oversimplifies",
        ],
        [
            "One single factor is definitively what killed the startup",
            "The other factors had no bearing on the outcome",
        ],
    ),
    (
        "A river flooded a town. Upstream deforestation, an unusually heavy storm, a "
        "poorly maintained levee, and new pavement that sped runoff all played a "
        "part. A lawsuit wants to pin it on the levee operator alone. Evaluate.",
        [
            "The flood resulted from several contributing causes acting together",
            "Singling out the levee ignores the storm, deforestation, and runoff that also mattered",
        ],
        [
            "The levee failure is the sole cause of the flood",
            "The storm and land-use changes made no difference",
        ],
    ),
    (
        "A team won a championship. They had a strong roster, a lucky bracket, a "
        "key rival's injury, and excellent coaching. A columnist credits the coach "
        "entirely. How should the win be attributed?",
        [
            "The win came from several factors together, including luck outside anyone's control",
            "Crediting the coach alone overstates one cause among many contributors",
        ],
        [
            "The coaching alone caused the championship",
            "Luck and the rival's injury played no role",
        ],
    ),
    (
        "An economy recovered from recession. Interest-rate cuts, a fiscal stimulus, "
        "a rebound in global trade, and renewed consumer confidence all coincided. "
        "A politician claims their stimulus did it. What is supportable?",
        [
            "Multiple simultaneous forces contributed to the recovery",
            "The timing does not let the stimulus's share be separated cleanly from the other forces",
        ],
        [
            "The stimulus alone caused the recovery",
            "Global trade and rate cuts contributed nothing",
        ],
    ),
    (
        "A patient recovered after starting a new diet, a new medication, more sleep, "
        "and less work stress at the same time. They credit the diet. What is the "
        "right epistemic stance on the cause of recovery?",
        [
            "Several changes happened together, so the diet's specific contribution is not isolable",
            "Recovery is consistent with any or all of the changes, or their combination",
        ],
        [
            "The diet is definitely what caused the recovery",
            "The medication and reduced stress can be ruled out as causes",
        ],
    ),
    (
        "A bridge collapsed. It was old, overloaded that day, corroded from road "
        "salt, and last inspected years late. An op-ed blames the heavy truck on it "
        "at the moment of collapse. How should responsibility for the collapse be "
        "framed?",
        [
            "The collapse had multiple contributing causes (age, corrosion, overload, missed inspection)",
            "The truck was a trigger acting on an already-weakened structure, not the whole cause",
        ],
        [
            "The truck alone is the cause of the collapse",
            "The corrosion and skipped inspections were irrelevant",
        ],
    ),
    # -- LOCALITY / partial perspective / who-can-know-what ----------------
    (
        "Ana stands at the north gate of a walled garden and sees no one inside; Ben "
        "stands at the south gate and sees three people. They radio each other. From "
        "their two partial views, what can be said about how many people are in the "
        "garden?",
        [
            "Neither single viewpoint sees the whole garden, so each alone is incomplete",
            "Combining their views gives at least three, but there may be people neither can see",
        ],
        [
            "There are exactly three people in the garden",
            "Ana's empty view proves her half of the garden is empty",
        ],
    ),
    (
        "A relay of runners passes a baton through fog. Each runner sees only the "
        "handoff just before and just after their own leg. At the end someone asks "
        "any single runner who is winning the whole race. What can one runner "
        "legitimately know?",
        [
            "Each runner has only local information about their own segment",
            "No single runner can know the whole race's standing from their local view alone",
        ],
        [
            "A single runner can correctly state who is winning overall",
            "Local knowledge of one handoff settles the global outcome",
        ],
    ),
    (
        "Two telescopes on opposite sides of the planet each track a comet for the "
        "few hours it is above their own horizon. A journalist asks one observatory "
        "for the comet's complete path. What is the honest answer about that single "
        "observatory's knowledge?",
        [
            "Each observatory sees only the arc above its own horizon, a partial track",
            "The full path requires combining observations; one site cannot supply it alone",
        ],
        [
            "One observatory already knows the comet's complete path",
            "The hours below an observatory's horizon are knowable from that site alone",
        ],
    ),
    (
        "In a distributed chat, each server only knows the messages it has received "
        "so far; network delays mean different servers have seen different subsets. "
        "Someone asks one server for the single true global order of all messages. "
        "What is the correct stance?",
        [
            "Each server has a partial, possibly stale view of the message set",
            "No one server can assert the definitive global ordering from its local state alone",
        ],
        [
            "One server can report the true global order of all messages",
            "A server's local view is guaranteed complete and current",
        ],
    ),
    (
        "Three witnesses each saw part of a street accident from a different corner. "
        "One saw the approach, one the impact, one the aftermath. A detective asks "
        "whether any one of them can give the complete account. What follows from "
        "the geometry of their views?",
        [
            "Each witness has only a partial slice of the event from one vantage",
            "A complete account needs the views combined; no single witness has it all",
        ],
        [
            "One witness can give the complete and certain account alone",
            "A partial view from one corner is the whole truth of the accident",
        ],
    ),
    (
        "A submarine and a plane each track a school of fish: the sub by sonar from "
        "below, the plane by sight from above, and each loses the school when it "
        "moves into the other's blind region. Asked for the school's full path, what "
        "can either platform alone provide?",
        [
            "Each sensor covers only part of the school's movement (its own region)",
            "Neither platform alone can give the full path; their coverage is complementary",
        ],
        [
            "Either one alone already has the school's full path",
            "The blind region of one platform is fully knowable from that platform",
        ],
    ),
    # -- MUTUAL conflict / both-sides causation ----------------------------
    (
        "Two roommates each say the other started leaving dishes in the sink, and "
        "each leaves more dishes 'in response' to the other. The pile grows. A "
        "mediator asks who is at fault. What is the most accurate description of the "
        "dynamic?",
        [
            "Each one's behaviour is both a reaction to and a cause of the other's, a mutual loop",
            "Assigning sole fault misses that both are sustaining the pattern by reacting to each other",
        ],
        [
            "One roommate is solely at fault for the dish pile",
            "Only one roommate's actions cause the other's",
        ],
    ),
    (
        "In a marriage, one partner withdraws when the other criticises, and the "
        "other criticises more when the first withdraws. Each cites the other's "
        "behaviour as the reason for their own. A counsellor is asked who is causing "
        "the conflict. Characterise the causal pattern.",
        [
            "Criticism and withdrawal cause each other in a self-perpetuating cycle",
            "Neither partner is simply the cause; the pattern is maintained mutually",
        ],
        [
            "One partner is the single cause of the conflict",
            "The withdrawal has no causal effect on the criticism",
        ],
    ),
    (
        "Two countries impose tariffs on each other; each new tariff is justified as "
        "retaliation for the other's last tariff, and trade between them shrinks. "
        "Each government blames the other for the trade war. Assess the causal "
        "structure of the escalation.",
        [
            "Each tariff is both effect of and cause for the next, a reciprocal escalation",
            "Blaming one side alone ignores that both are driving the spiral through retaliation",
        ],
        [
            "One country is solely responsible for the escalation",
            "Each country's tariffs are unrelated to the other's",
        ],
    ),
    (
        "On a team, the engineers say they hide problems because management punishes "
        "bad news, and management says it punishes bad news because the engineers "
        "hide problems until they explode. Trust keeps falling. Who is causing the "
        "breakdown in trust?",
        [
            "Hiding problems and punishing bad news reinforce each other in a vicious cycle",
            "The breakdown is a mutual dynamic, not the fault of one side alone",
        ],
        [
            "One side is solely causing the breakdown in trust",
            "The hiding and the punishing are causally independent of each other",
        ],
    ),
    (
        "Predators and prey in a valley cycle: when prey are plentiful, predators "
        "multiply and eat them down; when prey are scarce, predators starve and "
        "decline, letting prey recover. A naturalist asks whether predators control "
        "prey or prey control predators. What is the better framing?",
        [
            "Predator and prey numbers drive each other cyclically; causation runs both ways",
            "Neither simply controls the other; they are coupled in a feedback cycle",
        ],
        [
            "Predators one-sidedly control prey numbers with no reverse effect",
            "Prey abundance has no causal influence on predator numbers",
        ],
    ),
    (
        "A currency and confidence in it feed back: falling confidence drives the "
        "currency down, and a falling currency further erodes confidence. During a "
        "crisis both collapse together. An official insists the fall was caused "
        "purely by speculators 'from outside.' Evaluate.",
        [
            "Confidence and the currency's value drive each other downward in a loop",
            "The joint collapse is consistent with internal feedback, not solely an external push",
        ],
        [
            "Outside speculators are the sole cause of the collapse",
            "Confidence and the currency value do not affect each other",
        ],
    ),
    (
        "A teacher and a class fall into a spiral: the class acts out because they "
        "find the teacher harsh, and the teacher grows harsher because the class "
        "acts out. By term's end both are at their worst. The principal asks who is "
        "responsible. Describe the causation.",
        [
            "Harshness and acting-out cause each other, sustaining the spiral mutually",
            "Responsibility is shared in the loop rather than located in one party",
        ],
        [
            "One party alone is the cause of the deteriorating classroom",
            "The teacher's harshness and the class's behaviour are causally unrelated",
        ],
    ),
]


def build_items(n: int = 40, seed: int = 1443) -> list[Item]:
    """Return up to ``n`` correlative-reasoning items (deterministic order).

    ``seed`` is accepted for signature parity with the other task families; the
    scenarios are hand-authored and emitted in a fixed order, so the seed does not
    randomize anything here (the set is small and curated rather than sampled).
    """
    items: list[Item] = []
    for i, (question, points, traps) in enumerate(_SCENARIOS[:n]):
        items.append(
            Item(
                id=f"co-{i:03d}",
                task="correlative",
                question=question,
                gold={"points": list(points), "traps": list(traps)},
                meta={"n_points": len(points), "n_traps": len(traps)},
            )
        )
    return items


def score(item: Item, answer: str, full_text: str) -> dict[str, Any]:
    """Judge the plain-English conclusion against this item's points/traps rubric.

    Register-blind by construction: we judge ``extract_answer(full_text)`` (the
    final ANSWER line), never the reasoning above it. The panel call is resumable
    via the judge cache, so re-scoring is cheap. ``rubric_score`` is the fraction
    of points captured minus a penalty per trap committed (clamped to [-1, 1]).
    """
    points: list[str] = item.gold.get("points", [])
    traps: list[str] = item.gold.get("traps", [])

    # The conclusion the judge is allowed to see. ``answer`` is already the
    # extracted ANSWER line from the score driver; fall back to extracting it from
    # the full text if a caller passed something else.
    conclusion = (answer or "").strip() or extract_answer(full_text)

    result = judge_mod.judge_rubric(conclusion, points, traps)

    points_hit = result["n_points_hit"]
    points_total = result["n_points_total"]
    traps_triggered = result["n_traps_triggered"]

    frac = (points_hit / points_total) if points_total else 0.0
    rubric_score = frac - _TRAP_PENALTY * traps_triggered
    # Keep it in a sane band so a single confidently-wrong item can't dominate a mean.
    rubric_score = max(-1.0, min(1.0, rubric_score))

    return {
        "primary": "rubric_score",
        "points_hit": points_hit,
        "points_total": points_total,
        "traps_triggered": traps_triggered,
        "traps_total": result["n_traps_total"],
        "rubric_score": rubric_score,
        "per_point": result["per_point"],
        "per_trap": result["per_trap"],
        "answered": has_answer_sigil(full_text),
    }
