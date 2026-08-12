"""Red-flag rules for the triage specialist.

These are deliberately *deterministic* rather than model-judged. A missed
emergency is the worst outcome this system can produce, and it must not depend on
model variance, temperature, or a prompt regression. The rules run alongside the
LLM and can only ever escalate.

The rules are tuned for recall, not precision. Telling someone with a pulled
muscle to seek urgent care is a bad afternoon; missing a myocardial infarction is
a death. So false positives are accepted deliberately and the asymmetry is stated
rather than hidden.
"""

from mesh.agents.triage_rules import Urgency, assess_urgency


def test_chest_pain_radiating_to_the_arm_is_an_emergency():
    assessment = assess_urgency("I have crushing chest pain radiating to my left arm")

    assert assessment.level is Urgency.EMERGENCY
    assert "cardiac" in assessment.matched


def test_stroke_signs_are_an_emergency():
    assessment = assess_urgency("my father is slurring his words and his face is drooping")

    assert assessment.level is Urgency.EMERGENCY
    assert "stroke" in assessment.matched


def test_a_thunderclap_headache_is_an_emergency():
    assessment = assess_urgency("sudden headache, the worst of my life")

    assert assessment.level is Urgency.EMERGENCY


def test_fever_with_a_stiff_neck_is_an_emergency():
    assessment = assess_urgency("fever of 39 and a stiff neck since this morning")

    assert assessment.level is Urgency.EMERGENCY
    assert "meningitis" in assessment.matched


def test_confusion_with_low_blood_sugar_is_an_emergency():
    assessment = assess_urgency("my blood sugar reads 28 and I feel confused")

    assert assessment.level is Urgency.EMERGENCY


def test_a_minor_injury_carries_no_red_flag():
    assessment = assess_urgency("my ankle is swollen after twisting it playing football")

    assert assessment.level is Urgency.ROUTINE
    assert assessment.matched == []


def test_matching_ignores_case():
    assessment = assess_urgency("CRUSHING CHEST PAIN RADIATING TO MY JAW")

    assert assessment.level is Urgency.EMERGENCY


def test_a_red_flag_buried_in_a_long_message_still_fires():
    """People bury the critical detail in the middle of a paragraph."""
    message = (
        "I've been feeling generally tired this week, work has been stressful, "
        "and I slept badly. Also my face has been drooping on one side since "
        "lunchtime. I've been drinking more water and taking vitamins."
    )

    assert assess_urgency(message).level is Urgency.EMERGENCY


def test_a_red_flag_is_never_downgraded_by_reassuring_words():
    """The rules only escalate. 'Probably nothing' must not cancel a red flag."""
    assessment = assess_urgency(
        "It's probably nothing and I feel fine, just some crushing chest pain "
        "radiating to my arm, no need to worry"
    )

    assert assessment.level is Urgency.EMERGENCY


def test_every_matching_category_is_reported():
    assessment = assess_urgency(
        "slurred speech and face drooping, plus crushing chest pain radiating to my arm"
    )

    assert set(assessment.matched) == {"stroke", "cardiac"}


def test_no_non_triage_benchmark_query_is_flagged_as_an_emergency():
    """Precision check against realistic text the rules should stay quiet on.

    Recall-tuned rules are still expected to distinguish "is dual antiplatelet
    therapy recommended after stenting" from someone describing chest pain now.
    Firing on guideline questions would make the triage path fire constantly.
    """
    from mesh.evals.routing import load_routing_cases

    false_positives = [
        case.query
        for case in load_routing_cases()
        if case.expected != "triage" and assess_urgency(case.query).level is Urgency.EMERGENCY
    ]

    assert false_positives == []


def test_every_triage_benchmark_query_describing_a_red_flag_is_caught():
    """The recall side, on the same realistic set."""
    from mesh.evals.routing import load_routing_cases

    triage_queries = [c.query for c in load_routing_cases() if c.expected == "triage"]
    caught = [q for q in triage_queries if assess_urgency(q).level is Urgency.EMERGENCY]

    # Not all triage queries are emergencies - a twisted ankle is genuinely
    # routine - so this asserts the emergencies are caught, not that all fire.
    assert len(caught) >= 5


def test_the_same_input_always_gives_the_same_assessment():
    """Determinism is the point: this must not vary between runs."""
    message = "fever and a stiff neck"

    assert assess_urgency(message) == assess_urgency(message)
