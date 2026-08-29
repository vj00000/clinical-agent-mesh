"""Deterministic triage policy: what the rules decide without the model.

These are the decisions that must not depend on model variance — whether to ask a
follow-up at all, and whether the model may talk the urgency back down.
"""

from mesh.agents.triage_rules import (
    Urgency,
    apply_urgency_floor,
    assess_urgency,
    needs_more_detail,
)


def test_the_model_can_raise_urgency_above_the_rules():
    assert apply_urgency_floor(Urgency.EMERGENCY, floor=Urgency.ROUTINE) is Urgency.EMERGENCY


def test_the_model_cannot_talk_a_red_flag_down():
    """The rules are a floor. A prompt regression that made the model reassuring
    must not be able to downgrade a red flag that already fired."""
    assert apply_urgency_floor(Urgency.ROUTINE, floor=Urgency.EMERGENCY) is Urgency.EMERGENCY


def test_agreement_passes_through():
    assert apply_urgency_floor(Urgency.ROUTINE, floor=Urgency.ROUTINE) is Urgency.ROUTINE


def test_a_thin_description_is_asked_about():
    assert needs_more_detail("my head hurts", matched=[]) is True


def test_a_detailed_description_is_not():
    assert (
        needs_more_detail(
            "I have had a dull headache behind my eyes since yesterday afternoon",
            matched=[],
        )
        is False
    )


def test_a_red_flag_is_never_asked_about():
    """Never ask a clarifying question of someone describing an emergency. The
    follow-up exists for thin descriptions, not for stalling on a red flag."""
    assessment = assess_urgency("crushing chest pain")

    assert assessment.level is Urgency.EMERGENCY
    assert needs_more_detail("crushing chest pain", matched=assessment.matched) is False
