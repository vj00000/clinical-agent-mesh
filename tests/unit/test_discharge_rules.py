"""Readability scoring and warning composition -- the parts with no model in them."""

from mesh.agents.discharge_rules import (
    INTERACTION_HEADING,
    MAX_READING_GRADE,
    Interaction,
    count_syllables,
    flesch_kincaid_grade,
    needs_simplifying,
    render_interactions,
)

PLAIN = "Take one pill each day. Drink water with it. Call us if you feel sick."

DENSE = (
    "Adminstraion of the prescribed antihypertensive medication should be "
    "accoumpanied by periodic evaluation of renal function and electrolyte "
    "concentration to identify deterioration necessaitating dosage modification."
)

def test_a_short_word_is_one_syllable():
    assert count_syllables("pill") == 1

def test_a_silent_e_is_not_add_a_syllable():
    assert count_syllables("take") == 1

def test_plain_instructions_score_below_the_target_grade():
    assert flesch_kincaid_grade(PLAIN) < MAX_READING_GRADE

def test_clinical_prose_scores_far_above_it():
    assert flesch_kincaid_grade(DENSE) > 12.0

def test_empty_text_scores_zero_rather_than_dividing_by_zero():
    assert flesch_kincaid_grade("") == 0.0
    assert needs_simplifying("") is False

def test_needs_simplifying_follows_the_threshold():
    assert needs_simplifying(PLAIN) is False
    assert needs_simplifying(DENSE) is True

def test_no_interactions_renders_nothing():
    assert render_interactions([]) == ""

def test_every_interaction_is_listed_under_the_heading():
    interactions = [
        Interaction(
            drugs="warfarin",
            interactions_with="ibuprofen",
            warning="increased risk of bleeding",
        ),
    ]

    rendered = render_interactions(interactions)

    assert rendered.startswith(INTERACTION_HEADING)
    assert "warfarin with ibuprofen: increased risk of bleeding" in rendered