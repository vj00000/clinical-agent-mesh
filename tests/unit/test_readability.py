"""Reading-grade scoring.

The absolute number matters less than the ordering: plain instructions must score
below clinical prose, or the simplification gate fires on the wrong text.
"""

from mesh.readability import count_syllables, flesch_kincaid_grade

PLAIN = "Take one pill each day. Drink water with it."

CLINICAL = (
    "Discontinue the anticoagulant medication immediately if unexplained "
    "haemorrhage manifests and consult your prescribing physician regarding "
    "alternative antithrombotic therapy."
)


def test_a_silent_trailing_e_is_not_a_syllable():
    assert count_syllables("take") == 1


def test_adjacent_vowels_are_one_group():
    assert count_syllables("each") == 1


def test_a_long_word_counts_every_vowel_group():
    assert count_syllables("medication") == 4


def test_a_wordless_string_has_no_syllables():
    assert count_syllables("...") == 0


def test_plain_instructions_read_easily():
    assert flesch_kincaid_grade(PLAIN) < 8.0


def test_clinical_prose_reads_hard():
    assert flesch_kincaid_grade(CLINICAL) > 12.0


def test_empty_text_scores_zero_rather_than_raising():
    assert flesch_kincaid_grade("") == 0.0
