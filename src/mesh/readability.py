"""Reading-grade scoring for patient-facing text.

Flesch-Kincaid is computed rather than judged. Asking a model "is this easy to
read?" gets an agreeable answer; counting syllables gets a number that can fail a
build.

The syllable count is a vowel-group heuristic, not a dictionary lookup. It is
wrong on individual words ("queue", "business") and close enough in aggregate,
which is all a grade level over several sentences needs.
"""

import re

_VOWELS = "aeiouy"
_SENTENCE_END = re.compile(r"[.!?]+")


def count_syllables(word: str) -> int:
    """Count vowel groups, discounting a silent trailing 'e'."""
    cleaned = "".join(character for character in word.lower() if character.isalpha())
    if not cleaned:
        return 0

    groups = 0
    previous_was_vowel = False
    for character in cleaned:
        is_vowel = character in _VOWELS
        if is_vowel and not previous_was_vowel:
            groups += 1
        previous_was_vowel = is_vowel

    if cleaned.endswith("e") and groups > 1:
        groups -= 1

    return max(groups, 1)


def flesch_kincaid_grade(text: str) -> float:
    """US school grade required to read `text`.

    Empty or wordless text scores 0.0 rather than raising: an unwritten answer is
    not an unreadable one, and the caller's next step is to look at the answer.
    """
    words = [token for token in re.split(r"\s+", text) if any(c.isalpha() for c in token)]
    sentences = [part for part in _SENTENCE_END.split(text) if part.strip()]

    if not words or not sentences:
        return 0.0

    syllables = sum(count_syllables(word) for word in words)

    return 0.39 * (len(words) / len(sentences)) + 11.8 * (syllables / len(words)) - 15.59
