"""Deterministic rules for the discharge specialist: readability and warning.

Neither of these is left to the model. Reading grade is arithmetic, so it is 
computed rather than judged, and the interaction warning are composed here so a
model that forgets one cannot drop it from the instructions.

Demo depth. Flesch-Kincaid is a crude proxy for comprehension -- it counts
sullables, not whether "hold your metformin" means anything to the reader -- but 
it is objective, reproducible, and catches the worst offenders.
"""

import re

from pydantic import BaseModel, Field

# Us health-literacy guidance puts patient material at roughly grade 6-8. Eight
# is the generous end of that, chosen so the gate fires on genuinely dense prose
# rather than on every third sentence
MAX_READING_GRADE = 8.0

_VOWEL = "aeiouy"
_SENTENCE_END = re.compile(r"[.!?]+")
_WORD = re.compile(r"[A-Za-z']+")

class Medication(BaseModel):
    name: str = Field(min_length=1)
    dose: str = ""
    frequency: str = ""

class Interaction(BaseModel):
    """One interaction reported by the drug-label lookup."""

    drugs: str = Field(min_length=1)
    interactions_with: str = Field(min_length=1)
    warning: str = Field(min_length=1)

def count_syllables(word: str) -> int:
    """Count vowel groups, discounting a silent trailing 'e'.

    A hueristic, not a dictionary. It miscounts a handful of words, which is 
    acceptable: the score is compared agains a thresold, not reported as fact.
    """

    cleaned = "".join(character for character in word.lower() if character.isalpha())
    if not cleaned:
        return 0
    groups = 0
    previous_was_vowel = False
    for character in cleaned:
        is_vowel = character in _VOWEL
        if is_vowel and not previous_was_vowel:
            groups += 1
        previous_was_vowel = is_vowel

    if cleaned.endswith("e") and groups > 1:
        groups -= 1

    return max(groups, 1)  # every word has at least one syllable

def flesch_kincaid_grade(text: str) -> float:
    """The US school grade needed to read this text.
    
    Empty or wordless text scores 0.0 rather than raising: an empty draft is a 
    problem for the drafter to answer for, not for the readablity gate.
    """
    
    sentences = [part for part in _SENTENCE_END.split(text) if part.strip()]
    words = _WORD.findall(text)
    if not sentences or not words:
        return 0.0

    syllables = sum(count_syllables(word) for word in words)
    return (0.39 * (len(words) / len(sentences)) + 11.8 * (syllables / len(words)) - 15.59)

def needs_simplifying(text: str) -> bool:
    return flesch_kincaid_grade(text) > MAX_READING_GRADE

INTERACTION_HEADING = "Check these before you start:"

def render_interactions(interactions: list[Interaction]) -> str:
    """Compose the interaction warning, or an empty string when there are none.

    Written here rather than left to the model: a lookup that finds an 
    interaction the instruction then omit is worse than not looking it up.
    """
    if not interactions:
        return ""

    lines = [INTERACTION_HEADING]
    lines.extend(
        f"- {interaction.drugs} with {interaction.interactions_with}: {interaction.warning}"
        for interaction in interactions 
    )

    return "\n".join(lines)

