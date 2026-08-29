"""Deterministic red-flag rules for the triage specialist.

These run *alongside* the LLM rather than inside it. A missed emergency is the
worst outcome this system can produce, and it must not depend on model variance,
sampling temperature, or an unnoticed prompt regression. The rules can only
escalate — nothing here downgrades an urgency level.

Tuned for recall, not precision, and that trade is deliberate: sending someone
with a pulled muscle to urgent care costs them an afternoon, while missing a
myocardial infarction costs a life. The false positives are the price.

This is a demo-depth specialist. A production version would use a validated
protocol such as the Manchester Triage System with clinician review, not a
hand-written pattern table.

Known limitation - the scale has two levels, and that is too coarse. New
exertional dyspnoea ("short of breath climbing one flight of stairs") is
clinically concerning and warrants being seen soon, but it is not an immediate
emergency; with only EMERGENCY and ROUTINE it collapses into ROUTINE, which
undersells it. A real implementation needs an URGENT tier between the two. The
LLM half of the specialist can still surface such cases in its response - the
rules are a floor on urgency, not the whole assessment.
"""

import re
from enum import StrEnum

from pydantic import BaseModel


class Urgency(StrEnum):
    EMERGENCY = "emergency"
    ROUTINE = "routine"


class TriageAssessment(BaseModel):
    level: Urgency
    matched: list[str]


# Each pattern requires the co-occurring features that make a symptom a red flag,
# not the symptom alone: "chest pain" on its own is extremely common, while chest
# pain radiating to the arm or jaw is the classic presentation.
_RED_FLAGS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "cardiac",
        re.compile(
            r"chest (?:pain|pressure|tightness)[\s\S]{0,60}?"
            r"(?:radiat\w*|left arm|jaw|shoulder|back)"
            r"|(?:crushing|elephant)[\s\S]{0,30}?chest",
            re.IGNORECASE,
        ),
    ),
    (
        "stroke",
        re.compile(
            r"slurr\w*\s+(?:speech|words|his words|her words|my words)"
            r"|face\s+(?:is\s+|has\s+been\s+)?droop\w*"
            r"|droop\w*[\s\S]{0,20}?(?:face|mouth|eyelid)"
            r"|sudden\w*\s+(?:weakness|numbness)[\s\S]{0,30}?one side",
            re.IGNORECASE,
        ),
    ),
    (
        "thunderclap_headache",
        re.compile(
            r"sudden\w*[\s\S]{0,30}?headache|headache[\s\S]{0,40}?worst of (?:my|her|his) life"
            r"|worst[\s\S]{0,20}?headache",
            re.IGNORECASE,
        ),
    ),
    (
        "meningitis",
        re.compile(r"fever[\s\S]{0,60}?stiff neck|stiff neck[\s\S]{0,60}?fever", re.IGNORECASE),
    ),
    (
        "hypoglycaemia",
        re.compile(
            r"(?:blood sugar|glucose)[\s\S]{0,40}?(?:confus\w*|shak\w*|sweat\w*|faint\w*)"
            r"|(?:confus\w*|faint\w*)[\s\S]{0,40}?(?:blood sugar|glucose)",
            re.IGNORECASE,
        ),
    ),
    (
        "respiratory",
        re.compile(
            r"(?:can'?t|cannot|unable to)\s+breathe|struggling to breathe|gasping",
            re.IGNORECASE,
        ),
    ),
    (
        "anaphylaxis",
        re.compile(
            r"(?:throat|tongue|lips?)[\s\S]{0,30}?swell\w*|swell\w*[\s\S]{0,20}?throat",
            re.IGNORECASE,
        ),
    ),
)


def assess_urgency(text: str) -> TriageAssessment:
    """Match red-flag patterns and escalate if any fire.

    Reassuring language in the same message is ignored on purpose: patients
    routinely minimise ("it's probably nothing, just some chest pain"), and a rule
    that agreed with them would defeat its own purpose.
    """
    matched = [label for label, pattern in _RED_FLAGS if pattern.search(text)]

    return TriageAssessment(
        level=Urgency.EMERGENCY if matched else Urgency.ROUTINE,
        matched=matched,
    )


# Ascending severity. A tuple rather than the enum's own order, so adding the
# URGENT tier the module docstring calls for is a one-line change here.
_SEVERITY: tuple[Urgency, ...] = (Urgency.ROUTINE, Urgency.EMERGENCY)

# Below this, there is not enough to assess. Deliberately generous: "chest hurts
# when I breathe" is five words and worth one follow-up rather than a guess.
MIN_DESCRIPTION_WORDS = 6


def apply_urgency_floor(assessed: Urgency, *, floor: Urgency) -> Urgency:
    """Take the higher of the model's assessment and the rules' floor.

    The rules can only escalate. Nothing the model says downgrades a red flag, so
    a prompt regression cannot quietly reassure someone into staying home.
    """
    return max(assessed, floor, key=_SEVERITY.index)


def needs_more_detail(description: str, *, matched: list[str]) -> bool:
    """Whether to ask one follow-up before assessing.

    A fired red flag outranks any need for detail: someone describing crushing
    chest pain gets an answer, not a questionnaire.
    """
    if matched:
        return False

    return len(description.split()) < MIN_DESCRIPTION_WORDS
