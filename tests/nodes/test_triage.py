"""The triage specialist subgraph.

The rules are a floor, not a suggestion, and the follow-up question fires on a
deterministic gate rather than on the model deciding to ask one.
"""

from typing import Any

from mesh.agents.triage import EMERGENCY_BANNER, FOLLOW_UP_QUESTION, build_triage_subgraph
from mesh.agents.triage_rules import Urgency
from mesh.guardrails.nodes import SAFETY_ESCALATION_FLAG
from mesh.retrieval.chunking import Chunk
from mesh.state import Citation


class StubRetriever:
    def search(self, query: str, *, top_k: int = 20) -> list[str]:
        return ["c1", "c2"]


class StubStore:
    def fetch(self, chunk_ids: list[str]) -> list[Chunk]:
        return [
            Chunk(chunk_id=cid, source="medlineplus", text=f"passage {cid}", ordinal=i)
            for i, cid in enumerate(chunk_ids)
        ]


class PassThroughReranker:
    def rerank(self, query: str, candidates: list[Chunk], *, top_n: int) -> list[Chunk]:
        return list(candidates[:top_n])


def _adviser(text: str, level: Urgency):
    def advise(query: str, chunks: list[Chunk]) -> tuple[str, list[Citation], Urgency]:
        return text, [Citation(chunk_id="c1", source="medlineplus", quote="...")], level

    return advise


def _build(**overrides: Any):
    defaults: dict[str, Any] = {
        "retriever": StubRetriever(),
        "store": StubStore(),
        "reranker": PassThroughReranker(),
        "advise": _adviser("Rest and see your GP if it persists.", Urgency.ROUTINE),
    }
    return build_triage_subgraph(**{**defaults, **overrides})


def test_a_thin_description_gets_one_follow_up_question():
    result = _build().invoke({"query": "my head hurts"})

    assert result["answer"] == FOLLOW_UP_QUESTION
    assert result["route"] == "clarify"
    assert result["citations"] == []


def test_a_red_flag_is_answered_not_questioned():
    """Three words, well under the follow-up threshold — but nobody describing
    crushing chest pain should be handed a questionnaire."""
    result = _build().invoke({"query": "crushing chest pain"})

    assert result["answer"] != FOLLOW_UP_QUESTION
    assert result["answer"].startswith(EMERGENCY_BANNER)


def test_the_model_cannot_downgrade_a_red_flag():
    subgraph = _build(advise=_adviser("Probably muscular.", Urgency.ROUTINE))

    result = subgraph.invoke({"query": "crushing chest pain radiating to my left arm"})

    assert result["level"] is Urgency.EMERGENCY
    assert result["answer"].startswith(EMERGENCY_BANNER)
    assert "Probably muscular." in result["answer"]


def test_the_model_may_raise_urgency_when_no_rule_fired():
    """The floor works both ways: rules cannot be lowered, but they are a floor,
    not a ceiling. The rule table is deliberately incomplete."""
    subgraph = _build(advise=_adviser("This looks serious.", Urgency.EMERGENCY))

    result = subgraph.invoke({"query": "I have been vomiting blood since this morning"})

    assert result["answer"].startswith(EMERGENCY_BANNER)


def test_an_escalation_is_flagged_so_guard_out_lets_it_through():
    result = _build().invoke({"query": "crushing chest pain"})

    assert SAFETY_ESCALATION_FLAG in result["guard_flags"]


def test_a_routine_answer_carries_no_banner_and_no_escalation_flag():
    result = _build().invoke({"query": "I have had a mild sore throat for two days now"})

    assert not result["answer"].startswith(EMERGENCY_BANNER)
    assert result["guard_flags"] == []


def test_the_subgraph_returns_what_the_mesh_needs():
    result = _build().invoke({"query": "I have had a mild sore throat for two days now"})

    assert {"answer", "citations", "retrieved_ids"} <= set(result)
