"""The HTTP surface, driven by a stub mesh.

`create_app` takes the mesh as an argument, so request validation, response
shaping, and the SSE event sequence are all testable with no key, no Chroma and
no network.
"""

from typing import Any

import pytest
from fastapi.testclient import TestClient

from mesh.api.app import create_app
from mesh.state import Citation

ANSWERED = {
    "answer": "Thiazides are first-line.",
    "route": "guideline",
    "confidence": 0.92,
    "citations": [Citation(chunk_id="c1", source="cdc-htn", quote="...")],
    "guard_flags": [],
    "trace_id": "trace-1",
}


class StubMesh:
    def __init__(self, state: dict[str, Any]) -> None:
        self.state = state
        self.configs: list[Any] = []

    def invoke(self, initial: dict[str, Any], config: Any = None) -> dict[str, Any]:
        self.configs.append(config)
        return {**self.state, "query": initial["query"]}

    async def astream(self, initial: dict[str, Any], config: Any = None) -> Any:
        for node in ("guard_in", "supervisor", "guideline", "guard_out"):
            yield {node: self.state if node == "guard_out" else {}}


@pytest.fixture
def client():
    return TestClient(create_app(StubMesh(ANSWERED)))


def test_health_reports_ok(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_a_question_comes_back_with_its_citations(client):
    body = client.post("/ask", json={"query": "first-line therapy for hypertension?"}).json()

    assert body["answer"] == "Thiazides are first-line."
    assert body["route"] == "guideline"
    assert body["citations"][0]["chunk_id"] == "c1"


def test_the_trace_id_is_returned_so_a_run_can_be_found_in_the_logs(client):
    body = client.post("/ask", json={"query": "q"}).json()

    assert body["trace_id"] == "trace-1"


def test_an_empty_query_is_rejected_before_it_reaches_the_mesh(client):
    """A blank question costs a classifier call and returns nothing useful."""
    assert client.post("/ask", json={"query": ""}).status_code == 422


def test_an_oversized_query_is_rejected(client):
    assert client.post("/ask", json={"query": "x" * 5000}).status_code == 422


def test_a_thread_id_is_passed_through_to_the_checkpointer():
    """Without this the checkpointer has nothing to key on and every turn is a
    new conversation."""
    mesh = StubMesh(ANSWERED)

    TestClient(create_app(mesh)).post("/ask", json={"query": "q", "thread_id": "abc"})

    assert mesh.configs == [{"configurable": {"thread_id": "abc"}}]


def test_no_thread_id_means_no_config_rather_than_an_empty_one():
    """LangGraph treats a config with a null thread_id as an error, not as
    "unthreaded", so the absent case has to pass None."""
    mesh = StubMesh(ANSWERED)

    TestClient(create_app(mesh)).post("/ask", json={"query": "q"})

    assert mesh.configs == [None]


def test_the_stream_emits_one_event_per_node_then_the_answer(client):
    with client.stream("POST", "/ask/stream", json={"query": "q"}) as response:
        body = "".join(response.iter_text())

    assert body.count("event: node") == 4
    assert "event: answer" in body
    assert "Thiazides are first-line." in body
