# Build Brief — Guideline Copilot Subgraph

This is your build. The tests are written and failing; make them pass.

I am deliberately not writing the implementation. Read the tests first — they encode
the behaviour, and the reasoning is in their docstrings.

---

## What you are building

`src/mesh/agents/guideline.py` — a compiled LangGraph subgraph that the mesh's
`guideline` route hands off to.

```
plan_query → retrieve → rerank → draft → verify_citations ─┬→ revise → verify_citations
                                                            └→ contradiction_check → END
```

Every dependency is **injected** — the retriever, the reranker, the chunk store, and
the model. That is not ceremony: it is why you can test the whole subgraph without an
API key, and it is the pattern the rest of this repo already follows
(`build_mesh`, `make_supervisor`, `Reranker`).

## Node by node

| Node | Input | Output | Notes |
|---|---|---|---|
| `plan_query` | the query | 1-3 sub-questions | Multi-part clinical questions retrieve badly as one string. One sub-question for a simple query — do not invent complexity |
| `retrieve` | sub-questions | chunk ids | Union across sub-questions, deduplicated. Propagates `RetrievalUnavailable` — do **not** catch it |
| `rerank` | chunk ids | top 5 chunks | Rehydrate via `ChromaDense.fetch`, then `Reranker.rerank`. Fetch preserves fused order; rely on it |
| `draft` | query + chunks | answer + citations | The model must cite `chunk_id`s from what it was given |
| `verify_citations` | answer + citations | verdict | Reuse `mesh.guardrails.citations.verify_citations`. Do not write a second one |
| `revise` | answer + failed verdict | new answer | Bounded: max 2 attempts, then give up and refuse |
| `contradiction_check` | answer + chunks | annotated answer | If sources disagree, say so rather than silently picking one |

## State

The subgraph has **private** state. The mesh parent must not see `sub_questions` or
`revision_count` — that is the whole point of subgraph isolation.

```python
class GuidelineState(TypedDict):
    query: str
    sub_questions: list[str]
    retrieved_ids: list[str]
    chunks: list[Chunk]
    answer: str
    citations: list[Citation]
    revision_count: int
    unsupported: list[str]
```

Return to the parent only: `answer`, `citations`, `retrieved_ids`.

## The three decisions to get right

**1. The revise loop must be bounded.** Two attempts, then refuse. An unbounded
loop is how an agent burns $40 on one query. There is a test for this and it is the
one most likely to catch you out.

**2. `RetrievalUnavailable` must propagate.** Do not wrap `retrieve` in a
`try/except`. The system's contract is that it refuses rather than answering from a
degraded corpus — swallowing this here would silently break that guarantee two
layers down.

**3. `verify_citations` is already written.** Import it. If you find yourself
writing citation-checking logic, stop — you are duplicating
`src/mesh/guardrails/citations.py`, and the duplicate will drift.

## How to work

```bash
uv run pytest tests/nodes/test_guideline.py -x    # -x stops at the first failure
```

Take them one at a time, top to bottom — they are ordered so each builds on the last.
Watch each fail, write the minimum to pass it, then move on. Do not write the whole
file and then run the tests; you will lose the thread on which behaviour drove which
line.

When all node tests pass, the subgraph test at the bottom wires them together.

## When you are stuck

Ask me to explain a test, not to write the code. "Why does
`test_revision_stops_after_two_attempts` expect a refusal rather than the best
attempt so far?" is a much more useful question than "make it pass" — and the answer
is the kind of thing an interviewer asks.

## Afterwards

Wire it into the mesh by passing it as the `guideline` specialist to `build_mesh`,
run `make ingest`, and ask it a real question. Then the eval harness becomes
possible, and that is where the numbers for your resume come from.
