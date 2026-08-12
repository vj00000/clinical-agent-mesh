# Clinical Agent Mesh — Design & Implementation Plan

## Context

Sawai is building a portfolio project to prove **senior/lead-level** AI engineering ability,
showcasing LangGraph, LangChain, and production AI engineering. The trigger was reviewing the
IITR course repo (`shivam13juna/v3_language_model_iitr`), where `lec_26` contains a
"Chatbot Healthcare" whiteboard PDF. That PDF turned out to be a simple LangGraph conditional
router (`classify → {handle_complaint, handle_faq, handle_general, handle_order} → END`) with a
non-healthcare NovaMart retail example behind it.

The gap this project fills: the course teaches the pattern at demo depth (single graph,
`InMemorySaver`, dense-only retrieval, Flask, no evals). A resume piece aimed at senior roles has
to go further — isolated specialist subgraphs, hybrid retrieval, guardrails as graph nodes, and a
regression-gated evaluation harness. The intended outcome is a public GitHub repo a hiring manager
can clone, run via `docker compose up`, and read measured numbers from.

**Constraints fixed with the user:**
- Time budget: 3-4 weekends (~40-60h)
- LLM runtime: OpenAI API, paid key
- Architecture: hierarchical supervisor + specialist subgraphs (chosen over flat router and swarm)
- Depth: guideline copilot built deep; the other three specialists at demo depth, disclosed honestly
- No real patient data — public corpora plus LLM-generated synthetic notes

---

## Architecture

```
__start__ → guard_in → supervisor ─┬→ guideline_sg   ─┐
                                   ├→ triage_sg      ─┤
                                   ├→ prior_auth_sg  ─┼→ guard_out → __end__
                                   ├→ discharge_sg   ─┤
                                   └→ refuse ────────┘
```

- **`guard_in`** — PHI redaction, prompt-injection detection, scope check
- **`supervisor`** — `with_structured_output` returning `{route, confidence, rationale}`;
  confidence below threshold routes to `clarify`, not a guess
- **Four specialist subgraphs** — each compiled with its own private state
- **`guard_out`** — every claim must map to a retrieved chunk id, else revise or refuse

**Parent/child state contract** (the parent never sees specialist internals):

```python
class MeshState(TypedDict):
    query: str
    route: str
    confidence: float
    answer: str
    citations: list[Citation]
    trace_id: str
```

### Deep agent — guideline copilot subgraph

```
plan_query → retrieve → rerank → draft → verify_citations ─┬→ revise (loop, max 2)
                                                            └→ contradiction_check → answer
```

Falls through to an explicit "insufficient evidence" response when top-k scores sit below
threshold. Contradiction check surfaces disagreement *between* sources rather than silently
picking one.

## Shared spine (built once, reused by all four agents)

| Module | Responsibility |
|---|---|
| `retrieval/` | Ingest → chunk → embed → Chroma; hybrid BM25 + vector fusion via reciprocal rank fusion; rerank top-20 → top-5 with a local cross-encoder (`bge-reranker-base`) to keep rerank cost off the API bill |
| `guardrails/` | `guard_in` / `guard_out` node implementations, PHI patterns, injection heuristics |
| `observability/` | Langfuse tracing; token, cost, and latency per node |
| `evals/` | Golden sets, metrics, routing benchmark, red-team suite, CI gate |
| `models/` | Provider config behind one interface (OpenAI now, swappable later) |

### Deliberate deviations from the course (each defensible in interview)

1. **FastAPI + SSE streaming** instead of `lec_26`'s Flask + gunicorn — async, token streaming
2. **Postgres checkpointer** instead of `InMemorySaver` — conversations survive restarts
3. **Hybrid retrieval** instead of dense-only — drug names and clinical codes are exactly where
   pure embeddings underperform

## Data sources

| Agent | Source | Access |
|---|---|---|
| Guideline copilot | CDC + WHO guidelines, MedlinePlus (public domain), PubMed abstracts | E-utilities API, free |
| Discharge / med-rec | openFDA drug labels + interaction data | `api.fda.gov`, free, no key at low volume |
| Prior-auth | CMS Medicare coverage determinations (LCD/NCD) — real, public domain | Bulk download |
| Triage | MedlinePlus symptom pages + a hand-written red-flag rule table | Public domain |

Patient notes are **LLM-generated synthetic**. MIMIC is explicitly excluded (credentialing
required). The README states this prominently.

## Evaluation

- **Guideline golden set** — ~60 hand-written Q&A from the corpus, including ~15 deliberately
  unanswerable, to measure calibrated refusal
- **Metrics** — faithfulness, citation accuracy (does the cited chunk actually support the claim),
  context recall, refusal correctness
- **Routing eval** — ~100 labelled queries → route accuracy plus a confusion matrix
- **Red team** — prompt injection planted inside retrieved documents, PHI leakage attempts,
  jailbreaks toward unsafe dosing advice
- **Cost/latency benchmark** — p50/p95 per route, cost per query, with and without prompt caching
- **CI gate** — build fails if faithfulness or routing accuracy falls below threshold

## Reliability

Per-node timeouts; exponential-backoff retries; structured-output validation retry; fallback model
on rate-limit; loop counter capping revise at 2 passes. **If Chroma is unavailable the system
refuses rather than answering ungrounded.**

## Testing

- Fast pure-function unit tests — chunkers, guardrails, rule tables, tools (no LLM calls)
- Node tests with mocked LLM responses
- Subgraph state-contract tests
- Slow eval suite, CI-gated

## Deployment

`docker compose` with `api` + `chroma` + `postgres` + `langfuse`; `make ingest / eval / up`;
GitHub Actions running lint + unit tests on PR and evals nightly.

---

## Build order (3-4 weekends)

**Step 0 — repo setup.** `git init`, project skeleton, commit this spec to
`docs/superpowers/specs/2026-08-12-clinical-agent-mesh-design.md`.

**Weekend 1 — spine + supervisor.** `models/` provider config, `retrieval/` ingestion and hybrid
search over the guideline corpus, `MeshState` contract, supervisor node with structured output,
`refuse` and `clarify` paths. Verify: routing benchmark runs end to end.

**Weekend 2 — guideline copilot deep.** Full subgraph including the verify/revise loop and
contradiction check, FastAPI + SSE, Postgres checkpointer with `thread_id` sessions.
Verify: golden set produces real faithfulness and citation-accuracy numbers.

**Weekend 3 — guardrails + evals hardened.** `guard_in` / `guard_out` nodes, red-team suite,
Langfuse wiring, cost/latency benchmark with and without prompt caching, CI gate.

**Weekend 4 — three thin specialists + ship.** Triage (LangGraph interrupt for follow-up
questions, red-flag rules), prior-auth (Pydantic decision + deterministic criteria checker),
discharge (extraction + openFDA interaction tool + readability scoring). Compose, README with
architecture diagram, measured metrics table, and honest per-agent depth disclosure.

## Verification

- `make up` brings up all four services; `curl` a query per route returns a cited answer
- `make eval` prints the metrics table; CI reproduces it and fails on regression
- Kill the Chroma container mid-session → system refuses instead of hallucinating
- Injection string planted in a corpus document → `guard_out` blocks the response
- Langfuse UI shows a per-node trace with token cost for a single query

## Resume bullets (fill brackets with measured numbers)

> **Clinical Agent Mesh** — LangGraph · LangChain · FastAPI · Chroma · Docker
> - Built a hierarchical multi-agent clinical assistant: a structured-output supervisor routes
>   queries to four isolated LangGraph subgraphs, achieving **[X]%** routing accuracy across a
>   100-query labelled benchmark.
> - Cut ungrounded claims from **[X]%** to **[Y]%** by adding a citation-verification node that
>   forces a revise loop when an assertion has no supporting retrieved chunk.
> - Raised retrieval recall **[X]%** over dense-only search via hybrid BM25 + vector fusion with
>   reranking, targeting drug names and clinical codes where embeddings underperform.
> - Shipped a regression-gated eval harness (faithfulness, citation accuracy, calibrated refusal,
>   adversarial red-team) wired into CI, plus Langfuse tracing giving per-query cost and p95
>   latency of **[X]ms**.
