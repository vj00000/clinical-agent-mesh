# Clinical Agent Mesh

A hierarchical multi-agent clinical assistant. A structured-output **supervisor** routes each query
to one of four **isolated LangGraph specialist subgraphs**, over grounded public clinical corpora,
behind input and output guardrail nodes.

```
__start__ → guard_in → supervisor ─┬→ guideline_sg   ─┐
                                   ├→ triage_sg      ─┤
                                   ├→ prior_auth_sg  ─┼→ guard_out → __end__
                                   ├→ discharge_sg   ─┤
                                   └→ refuse ────────┘
```

## Read this first

- **No real patient data.** Every corpus is public-domain or open-access (CDC, WHO, MedlinePlus,
  PubMed, openFDA, CMS coverage determinations). Patient notes are LLM-generated synthetic
  documents. MIMIC is deliberately **not** used — it requires credentialed access.
- **This is not a medical device** and produces no clinical advice. It is an engineering
  portfolio project demonstrating retrieval grounding, agent routing, and evaluation.
- **Depth is uneven by design.** The guideline copilot is built to production depth with a full
  eval suite. The other three specialists are demo-depth. The metrics table below states which
  is which rather than implying uniform rigour.

## Status

Under construction. See `docs/superpowers/specs/2026-08-12-clinical-agent-mesh-design.md`
for the full approved design.

| Component | Depth | State |
|---|---|---|
| Shared spine (models, state contract) | production | in progress |
| Retrieval (hybrid BM25 + vector, rerank) | production | pending |
| Supervisor + routing benchmark | production | pending |
| Guideline copilot | production | pending |
| Triage / prior-auth / discharge | demo | pending |

## Quick start

```bash
uv sync --extra postgres --extra observability   # install
cp .env.example .env                             # add your OPENAI_API_KEY
make up                                          # chroma + postgres + api
make ingest                                      # build the corpora indexes
make eval                                        # print the metrics table
```

Optional extras are separated on purpose: `--extra rerank` pulls `sentence-transformers`
(and torch, ~2GB), so CI and unit tests do not pay for it.

## Design decisions worth defending

| Decision | Why |
|---|---|
| Specialist **subgraphs**, not handler nodes | Each agent owns private state and is testable in isolation; adding a fifth agent touches no existing one |
| **Hybrid** BM25 + vector retrieval with RRF | Drug names and clinical codes are exactly where pure dense embeddings underperform |
| Guardrails as **graph nodes** | Safety is a step with its own tests, not a paragraph appended to a prompt |
| **Confidence-gated routing** | Below threshold the supervisor asks a clarifying question instead of guessing a route |
| Refuse when retrieval is **unavailable** | If Chroma is down the system says so rather than answering ungrounded |
| Regression-**gated** evals in CI | A faithfulness or routing-accuracy drop fails the build |

## Development

```bash
uv run pytest              # fast unit + node + contract tests, no LLM calls
uv run ruff check .
uv run mypy src
uv run pytest -m eval      # slow, CI-gated evaluation suite
```

Tests marked `llm` perform real API calls and are excluded from the default run.

## Licence

Code MIT. Corpora retain their original public-domain / open-access terms; see
`corpora/SOURCES.md` after running `make ingest`.
