# Clinical Agent Mesh

A hierarchical multi-agent clinical assistant. A structured-output **supervisor** routes each query
to one of four **isolated LangGraph specialist subgraphs**, over grounded public clinical corpora,
behind input and output guardrail nodes.

Rendered from the compiled graph (`mesh.get_graph().draw_mermaid()`), so it cannot
drift from the code:

```mermaid
graph TD;
	__start__([__start__]):::first
	guard_in(guard_in)
	supervisor(supervisor)
	refuse(refuse)
	clarify(clarify)
	guard_out(guard_out)
	guideline(guideline)
	triage(triage)
	prior_auth(prior_auth)
	discharge(discharge)
	__end__([__end__]):::last
	__start__ --> guard_in;
	guard_in -.-> refuse;
	guard_in -.-> supervisor;
	supervisor -.-> guideline;
	supervisor -.-> triage;
	supervisor -.-> prior_auth;
	supervisor -.-> discharge;
	supervisor -.-> clarify;
	supervisor -.-> refuse;
	guideline --> guard_out;
	triage --> guard_out;
	prior_auth --> guard_out;
	discharge --> guard_out;
	clarify --> guard_out;
	refuse --> guard_out;
	guard_out --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```

Note the edge from `guard_in` straight to `refuse`: a blocked query never reaches
the supervisor, so the classifier never sees an injection payload.

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
| Shared spine (models, state contract) | production | done |
| Retrieval (hybrid BM25 + vector) | production | done |
| Ingestion (PubMed + MedlinePlus) | production | done |
| Guardrails (PHI, injection, citations) | production | done |
| Mesh graph wiring | production | done |
| Cross-encoder rerank | production | pending |
| Supervisor + routing benchmark | production | routing policy done, LLM node pending |
| Guideline copilot | production | pending |
| Eval harness | production | pending |
| Triage / prior-auth / discharge | demo | pending |

108 tests passing, `ruff` and `mypy --strict` clean.

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
