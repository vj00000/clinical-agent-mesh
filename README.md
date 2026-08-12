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
| Cross-encoder rerank | production | done (CPU-pinned) |
| Supervisor node + routing benchmark | production | done (33-case benchmark; target 100) |
| Guideline copilot | production | pending |
| Eval harness | production | pending |
| Triage red-flag rules | demo | done |
| Prior-auth / discharge specialists | demo | pending |

149 fast tests, plus 3 rerank and 4 network tests. `ruff` and `mypy --strict` clean.

## Quick start

```bash
uv sync --extra postgres --extra observability   # install
cp .env.example .env                             # add your OPENAI_API_KEY
make up                                          # chroma + postgres + api
make ingest                                      # build the corpora indexes
make eval                                        # print the metrics table
```

Optional extras are separated on purpose: `--extra rerank` pulls `sentence-transformers`
and torch, so CI and the default test run do not pay for it.

torch is pinned to the **CPU wheel** via `[tool.uv.sources]`. The default resolution
installed `torch+cu130` and 2.7GB of NVIDIA CUDA libraries — a 5.0GB virtualenv to run
a small cross-encoder that is CPU-only by design. Pinning the CPU build takes it to
1.4GB and the rerank tests run roughly twice as fast.

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
make check                 # lint + strict types + 149 tests, no LLM calls, no network
make test-network          # tests against the live PubMed and MedlinePlus APIs
make test-rerank           # tests the real cross-encoder (needs --extra rerank)
make eval-routing          # score the labelled routing benchmark (one LLM call per case)
```

`llm`, `network`, and `rerank` tests are excluded from the default run: the first costs
money, the second depends on NCBI rate limits, and the third spends ~30s importing torch
against ~7s for everything else.

## Documentation

| Document | What it covers |
|---|---|
| `docs/superpowers/specs/` | The approved design spec |
| `docs/DECISIONS.md` | Every design decision, the alternatives rejected, and environment findings |
| `docs/INTERVIEW-GUIDE.md` | How to discuss the project, the bugs found and how, and its real limits |
| `docs/RESUME-BULLETS.md` | Bullet inventory tagged built / pending / needs-measurement |

## Licence

Code MIT. Corpora retain their original public-domain / open-access terms; see
`corpora/SOURCES.md` after running `make ingest`.
