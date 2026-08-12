# Clinical Agent Mesh — Decision Record

Captured 2026-08-12. Everything here came out of the design conversation and the
first implementation session; none of it is recoverable from the code alone.

---

## 1. Origin

The starting question was whether the IITR course repo
`github.com/shivam13juna/v3_language_model_iitr` contained a healthcare project.

**Finding: no.** `list_of_projects.md` lists five projects, none clinical:

1. Support-ticket resolution copilot
2. Multi-document research assistant for PDFs
3. Bug triage and incident assistant
4. Sales-call and CRM copilot
5. Personal productivity agent

**But** `lec_26_langchain_langraph_deployment/` contains `Copy of Chatbot Healthcare.pdf`
(preserved in `source-material/`). Reading it revealed it is **not** healthcare content —
it is a 3-page whiteboard session:

- p1: title slide, Calvin & Hobbes filler
- p2: hand-drawn `msg → Classifier → {complaints, FAQ, General Comm, order} → END`
- p3: rendered LangGraph `__start__ → classify → {handle_complaint, handle_faq,
  handle_general, handle_order} → __end__`, plus notes on BytePair Encoding

The backing notebook uses a **NovaMart retail** example, not clinical data. "Healthcare"
was only the session title. So the pattern taught is a **LangGraph conditional router**,
and this project is that pattern taken to production depth in a genuinely clinical domain.

## 2. What the course actually teaches (the baseline we build past)

| Lecture | Content | Our upgrade |
|---|---|---|
| lec_19 | FAISS, `embeddinggemma` via Ollama, committed `faiss_index.bin` | Chroma + hybrid BM25 |
| lec_20 | ChromaDB embedded, Ollama demos | Chroma as a networked service |
| lec_21 | Langfuse, deepteam red-teaming | Full eval harness, CI-gated |
| lec_23 | Prompt caching and batching | Cost-per-query benchmark |
| lec_26 | LangGraph `StateGraph`, `TypedDict`, `InMemorySaver`, `create_agent`, `@tool`, Flask, Docker | Subgraphs, Postgres checkpointer, FastAPI |

**No Pinecone anywhere in the course** — it never introduces a managed cloud vector DB.

## 3. Fixed constraints (decided with the user)

| Decision | Choice | Reasoning |
|---|---|---|
| Target audience | Senior / lead-level proof | Differentiator is engineering rigour, not a working demo |
| Domain | All four clinical agents, multi-agent system | User explicitly wanted one agent per problem area |
| Time budget | 3-4 weekends (~40-60h) | Drives "one deep agent, three at demo depth" |
| LLM runtime | OpenAI API, paid key | User has a key; enables a real cost story |
| Deep agent | Clinical guideline copilot | Richest measurable metrics: citation faithfulness, calibrated refusal |

## 4. Architecture: why supervisor + subgraphs

Three options were weighed:

- **A. Hierarchical supervisor + subgraphs — CHOSEN.** Each specialist owns private state,
  is independently testable, and can be built on its own weekend. Degrades gracefully: if
  time runs out, three shallow agents still work.
- **B. Flat router** (the lecture pattern scaled up). Rejected: handlers get fat, one shared
  state blob for four unrelated jobs, and multi-step specialists like triage don't fit a
  single node. Reads as a course exercise.
- **C. Swarm / peer handoff.** Rejected: hard to evaluate, hard to bound, prone to handoff
  loops. Too costly to debug in 40-60h and hard to demo reliably.

## 5. Deliberate deviations from the course (each defensible in interview)

1. **FastAPI + SSE** instead of Flask + gunicorn — async and token streaming.
2. **Postgres checkpointer** instead of `InMemorySaver` — conversations survive restarts.
3. **Hybrid retrieval** instead of dense-only — drug names and ICD codes are exactly where
   pure embeddings fail.

## 6. Implementation decisions made while building

| Decision | Why it was made |
|---|---|
| **Content-addressed chunk ids** (sha256 of source+text) | A citation stored in a checkpoint or eval result must still resolve after the corpus is rebuilt. A counter-based id breaks on re-ingest. |
| **Fail closed on retrieval outage** | `RetrievalUnavailable` is raised rather than falling back to keyword-only. Answering from a degraded corpus is how a grounded system quietly starts hallucinating. This is the strongest interview story in the project. |
| **BM25 drops zero-scoring chunks** | A chunk sharing no terms with the query is noise; passing it to fusion dilutes the dense results. |
| **Reciprocal rank fusion, not score addition** | Dense similarity and BM25 produce scores on incomparable scales. |
| **Embedder injected into the vector store** | Keeps ingest-time and query-time vectors identical, and lets retrieval be tested against real Chroma with no API key — no mocking the vector store. |
| **`SecretStr` for the API key** | mypy strict rejected the plain-`str` version; `SecretStr` also stops the key leaking through a repr, log line, or traceback. |
| **Routing policy split from the LLM call** | Makes the confidence gate unit-testable without a network round trip, and auditable. |
| **Refusal honoured regardless of confidence** | An out-of-scope question does not become in-scope because the classifier hedged. |
| **rerank / postgres / observability as optional extras** | `sentence-transformers` pulls torch (~2GB). CI must not download torch to run unit tests. |
| **Local cross-encoder rerank** | Keeps reranking cost off the API bill entirely. |
| **Reranker degrades, retrieval does not** | Opposite failure handling on purpose. An unreachable vector store changes what evidence *exists*, so it raises. A failed reranker only *reorders* evidence already retrieved and grounded, so it falls back to fusion order rather than failing the query. |
| **torch pinned to the CPU wheel** | Default resolution installed `torch+cu130` plus 2.7GB of NVIDIA libraries — a 5.0GB venv for a CPU-only cross-encoder. Pinning takes it to 1.4GB. Two gotchas: `[tool.uv.sources]` applies only to *direct* dependencies (so torch had to be declared explicitly even though sentence-transformers pulls it), and `uv sync` reuses the lockfile — `uv lock` must regenerate it first. |
| **Triage rules deterministic, escalate-only** | A missed emergency must not depend on model variance or a prompt regression, so the rules run alongside the LLM and never downgrade. Tuned for recall: a pulled muscle sent to urgent care costs an afternoon, a missed MI costs a life. |

### The BM25 small-corpus trap (worth remembering)

`BM25Okapi`'s IDF is `log(N - df + 0.5) - log(df + 0.5)`. At **N=2, df=1 this is exactly
zero**, so every score collapses and the `score > 0` filter returns nothing. Two hybrid tests
failed on a two-document fixture because of this.

`BM25Plus` was evaluated as a fix and **rejected**: it assigns non-matching documents a
nonzero score (a `metformin` doc scored 1.09 on a `lisinopril` query), which would poison
fusion. The fixture was enlarged to four documents instead — the degeneracy is a small-corpus
artifact, not a property of any real corpus. Assertions were not weakened.

## 7. Data policy

Every corpus is public-domain or open-access; patient notes are LLM-generated synthetic.
**MIMIC is deliberately excluded** because it requires credentialed access. This judgement is
stated prominently in the README — it is itself a hiring signal.

| Agent | Source |
|---|---|
| Guideline copilot | CDC, WHO, MedlinePlus, PubMed (E-utilities) |
| Discharge / med-rec | openFDA drug labels |
| Prior-auth | CMS Medicare coverage determinations (LCD/NCD) |
| Triage | MedlinePlus symptom pages + hand-written red-flag table |

## 8. Environment findings (this machine)

- **PyPI is reachable; the npm registry is not.** Python work is unblocked on this box.
- **No Python package manager existed** — no pip, no pipx, no venv support, no conda.
  `ensurepip` is missing, so `python3 -m venv` fails without `apt install python3-venv`.
  **Resolved by installing `uv` 0.12.3** to `~/.local/bin` (no sudo). `~/.local/bin` is
  already on PATH via both `.bashrc` and `.profile`.
- **Docker 29.1.3 + Compose v5.1.1**, daemon accessible without sudo.
- **Docker Hub is reachable but slow** — ~8s just for a registry handshake; the Chroma image
  pull exceeded a 5-minute foreground timeout and had to run in the background.
- **LangChain resolved to 1.3.15**, a major version beyond the course's 0.3.x. `create_agent`
  now lives in `langchain.agents`; lecture snippets will not run verbatim.

## 9. Status at time of backup

**Built and verified — 37 tests passing, ruff clean, mypy --strict clean, 3 commits.**

- Shared spine: `Settings` (SecretStr), provider factory, `MeshState` / `Citation` contract
- Retrieval: chunking, reciprocal rank fusion, BM25, hybrid orchestrator, Chroma dense adapter
- Supervisor routing policy (confidence gate)
- Docker Compose (Chroma + Postgres), Makefile

**Not yet built:** ingestion pipeline, supervisor LLM node, guideline subgraph, guardrail
nodes, eval harness, three thin specialists, FastAPI layer, CI.

**Next blocker:** ingestion needs `OPENAI_API_KEY` in `.env` (`cp .env.example .env`).
Everything downstream depends on it.

## 10. Standing rule on metrics

Every bracketed number in the resume bullets must come from an actual `make eval` run.
Fabricated metrics are the fastest way to lose a senior interview, because the obvious
follow-up is "walk me through how you measured that."
