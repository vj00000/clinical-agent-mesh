# Resume Bullets — Clinical Agent Mesh

Tags: **[BUILT]** true today · **[PENDING]** needs the code · **[MEASURE]** needs a real
number from `make eval`.

**Rule: never fill a bracket with an estimate.** The follow-up question in any senior
interview is "how did you measure that."

---

## The one to actually put on the resume (5 bullets)

> **Clinical Agent Mesh** — LangGraph · LangChain · Chroma · FastAPI · Docker
>
> - Architected a hierarchical multi-agent clinical assistant in LangGraph: a
>   structured-output supervisor routes queries to four isolated specialist subgraphs behind
>   PHI-redaction and citation-verification guardrail nodes, hitting **[X]%** routing accuracy
>   on a 100-query labelled benchmark.
> - Built hybrid retrieval fusing BM25 with dense vectors by reciprocal rank, improving recall
>   **[X]%** over a dense-only baseline on the tokens clinical text depends on — drug names,
>   dosages, and ICD codes.
> - Cut ungrounded claims from **[X]%** to **[Y]%** with a citation-verification node that
>   rejects any assertion untraceable to a retrieved chunk and triggers a bounded revise loop
>   before refusing.
> - Shipped a regression-gated eval harness (faithfulness, citation accuracy, calibrated
>   refusal, adversarial red-team) wired into CI so a quality drop fails the build.
> - Engineered for production: FastAPI with SSE streaming, Postgres-checkpointed sessions,
>   fail-closed retrieval, and per-node cost tracing giving **[X]ms** p95 at **[$Y]** per
>   1,000 queries.

## Three-line version

> - Architected a LangGraph multi-agent clinical assistant: structured-output supervisor
>   routing to four isolated specialist subgraphs behind PHI-redaction and
>   citation-verification guardrail nodes.
> - Built hybrid BM25 + dense retrieval with reciprocal rank fusion over Chroma, chosen
>   because embedding-only search loses on drug names and clinical codes.
> - Shipped test-first with mypy strict and a regression-gated eval harness measuring
>   faithfulness, citation accuracy, and calibrated refusal.

## Honest version usable before the project is finished

> - Architected a hierarchical multi-agent clinical assistant in LangGraph: a
>   structured-output supervisor routes queries to four isolated specialist subgraphs behind
>   input/output guardrail nodes, keeping specialist state private so a new agent ships
>   without touching existing ones.
> - Built hybrid retrieval — BM25 fused with dense vectors by reciprocal rank — after
>   identifying that pure embedding search degrades on the exact tokens clinical text depends
>   on: drug names, dosages, and ICD codes.
> - Designed the system to fail closed: an unreachable vector store raises rather than
>   silently degrading to keyword-only results, and content-addressed chunk ids keep stored
>   citations valid across full corpus rebuilds.
> - Separated routing policy from the LLM call so the confidence gate is unit-testable without
>   network calls; sub-threshold classifications ask a clarifying question instead of guessing.
> - Developed test-first under `mypy --strict`, with dense retrieval verified against a real
>   containerized Chroma instance using an injected embedder rather than mocks.

---

# Full inventory (the pool to draw from)

## 1. Multi-agent orchestration

- **[BUILT]** Architected a hierarchical multi-agent clinical assistant in LangGraph:
  structured-output supervisor routing to four isolated specialist subgraphs behind
  input/output guardrail nodes.
- **[BUILT]** Enforced a narrow parent/child state contract so specialist internals stay
  private — adding a fifth agent touches no existing subgraph.
- **[BUILT]** Separated routing policy from the LLM call, making the confidence gate
  unit-testable without network calls.
- **[PENDING]** Implemented multi-intent handling: queries spanning two specialists fan out
  to both and merge under a single citation set.
- **[PENDING]** Used LangGraph `interrupt` for human-in-the-loop follow-up questions in
  triage, pausing the graph mid-execution and resuming from a checkpoint.
- **[MEASURE]** Achieved **[X]%** routing accuracy across a 100-query labelled benchmark; the
  confusion matrix identified **[pair]** as the dominant misroute and drove a prompt revision
  recovering **[Y]** points.

## 2. Retrieval / RAG

- **[BUILT]** Built hybrid retrieval fusing BM25 with dense vectors by reciprocal rank.
- **[BUILT]** Made chunk ids content-addressed so rebuilding the corpus never invalidates
  citations already stored in checkpoints or eval results.
- **[BUILT]** Injected the embedding function into the vector store adapter, keeping
  ingest-time and query-time vectors identical and enabling tests without an API key.
- **[PENDING]** Added a local cross-encoder rerank stage (top-20 → top-5), keeping rerank
  cost off the API bill entirely.
- **[PENDING]** Implemented query decomposition so multi-part clinical questions retrieve per
  sub-question before synthesis.
- **[MEASURE]** Improved retrieval recall **[X]%** over a dense-only baseline on a
  60-question golden set.

## 3. Guardrails and safety

- **[BUILT]** Designed the system to fail closed: unreachable vector store raises rather than
  degrading to keyword-only results.
- **[PENDING]** Implemented PHI redaction and prompt-injection detection as a dedicated graph
  node — safety as a step with its own tests, not a prompt suffix.
- **[PENDING]** Built a citation-verification node rejecting any claim not traceable to a
  retrieved chunk, with a bounded revise loop (max 2 passes) before refusal.
- **[PENDING]** Added contradiction detection surfacing disagreement *between* guideline
  sources instead of silently selecting one.
- **[PENDING]** Implemented calibrated refusal: retrieval scores below threshold produce an
  explicit "insufficient evidence" response.
- **[MEASURE]** Reduced ungrounded claims from **[X]%** to **[Y]%** via citation verification.
- **[MEASURE]** Defended against **[N]** adversarial prompts including injections planted
  inside retrieved documents; **[X]/[N]** blocked at the guardrail node.

## 4. Evaluation engineering

- **[PENDING]** Built a regression-gated eval harness measuring faithfulness, citation
  accuracy, context recall, and refusal correctness, wired into CI so a quality drop fails
  the build.
- **[PENDING]** Authored a 60-question golden set including 15 deliberately unanswerable
  questions, making calibrated refusal a measured property rather than an aspiration.
- **[PENDING]** Built a red-team suite covering prompt injection, PHI leakage, and jailbreaks
  toward unsafe clinical advice.
- **[MEASURE]** Held faithfulness at **[X]** and routing accuracy at **[Y]%** as CI gates
  across **[N]** commits.

## 5. Cost and latency

- **[PENDING]** Instrumented per-node token, cost, and latency tracing with Langfuse.
- **[MEASURE]** Cut cost per query **[X]%** via prompt caching, at p95 latency of **[Y]ms**.
- **[MEASURE]** Reduced rerank spend to zero by moving it to a local cross-encoder, saving
  **[X]** per 1,000 queries versus an API reranker.

## 6. Reliability

- **[PENDING]** Added per-node timeouts, exponential-backoff retries, structured-output
  validation retries, and a fallback model on rate-limit.
- **[PENDING]** Bounded every LLM loop with an explicit pass counter, making runaway agent
  cycles structurally impossible.
- **[PENDING]** Persisted conversation state to a Postgres checkpointer keyed by thread id.

## 7. Deployment and practice

- **[BUILT]** Developed test-first throughout under `mypy --strict`, with retrieval verified
  against a real containerized Chroma instance rather than mocks.
- **[BUILT]** Containerized the stack with Docker Compose behind a `make` interface.
- **[PENDING]** Served the graph over FastAPI with SSE token streaming.
- **[PENDING]** Set up GitHub Actions running lint, strict types, and unit tests per PR with
  the eval suite gated nightly.

## 8. Per-specialist capability

- **[PENDING]** *Guideline copilot:* grounded clinical Q&A over CDC/WHO/MedlinePlus/PubMed
  with mandatory citations and cross-source contradiction detection.
- **[PENDING]** *Triage agent:* symptom intake with red-flag escalation, optimising recall
  over precision because a missed emergency costs more than a false alarm.
- **[PENDING]** *Prior-auth assistant:* deterministic criteria checking alongside the LLM over
  real CMS coverage determinations, with an audit trail of which clause drove each decision.
- **[PENDING]** *Discharge / med-rec:* structured medication extraction with an openFDA
  interaction tool and readability-targeted patient summaries.

## 9. Data judgement

- **[BUILT]** Sourced every corpus from public-domain or open-access clinical data and
  generated synthetic patient notes, deliberately excluding credentialed datasets such as
  MIMIC.

---

## Interview prep note

The **fail-closed retrieval decision** is the strongest story here. "What happens when your
vector DB goes down" separates people who have run systems from people who have demoed them.
Have the `RetrievalUnavailable` reasoning ready: answering from a degraded corpus is how a
grounded system quietly starts hallucinating.

Second-strongest: the **BM25 small-corpus IDF trap** — it shows you read library internals
rather than trusting defaults, and that you fixed the fixture instead of weakening the
assertion.
