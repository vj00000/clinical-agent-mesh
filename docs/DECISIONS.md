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

## 9. Status at completion (2026-09-10)

**Complete and verified: 273 fast tests, 18 integration, 8 network. ruff, ruff
format and `mypy --strict` clean.**

Built: shared spine, retrieval (hybrid BM25 + Chroma + cross-encoder rerank),
ingestion over four corpora, guardrails as nodes, mesh wiring, supervisor,
four specialist subgraphs, the composition root, the eval harness and red-team
suite, FastAPI + SSE, Dockerfile, compose, GitHub Actions.

**Never executed:** the six model-backed prompt factories, and therefore
`make eval`. There was no API key on the build machine. No metric in this repo
is a measured metric yet. See `HANDOVER.md` §6.

## 10. Standing rule on metrics

Every bracketed number in the resume bullets must come from an actual `make eval` run.
Fabricated metrics are the fastest way to lose a senior interview, because the obvious
follow-up is "walk me through how you measured that."


---

# Decisions made during the build (2026-08-13 → 2026-09-10)

Everything above was decided before implementation. These came out of building
it, and each one changed the code.

## 11. The guideline revise loop refuses; the discharge one does not

Both loops are bounded. They end differently, and the difference is the point.

When the guideline drafter exhausts its two revisions still citing chunks that
were never retrieved, the subgraph returns `UNGROUNDED_REFUSAL` rather than the
best attempt so far — every attempt cited something invented, so there is no
grounded answer to fall back to. When the discharge rewriter fails to get the
reading grade down, it returns the dense text anyway. Prose a grade too hard is
still usable; an ungrounded clinical claim never is.

## 12. `guard_out` exempts deterministic safety escalations

`guard_out` refuses any answer whose citations do not map to retrieved chunks.
That is right for clinical claims and wrong for a red-flag escalation, which is
a rule firing rather than a claim derived from evidence. Without an exemption, a
Chroma outage silently converts "call an ambulance" into "I don't have the
evidence" — the worst output this system can produce.

The exemption is a `guard_flags` entry (`safety_escalation`) rather than a new
route, so the supervisor's routing vocabulary is untouched.

## 13. The model never decides what code can decide

Three instances, each with tests that pin it:

- **Triage.** `assess_urgency` produces a floor. `apply_urgency_floor` takes the
  higher of the rules' reading and the model's. A prompt regression cannot
  reassure someone into staying home.
- **Prior auth.** The model marks each criterion met / not met / unresolved.
  `decide_coverage` computes approve / deny / more-info, and `render_decision`
  writes the prose, so the wording cannot contradict the verdict. `met=None`
  means the request is *silent*, not that it failed — collapsing those two
  turns a missing form into a denial.
- **Discharge.** Reading grade is Flesch-Kincaid arithmetic. Interaction
  warnings are appended in code, because a tool that finds an interaction the
  model then forgets to mention is worse than no tool.

## 14. The triage LangGraph `interrupt` was deferred

The spec called for an `interrupt` for the follow-up question. A genuine
interrupt needs a checkpointer and `thread_id` sessions, which did not exist
when triage was built. The follow-up is returned as a clarifying question
instead, and the next turn carries the detail. The checkpointer now exists, so
this is a live piece of work rather than a permanent decision.

## 15. Interaction warnings carry provenance

The discharge `Interaction` model has `chunk_id` and `source`, and those ids
join `retrieved_ids`. Without that, `guard_out` would see warnings that no
retrieved chunk supports and refuse the whole answer. A tool result is evidence
too, and evidence needs provenance.

This forced `parse_openfda_interaction_notes`, which returns notes paired with
their label id; `parse_openfda_interactions` is now a bare-text view over it.

## 16. Interaction pairing is deterministic, not a model call

A drug label's interaction section warns about entire drug classes. A note only
becomes an `Interaction` when it names another medication the patient is
actually taking — otherwise every warfarin label warns about every drug on
earth. Both directions are checked, because only one of the two labels may
mention the other, and duplicates collapse on the pair plus the warning text.

## 17. One collection per agent, not one index

`guideline`, `triage`, `coverage`, `drug`. Three reasons: a coverage query
should not compete with drug labels for the top-20 fusion slots; BM25 term
statistics on an already-small corpus get worse when document types are mixed
(see §6's small-corpus trap); and per-agent recall is only measurable in evals
if you know which corpus an answer should have come from.

`COLLECTION_BY_ROUTE` states the mapping explicitly because route and collection
names deliberately differ — `prior_auth` reads `coverage`, `discharge` reads
`drug`. A collection is named for what it holds, an agent for what it does.

## 18. The CMS corpus is a policy index, not a rulebook

`api.coverage.cms.gov/v1/reports/*` is keyless and returns the current NCD and
final-LCD listings. The sibling `/v1/data/*` routes carry the actual coverage
criteria and require a licence token (AMA CPT, ADA CDT, AHA UB-04).

So the `coverage` corpus holds policy titles, identifiers, contractors and
dates — enough to say *which* policy governs a request, not enough to say
whether the request meets it. `prior_auth` will therefore usually answer
`more_info`.

Three alternatives were rejected: leaving the route dead (a whole specialist
that refuses everything looks like a bug); scraping the licensed text (a licence
is a licence); and writing plausible criteria by hand (fabricated clinical
policy in a portfolio project is indefensible). An honest, thin corpus with the
limitation documented beats all three.

## 19. The composition root exposed two latent bugs

Nothing in production built a mesh until it existed — `build_mesh` and
`as_specialist` had only test callers. Writing it surfaced:

- **`BM25Index` could never have been populated.** It scores against the whole
  corpus in memory, and `ChromaDense` had no way to enumerate a collection.
  Hybrid retrieval would have silently degraded to dense-only with nothing
  raising. Fixed by `ChromaDense.all_chunks()`.
- **`guideline.Retriever` had a typo**, `tok_k` for `top_k`, so `HybridRetriever`
  never satisfied the protocol. Invisible because tests pass stubs and `mypy`
  only checks `src/`; the composition root is the first code to type-check the
  protocol against the real retriever.

## 20. `integration` is excluded from the default test run

`network` and `rerank` were already excluded so `make check` stays fast. When
Chroma is running, the integration tests took the default run from ~16s to
~4min, which is long enough that people stop running it — and when Chroma is
*not* running they skip silently, which is how the whole Chroma layer went
unverified for several commits. Now `make test-integration` runs them
deliberately and CI runs them in their own job.

## 21. The red team found two real gaps on its first run

Writing the adversarial suite immediately broke the injection detector:
`disregard your rules` (the qualifier list had `previous` and `all` but not
`your`) and persona jailbreaks (`You are now DAN…`), which no pattern covered.
Both are fixed, with a `persona_override` pattern that still requires two
elements together so `act as a nurse would` does not fire.

The value was not the fix. It was that a suite written to be adversarial found
holes in a guardrail its author believed was finished.

## 22. Streaming is node-level, and says so

SSE emits one event per graph node, then the final answer. Real token streaming
needs streaming pushed down into every model call. Node-level events still make
the mesh legible from outside — you watch `guard_in → supervisor → guideline →
guard_out` arrive — but calling it token streaming would be a lie.
