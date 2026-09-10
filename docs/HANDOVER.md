# Handover — Clinical Agent Mesh

Written 2026-09-10, at the point the build was finished on the machine it was
built on. It assumes you have the repository and nothing else: no notes, no
memory of the decisions, no running services.

---

## 1. What this is

A hierarchical multi-agent clinical assistant. A structured-output supervisor
routes each query to one of four isolated LangGraph specialist subgraphs, over
grounded public clinical corpora, behind input and output guardrail nodes.

It is an engineering portfolio project. It is **not a medical device** and
produces no clinical advice.

```
__start__ → guard_in → supervisor ─┬→ guideline_sg   ─┐
                                   ├→ triage_sg      ─┤
                                   ├→ prior_auth_sg  ─┼→ guard_out → __end__
                                   ├→ discharge_sg   ─┤
                                   └→ refuse ────────┘
```

## 2. Getting it running from a clean clone

```bash
uv sync --extra postgres --extra observability   # install
cp .env.example .env                             # add OPENAI_API_KEY
make up                                          # chroma + postgres + api
make ingest                                      # build the four corpora
make ask q="what is first-line therapy for hypertension?"
```

Without a key you can still do everything except run the mesh for real:

```bash
make check              # lint, types, 273 tests -- no key, no network, no Chroma
make up && make test-integration   # 18 more against real Chroma, still no key
make test-network       # 8 against the live public APIs, still no key
```

## 3. What was built, in the order it was built

| Step | What | Why it came here |
|---|---|---|
| 1 | Spine: `models/`, `state.py`, `retrieval/` | Everything else depends on the state contract and on retrieval existing |
| 2 | Guardrails as graph nodes | Safety had to be a step with tests, not a paragraph in a prompt |
| 3 | Mesh wiring + supervisor | The router, with confidence-gated `clarify` |
| 4 | Cross-encoder rerank | Retrieval quality before agents, so agents are judged on their own logic |
| 5 | `guideline` subgraph (production depth) | The deep agent: plan → retrieve → rerank → draft → verify → revise → contradiction check |
| 6 | `as_specialist` adapter | The parent/child state boundary, written deliberately rather than by coincidence |
| 7 | `triage`, `prior_auth`, `discharge` (demo depth) | Three thin specialists proving the pattern generalises |
| 8 | Corpora: openFDA, CMS, per-agent collections | Without these, three of four routes retrieve from an empty index |
| 9 | Composition root | The first code that builds a real mesh; before it, `build_mesh` had only test callers |
| 10 | Eval harness, red team, CI gate | The numbers, and the thing that stops them silently regressing |
| 11 | FastAPI + SSE, Docker, CI | Making it a system someone else can run |

## 4. The decisions worth defending

Full record in `DECISIONS.md`. The ones an interviewer will actually probe:

**Specialist subgraphs, not handler nodes.** Each agent owns private state and
is testable in isolation. Adding a fifth touches no existing one. The cost is
the adapter in `graph.py:as_specialist`, which is where the two state shapes
meet on purpose.

**The model never decides what code can decide.** Three examples, each with
tests: triage's red-flag rules are a floor the model may raise but never lower;
prior-auth's approve/deny is computed by `decide_coverage` from marked criteria,
and the answer prose is composed in code so it cannot contradict the verdict;
discharge's reading grade is arithmetic, not a judgement.

**Refuse rather than answer ungrounded.** `guard_out` rejects any answer citing
a chunk that was not retrieved. Retrieval failure propagates rather than
degrading to keyword-only. The one deliberate exemption is a deterministic
safety escalation — without it, a Chroma outage silently converts "call an
ambulance" into "I don't have the evidence".

**Hybrid BM25 + vector with RRF.** Drug names and clinical codes are exactly
where dense embeddings underperform.

**One collection per agent.** A coverage query should not compete with drug
labels for the top-20 slots, and per-agent recall is only measurable if you know
which corpus an answer should have come from.

## 5. Honest limits

Do not let anyone discover these by surprise; say them first.

- **No measured numbers yet.** The eval harness is built, tested and gated, but
  it has never been run — there was no API key on the build machine. Every
  bracket in `RESUME-BULLETS.md` is still a bracket. See §6.
- **The prompts have never executed.** Six model-backed factories
  (`build_planner`, `build_drafter`, `build_contradiction_detector`,
  `build_adviser`, `build_criteria_reader`, `build_medication_extractor`,
  `build_instruction_drafter`) are wired and type-checked but have never seen a
  real model. Expect at least one to need adjusting on first run; the most
  likely is `PolicyReading`'s tri-state `met: bool | None`, because models
  avoid returning null.
- **The `coverage` corpus is an index, not a rulebook.** CMS publishes the
  policy list without a key, but the criteria text sits behind an AMA/CPT
  licence token. So `prior_auth` can say which policy governs a request and
  usually cannot say whether the request meets it — it answers `more_info`.
  That is the system reporting its evidence honestly, not a bug, but it does
  mean prior-auth is the weakest of the four.
- **Depth is uneven by design.** `guideline` is production depth. The other
  three are demo depth. The README says so in a table.
- **SSE streams nodes, not tokens.** Real token streaming needs streaming
  pushed down into each model call.
- **Checkpointing is durability, not memory.** Turns persist per `thread_id`;
  no specialist reads prior turns.
- **The triage `interrupt` was deferred.** A genuine LangGraph interrupt needs
  thread_id sessions; the follow-up is returned as a clarifying question
  instead.
- **Synthetic and public data only.** No MIMIC, no real patient data.

## 6. Filling in the numbers (the first thing to do with a key)

```bash
cp .env.example .env          # add OPENAI_API_KEY
make up
make ingest                   # ~cents, embeddings only
make eval                     # prints the metrics table, exits 1 below threshold
make eval-routing             # routing accuracy + confusion matrix
```

Then copy the printed table into the README's metrics section and replace the
brackets in `docs/RESUME-BULLETS.md` with what it printed. **Do not estimate
them.** A number you cannot reproduce on demand in an interview is worse than
no number.

Expect the first `make eval` to fail. That is the gate working: thresholds are
in `src/mesh/evals/harness.py:THRESHOLDS`, and the failure names the metric.

## 7. Where things live

| Path | What |
|---|---|
| `src/mesh/state.py` | The parent/child state contract |
| `src/mesh/graph.py` | Mesh wiring, `as_specialist`, refusal texts |
| `src/mesh/agents/` | Four specialists; `*_rules.py` are the deterministic halves |
| `src/mesh/agents/citing.py` | Model-reported citations → grounded ones |
| `src/mesh/guardrails/` | PHI, injection, citation verification, the two nodes |
| `src/mesh/retrieval/` | Chunking, BM25, Chroma, fusion, rerank, sources, ingest |
| `src/mesh/composition.py` | The only module that builds the real thing |
| `src/mesh/evals/` | Metrics, harness, red team, routing benchmark |
| `src/mesh/api/app.py` | HTTP surface |
| `evals/golden/` | 64 labelled mesh cases, 33 routing cases |
| `evals/redteam/` | 18 adversarial cases |

## 8. Things I would do next, in order

1. Run `make eval`, fix whatever the prompts get wrong, fill in the numbers.
2. Grow the routing benchmark from 33 to 100 cases (the spec's target).
3. Migrate `guideline.py` onto `agents/citing.py` — it still has its own private
   copy of that mapping, the only remaining duplication of it.
4. Label relevant chunk ids in the golden set so `context_recall` becomes
   computable; it is defined but unused.
5. Wire Langfuse (`src/mesh/observability/` is an empty directory and the
   dependency is declared).
6. Push token streaming down into the model calls, then make SSE honest about
   streaming tokens.
