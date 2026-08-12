# Interview Guide — Clinical Agent Mesh

How to talk about this project. Every claim here is backed by code in this repo;
nothing is aspirational. If a section says `[NOT BUILT]`, say so out loud rather
than implying otherwise — an interviewer who catches one inflated claim discounts
everything else you said.

---

## 1. The 30-second version

> It's a hierarchical multi-agent clinical assistant. A supervisor classifies the
> query and routes it to one of four isolated LangGraph subgraphs — guidelines,
> triage, prior authorisation, discharge — with guardrail nodes on the way in and
> the way out. The interesting part isn't the routing, it's what happens when
> something goes wrong: retrieval outages, invented citations, and injections
> planted inside the corpus all have defined, tested behaviour.

Then stop. Let them pick the thread.

## 2. The three questions to steer toward

These are where the project is strongest. If given an opening, go here.

### "What happens when your vector database goes down?"

**The answer:** it refuses. `HybridRetriever` raises `RetrievalUnavailable`
rather than falling back to keyword-only results.

**Why it matters:** the tempting design is graceful degradation — dense search
fails, so serve BM25 hits and carry on. But the whole promise of the system is
that answers are grounded in retrieved evidence. Serving from a silently degraded
corpus means the model is answering from a *worse* evidence base while the user
sees no difference. That is exactly how a grounded system starts hallucinating
without anyone noticing. Failing loudly is a feature.

**Where:** `src/mesh/retrieval/hybrid.py`, test
`test_an_unreachable_dense_backend_raises_rather_than_degrading_silently`.

**Follow-up they may ask — "isn't some answer better than none?"** In clinical
context, no. A confidently wrong answer about therapy is worse than "I can't
reach my sources." The same reasoning drives the supervisor: if the classifier
errors, it refuses rather than picking a specialist at random.

### "How do you stop it from hallucinating?"

Two layers, and be clear which is which.

**Deterministic (built):** `guard_out` checks that every citation points at a
chunk id the retriever actually returned this turn. A model can invent a
plausible chunk id as easily as it can invent a fact, and an invented citation is
*worse* than none because it looks like evidence. An answer with no citations at
all is also rejected — for a clinical claim, uncited is indistinguishable from
guessed.

**Model-based (`[NOT BUILT]`):** judging whether the cited chunk genuinely
*supports* the claim needs an LLM and belongs in the guideline subgraph's
verify/revise loop. Say plainly that this is designed but not yet implemented.

**Where:** `src/mesh/guardrails/citations.py`, `src/mesh/guardrails/nodes.py`.

### "How would you know if it got worse?"

Routing accuracy against a labelled benchmark with a **confusion matrix**, not
just a single number. Accuracy tells you something regressed; the confusion
matrix tells you *which two specialists* the classifier now conflates, which is
the prompt you go fix. A classifier error scores as a miss rather than aborting
the run — one bad response shouldn't cost a whole paid benchmark sweep.

**Where:** `src/mesh/evals/routing.py`, benchmark at
`evals/golden/routing_cases.json` (33 cases; the target is 100 — say the real
number).

**Honest caveat:** the faithfulness and citation-accuracy evals are designed but
`[NOT BUILT]`. No metric has been produced yet because no eval has run.

## 3. Bugs found, and how — the credibility section

Interviewers trust specific failures more than clean narratives. These are real,
and each shows a different kind of rigour.

### BM25 collapses to zero on a tiny corpus

Two hybrid retrieval tests failed. `BM25Okapi`'s IDF is
`log(N-df+0.5) - log(df+0.5)`, which at **N=2, df=1 is exactly zero** — so every
score was 0 and my "drop zero-scoring chunks" filter returned nothing.

I evaluated `BM25Plus` as a fix and **rejected it**: it assigns non-matching
documents a nonzero score (a metformin document scored 1.09 on a `lisinopril`
query), which would feed noise into rank fusion. The filter was right; the
*fixture* was pathological. I enlarged the test corpus to four documents rather
than weakening the assertion.

**The point to make:** I read the library's scoring internals instead of trusting
defaults, and I fixed the test data rather than the test's expectations.

### MedlinePlus double-encodes its HTML

`clean_text` passed its unit tests, then real payloads showed the bug. MedlinePlus
returns `&lt;p&gt;` — entity-encoded markup. My implementation stripped tags
*then* unescaped, so decoding *revealed* literal `<p>` tags that survived into the
text and would have been embedded as if they were clinical prose. Fixed by
repeating strip-then-unescape until the text stabilises, with a live-API
regression test asserting no `<` or `&lt;` survives.

**The point to make:** unit tests against invented fixtures pass happily while
being wrong about the real world. I found this by inspecting an actual response.

### The graph caught a bug the unit tests couldn't

`guard_out` demanded citations from *clarifying questions*. So when the supervisor
was unsure and asked "could you say more about what you mean?", guard_out saw an
uncited answer and replaced it with a refusal — the user would never be asked
what they meant. Both `refuse` and `clarify` are now exempt: neither asserts
anything clinical.

**The point to make:** this was invisible at the unit level. It only appeared once
the nodes were wired into a compiled graph, which is an argument for integration
tests that exercise real wiring.

### mypy strict caught a leaking secret

The API key was typed as a plain `str`. mypy rejected it against the provider
signature, and fixing it properly meant `SecretStr` — which also stops the key
appearing in a repr, log line, or traceback.

## 4. Design decisions and the alternatives rejected

| Decision | Alternative rejected | Reasoning |
|---|---|---|
| Supervisor + isolated subgraphs | Flat router (one graph, four handler nodes) | Handlers get fat, one state blob serves four unrelated jobs, and multi-step specialists like triage don't fit in a single node |
| Supervisor + isolated subgraphs | Swarm / peer handoff | Hard to evaluate, hard to bound, prone to handoff loops — too costly to debug and hard to demo reliably |
| Hybrid BM25 + dense | Dense only (as the course taught) | Embeddings blur exact tokens; drug names, dosages, and ICD codes are where vector-only search loses |
| Reciprocal rank fusion | Adding the two scores | Dense similarity and BM25 scores are on incomparable scales |
| Content-addressed chunk ids | Sequential counter ids | A citation stored in a checkpoint or eval result must still resolve after the corpus is rebuilt |
| Guardrails as graph nodes | Safety instructions in the prompt | A node has its own tests and its decisions land in the audit trail; prompt text can be argued away by the model |
| Confidence-gated routing | Always take the model's top route | Below threshold, asking the user is better than a confident guess |
| Local cross-encoder rerank `[NOT BUILT]` | API reranker | Keeps rerank cost off the per-query bill |

## 5. Questions that expose the project's limits

Answer these honestly. Each has a real answer.

**"Your PHI redaction is regex. Isn't that weak?"**
Yes, and deliberately narrow. The dangerous failure here is *over*-redaction: a
naive date pattern turns a blood pressure of `120/80` into `[DATE]` and the model
then answers a different question, silently. So every pattern is anchored on
structure clinical values don't share — a full date has three components, a
reading has two. **Patient names are explicitly not detected**, because regex
can't separate a surname from an eponymous condition and a name pass would redact
"Crohn" and "Parkinson". That belongs to a dedicated NER model. The limitation is
written into the module docstring rather than hidden.

**"Can't someone bypass your injection detection trivially?"**
Yes. They're heuristics that raise the cost of the obvious attacks; they don't
make injection impossible. The design constraint people miss is the *false
positive* side: "patient instructions" and "discharge instructions" are ordinary
clinical phrases, so matching the word "instructions" alone makes the detector
useless in this domain — and a guardrail that fires on normal vocabulary gets
switched off by its operators. So detection requires an overriding verb *and* a
reference to prior instructions together. The real defence is that untrusted text
is never given authority, plus guard_out checking grounding independently.

**"Why is only one of the four agents built deep?"**
A deliberate time allocation across ~40-60 hours. The guideline copilot gets the
full evaluation treatment; the other three are demo depth. The README states
which is which rather than implying uniform rigour. Four agents at equal depth
would have been four shallow agents.

**"Why no real patient data?"**
Every corpus is public-domain or open-access — CDC, WHO, MedlinePlus, PubMed,
openFDA, CMS coverage determinations — and patient notes are LLM-generated
synthetic. MIMIC was excluded on purpose because it requires credentialed access,
and putting credentialed data in a public portfolio repo is exactly the judgement
failure you don't want to demonstrate.

**"What's the weakest part?"**
The evaluation harness is the thinnest relative to how much it matters, and the
routing benchmark is 33 cases against a target of 100. Don't dress this up.

## 6. What is actually built — say this accurately

**Built and tested (124 tests, ruff and mypy --strict clean):**
shared spine and state contract · chunking with content-addressed ids · BM25 ·
reciprocal rank fusion · Chroma dense adapter · hybrid retriever with fail-closed
behaviour · ingestion from PubMed and MedlinePlus · PHI redaction · injection
detection · citation verification · guard_in/guard_out nodes · compiled mesh
graph with conditional routing · supervisor node with confidence gate and
fail-closed error handling · routing benchmark scorer with confusion matrix ·
Docker Compose stack

**Not built:** guideline subgraph (plan → retrieve → rerank → draft → verify →
revise) · cross-encoder rerank · faithfulness and citation-accuracy evals · red-team
suite · Langfuse tracing · cost and latency benchmarks · Postgres checkpointer
wiring · FastAPI/SSE layer · the three thin specialists · CI

**No metrics exist yet.** Nothing has been measured because no eval has been run
against a live model. If asked for numbers, say that — do not estimate.

## 7. Tone notes

- Lead with the failure modes, not the happy path. Anyone can demo a working RAG
  chain; far fewer can say what their system does when the vector store is down.
- When you don't know, say so and say what you'd measure to find out.
- Don't call it production-ready. Call it production-*shaped*: the reliability
  concerns are designed in and partly implemented, and you can name exactly what's
  missing.
