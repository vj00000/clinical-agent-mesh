# Resume bullets

**Standing rule: every bracketed number must come from an actual `make eval` run.**
Fabricated metrics are the fastest way to lose a senior interview, because the obvious
follow-up is "walk me through how you measured that." No number in this file has been
measured yet — there was no API key on the build machine.

To fill them in: `make up && make ingest && make eval && make eval-routing`, then paste
what those printed. Nothing else.

---

## The bullets

> **Clinical Agent Mesh** — LangGraph · LangChain · FastAPI · Chroma · Docker
> [github.com/vj00000/clinical-agent-mesh](https://github.com/vj00000/clinical-agent-mesh)

**1. Architecture — safe to use now**

> Built a hierarchical multi-agent clinical assistant in which a structured-output
> supervisor routes queries to four isolated LangGraph specialist subgraphs, each owning
> private state behind a typed parent/child contract, over public clinical corpora
> (MedlinePlus, PubMed, openFDA, CMS).

*Measured?* No number needed. Verifiable by reading the repo.

**2. Grounding — needs `make eval`**

> Cut ungrounded clinical claims to **[X]%** by making citation verification a graph node:
> every claim must map to a chunk the retriever actually returned, and an answer that
> fails is revised up to twice and then refused rather than returned.

*Fill from:* `make eval` → `citation_accuracy`. State it as accuracy, not as a
"reduction", unless you also run the no-guardrail baseline — see below.

**3. Calibrated refusal — needs `make eval`**

> Measured calibrated refusal on a 64-case labelled golden set containing 18 deliberately
> unanswerable questions, reaching **[X]%** refusal correctness with **[N]** answers given
> where no evidence supported one.

*Fill from:* `make eval` → `refusal_correctness` and `answered_when_unanswerable`.
This is the strongest bullet in the set, because almost nobody measures it.

**4. Routing — needs `make eval-routing`**

> Achieved **[X]%** routing accuracy across a labelled benchmark, with a confidence gate
> that asks a clarifying question instead of guessing below threshold.

*Fill from:* `make eval-routing`. Note the benchmark is 33 cases today; the spec's target
is 100. Either grow it first or say "33-case" — do not imply 100.

**5. Retrieval — needs a measured baseline**

> Raised retrieval recall **[X]%** over dense-only search via hybrid BM25 + vector fusion
> with cross-encoder reranking, targeting drug names and clinical codes where embeddings
> underperform.

*Measured?* **Not yet, and not by `make eval` either.** This needs a dense-only A/B run
that does not exist. Until then either drop the number and describe the design, or build
the comparison. Do not guess it.

**6. Evaluation and safety — partly measurable today**

> Shipped a regression-gated eval harness (calibrated refusal, citation accuracy, routing
> accuracy, injected-judge faithfulness) wired into CI, plus an 18-case adversarial suite
> whose injection and PHI cases run in the ordinary test suite with no API key.

*Measured?* The red-team half is real today — 273 fast tests include every injection and
PHI case. The metric half needs `make eval`.

---

## Talking points that need no numbers

Use these when asked "what was hard" or "what would you do differently":

- **The red team broke my own guardrail on its first run.** Writing the adversarial suite
  immediately found two holes in the injection detector — `disregard your rules` (the
  qualifier list had `previous` and `all` but not `your`) and persona jailbreaks. The
  lesson is not the fix; it is that a suite written to be adversarial found holes in a
  guardrail I believed was finished.

- **The composition root found two latent bugs that tests could not.** Nothing in
  production built a mesh until the very end, so `BM25Index` could never have been
  populated — hybrid retrieval would have silently degraded to dense-only with nothing
  raising — and a `tok_k`/`top_k` typo meant the real retriever never satisfied its own
  protocol. Both were invisible because tests pass stubs and `mypy` only checks `src/`.

- **Two bounded loops that end differently.** The guideline revise loop refuses when it
  runs out of attempts; the discharge rewrite loop returns its best attempt. Prose a grade
  too dense is still usable; an ungrounded clinical claim never is.

- **A safety escalation is exempt from the citation check.** Getting that wrong means a
  Chroma outage turns "call an ambulance" into "I don't have the evidence".

- **I chose a thin honest corpus over a rich fabricated one.** CMS publishes the coverage
  policy index without a key but keeps the criteria behind a licence. Rather than scrape
  it or invent plausible criteria, the prior-auth agent says which policy governs a
  request and admits it cannot resolve the criteria.

- **What I would do next**, in order: run the evals and fill in the numbers; grow routing
  to 100 cases; build the dense-only baseline that bullet 5 needs; label relevant chunks
  so context recall becomes computable.
