"""The composition root: the one place that builds the real thing.

Every other module takes its dependencies as arguments, which is what makes the
whole package testable without an API key or a network. That has to bottom out
somewhere, and this is the somewhere -- the only module that reads Settings,
constructs models, and reaches for Chroma.
"""

from dataclasses import dataclass
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from mesh.agents.discharge import (
    NoteFetcher,
    build_discharge_subgraph,
    build_instruction_drafter,
    build_interaction_lookup,
    build_medication_extractor,
)
from mesh.agents.guideline import (
    build_contradiction_detector,
    build_drafter,
    build_guideline_subgraph,
    build_planner,
)
from mesh.agents.prior_auth import build_criteria_reader, build_prior_auth_subgraph
from mesh.agents.supervisor import build_classifier, make_supervisor
from mesh.agents.triage import build_adviser, build_triage_subgraph
from mesh.graph import as_specialist, build_mesh
from mesh.models.config import Settings
from mesh.retrieval.dense import ChromaDense
from mesh.retrieval.hybrid import HybridRetriever
from mesh.retrieval.ingest import COLLECTION_BY_ROUTE
from mesh.retrieval.keyword import BM25Index
from mesh.retrieval.rerank import Reranker, build_cross_encoder


@dataclass(frozen=True)
class Retrieval:
    """One collection's retriever, and the store that rehydrates its ids.

    The two travel together because every specialist needs both: the retriever
    ranks chunk ids, the store turns them back into text for the reranker.
    """

    retriever: HybridRetriever
    store: ChromaDense


def build_retrieval(settings: Settings, *, collection: str, embeddings: Embeddings) -> Retrieval:
    """Assemble hybrid retrieval over one collection.

    The lexical index is built here, once, from everything ingestion wrote.
    BM25 scores a query against the whole corpus in memory, so unlike the dense
    half it cannot be queried lazily -- and a BM25Index built from nothing would
    leave "hybrid" retrieval quietly dense-only, which is the specific failure
    the README claims this design avoids.
    """
    dense = ChromaDense(
        host=settings.chroma_host,
        port=settings.chroma_port,
        collection=collection,
        embed_query=embeddings.embed_query,
    )

    return Retrieval(
        retriever=HybridRetriever(dense=dense, keyword=BM25Index(dense.all_chunks())),
        store=dense,
    )


def _no_reranking(pairs: list[tuple[str, str]]) -> list[float]:
    """Equal scores, so the reranker's stable sort leaves the fusion order alone."""
    return [0.0] * len(pairs)


def build_reranker() -> Reranker:
    """The cross-encoder when the `rerank` extra is installed, otherwise a no-op.

    Reranking only reorders evidence that is already retrieved and grounded, so
    running without it costs answer quality rather than correctness. Refusing to
    start would be the wrong trade -- the same reasoning as Reranker's own
    fallback when the scorer raises.
    """
    try:
        score_pairs = build_cross_encoder()
    except ImportError:
        return Reranker(score_pairs=_no_reranking)

    return Reranker(score_pairs=score_pairs)


def build_default_mesh(
    *,
    settings: Settings,
    model: BaseChatModel,
    embeddings: Embeddings,
    fetch_notes: NoteFetcher,
    checkpointer: Any = None,
) -> Any:
    """Wire the whole mesh: a retriever per collection, four specialists, one supervisor.

    The model, embeddings, and note fetcher are arguments rather than built here
    so this function stays the wiring and `main` stays the provider setup. It is
    also what lets the whole mesh be exercised end to end with a stub model and
    no API key.

    One reranker is shared across all four specialists: the cross-encoder is a
    loaded model, and four copies would be four times the memory for identical
    weights.
    """
    reranker = build_reranker()
    retrieval = {
        route: build_retrieval(settings, collection=collection, embeddings=embeddings)
        for route, collection in COLLECTION_BY_ROUTE.items()
    }

    guideline = build_guideline_subgraph(
        plan=build_planner(model),
        retriever=retrieval["guideline"].retriever,
        store=retrieval["guideline"].store,
        reranker=reranker,
        draft=build_drafter(model),
        detect_contradictions=build_contradiction_detector(model),
    )

    triage = build_triage_subgraph(
        retriever=retrieval["triage"].retriever,
        store=retrieval["triage"].store,
        reranker=reranker,
        advise=build_adviser(model),
    )

    prior_auth = build_prior_auth_subgraph(
        retriever=retrieval["prior_auth"].retriever,
        store=retrieval["prior_auth"].store,
        reranker=reranker,
        read_criteria=build_criteria_reader(model),
    )

    discharge = build_discharge_subgraph(
        extract_medications=build_medication_extractor(model),
        lookup_interactions=build_interaction_lookup(fetch_notes),
        retriever=retrieval["discharge"].retriever,
        store=retrieval["discharge"].store,
        reranker=reranker,
        draft=build_instruction_drafter(model),
    )

    return build_mesh(
        supervisor=make_supervisor(
            build_classifier(model), threshold=settings.route_confidence_threshold
        ),
        specialists={
            "guideline": as_specialist(guideline),
            "triage": as_specialist(triage),
            "prior_auth": as_specialist(prior_auth),
            "discharge": as_specialist(discharge),
        },
        checkpointer=checkpointer,
    )
