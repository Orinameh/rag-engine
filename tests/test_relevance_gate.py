"""Relevance gate logic (pure — no DB, no reranker, no LLM)."""

import pytest

from rag_engine.api import relevance


def chunk(score: float) -> dict:
    return {"doc_id": "d", "content": "x", "rerank_score": score}


@pytest.fixture(autouse=True)
def _defaults(monkeypatch):
    # Gate ON at the calibrated default, floor 0.10; keep the suite
    # independent of any local .env overrides.
    monkeypatch.setattr(relevance.get_settings(), "rerank_min_score", 0.50)
    monkeypatch.setattr(relevance.get_settings(), "rerank_chunk_min_score", 0.10)


def test_off_corpus_top_score_refuses():
    kept, refusal = relevance.apply_relevance_gate([chunk(0.0), chunk(0.0)])
    assert kept == []
    assert refusal == relevance.REFUSAL_MESSAGE


def test_on_corpus_scores_pass():
    kept, refusal = relevance.apply_relevance_gate([chunk(0.98), chunk(0.92)])
    assert refusal is None
    assert [c["rerank_score"] for c in kept] == [0.98, 0.92]


def test_tail_chunks_below_floor_are_trimmed():
    kept, refusal = relevance.apply_relevance_gate(
        [chunk(0.98), chunk(0.09), chunk(0.01)]
    )
    assert refusal is None
    assert [c["rerank_score"] for c in kept] == [0.98]


def test_floor_never_trims_everything():
    # Degenerate: top passes the gate at 0.50 but every chunk is under the
    # floor — the gate already said "proceed", so keep the ranking as-is
    # rather than handing the LLM an empty context.
    kept, refusal = relevance.apply_relevance_gate([chunk(0.51)])
    assert refusal is None
    assert kept == [chunk(0.51)]


def test_empty_retrieval_refuses():
    kept, refusal = relevance.apply_relevance_gate([])
    assert kept == []
    assert refusal == relevance.REFUSAL_MESSAGE


def test_zero_threshold_disables_gate(monkeypatch):
    monkeypatch.setattr(relevance.get_settings(), "rerank_min_score", 0.0)
    _kept, refusal = relevance.apply_relevance_gate([chunk(0.0)])
    # Gate off: even a 0.0 top proceeds (floor still trims nothing useful).
    assert refusal is None


def test_boundary_is_inclusive():
    _kept, refusal = relevance.apply_relevance_gate([chunk(0.50)])
    assert refusal is None  # < threshold refuses; == passes
