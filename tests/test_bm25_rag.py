"""Unit tests for BM25 and Hybrid RRF retrieval baselines (in-memory, no Chroma)."""

import pytest

from src.baselines.sparse_retrievers import (
    BM25Index,
    BM25RAG,
    HybridRAG,
    bm25_tokenize,
    reciprocal_rank_fusion,
)


def _sample_docs():
    return [
        (
            "d1",
            "Metformin is contraindicated in severe renal impairment due to lactic acidosis risk.",
            {"subject": "Metformin"},
        ),
        (
            "d2",
            "Warfarin interacts with aspirin and increases bleeding risk.",
            {"subject": "Warfarin"},
        ),
        (
            "d3",
            "Metformin is first-line therapy for type 2 diabetes mellitus.",
            {"subject": "Metformin"},
        ),
        (
            "d4",
            "Lisinopril is an ACE inhibitor used for hypertension.",
            {"subject": "Lisinopril"},
        ),
    ]


def test_bm25_tokenize_preserves_clinical_hyphenated_and_alphanumeric_terms():
    tokens = bm25_tokenize("Type-2 diabetes & HbA1c, CYP2C19!")
    assert "type-2" in tokens
    assert "diabetes" in tokens
    assert "hba1c" in tokens
    assert "cyp2c19" in tokens


def test_bm25_ranks_exact_lexical_match_first():
    rows = _sample_docs()
    idx = BM25Index([r[1] for r in rows], ids=[r[0] for r in rows], metadatas=[r[2] for r in rows])
    hits = idx.search("lactic acidosis renal impairment metformin", top_k=2)
    assert [h["id"] for h in hits] == ["d1", "d3"]
    assert hits[0]["score"] > hits[1]["score"] > 0


def test_bm25_validates_alignment_and_top_k():
    with pytest.raises(ValueError, match="align"):
        BM25Index(["doc1"], ids=["d1", "d2"])
    idx = BM25Index(["doc1"], ids=["d1"])
    with pytest.raises(ValueError, match="positive"):
        idx.search("doc1", top_k=0)


def test_hybrid_rrf_fuses_sparse_and_dense_rankings():
    rows = _sample_docs()
    idx = BM25Index([r[1] for r in rows], ids=[r[0] for r in rows], metadatas=[r[2] for r in rows])

    def fake_dense(_query, top_k):
        return [
            {"id": "d3", "text": rows[2][1], "score": 0.91, "metadata": rows[2][2]},
            {"id": "d2", "text": rows[1][1], "score": 0.70, "metadata": rows[1][2]},
        ][:top_k]

    hybrid = HybridRAG(index=idx, dense_retrieve=fake_dense, top_k=2, pool=4)
    hits = hybrid.retrieve("metformin diabetes")
    assert len(hits) == 2
    assert hits[0]["id"] == "d3"  # strong in both sparse and dense


def test_reciprocal_rank_fusion_is_deterministic_on_ties():
    a = [{"id": "x", "text": "x"}, {"id": "y", "text": "y"}]
    b = [{"id": "y", "text": "y"}, {"id": "x", "text": "x"}]
    fused = reciprocal_rank_fusion([a, b], top_k=2, k=60)
    assert [h["id"] for h in fused] == ["x", "y"]
    assert fused[0]["score"] == fused[1]["score"]


def test_bm25_rag_reuses_generator_llm_without_recreating_retriever():
    class FakeLLM:
        model = "llama3.1:8b-instruct-q4_K_M"

    class FakeGenerator:
        llm = FakeLLM()

    rows = _sample_docs()
    idx = BM25Index([r[1] for r in rows], ids=[r[0] for r in rows], metadatas=[r[2] for r in rows])
    rag = BM25RAG(index=idx, top_k=2, generator=FakeGenerator())
    hits = rag.retrieve("warfarin bleeding")
    assert hits[0]["id"] == "d2"
    msgs = rag.build_messages("Does warfarin cause bleeding?", hits)
    assert any("Warfarin interacts with aspirin" in m["content"] for m in msgs)
