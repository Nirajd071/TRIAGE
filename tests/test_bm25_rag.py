"""Unit tests for BM25RAG and HybridRAG baselines."""

import pytest
from unittest.mock import MagicMock, patch

from src.baselines.bm25_rag import (
    BM25RAG,
    HybridRAG,
    _SimpleBM25,
    tokenize_corpus_text,
)


def test_tokenize_corpus_text():
    text = "Metformin is indicated for Type-2 Diabetes in patients with normal eGFR."
    tokens = tokenize_corpus_text(text)
    assert "metformin" in tokens
    assert "indicated" in tokens
    assert "diabetes" in tokens
    # Stopwords should be removed
    assert "is" not in tokens
    assert "for" not in tokens
    assert "in" not in tokens


def test_simple_bm25_scoring():
    corpus = [
        ["metformin", "indicated", "diabetes"],
        ["aspirin", "contraindicated", "reye", "syndrome"],
        ["doxycycline", "lyme", "disease"],
    ]
    bm25 = _SimpleBM25(corpus)
    scores = bm25.get_scores(["metformin", "diabetes"])
    assert scores[0] > scores[1]
    assert scores[0] > scores[2]


def test_bm25_rag_mocked_retrieval(tmp_path):
    rag = BM25RAG.__new__(BM25RAG)
    rag.index_path = tmp_path / "test_bm25.pkl"
    rag.k1 = 1.5
    rag.b = 0.75
    rag.corpus_docs = [
        {"id": "doc1", "document": "Metformin treats diabetes mellitus efficiently.", "metadata": {}},
        {"id": "doc2", "document": "Lisinopril is an ACE inhibitor for hypertension.", "metadata": {}},
    ]
    rag.bm25 = _SimpleBM25([
        tokenize_corpus_text(d["document"]) for d in rag.corpus_docs
    ])

    results = rag.retrieve("diabetes metformin", top_k=1)
    assert len(results) == 1
    assert results[0]["id"] == "doc1"
    assert results[0]["retriever"] == "bm25"
    assert results[0]["score"] > 0.0


def test_hybrid_rag_convex_fusion():
    mock_dense = MagicMock()
    mock_dense.retrieve.return_value = [
        {"id": "d1", "document": "Doc 1", "metadata": {}, "score": 0.8},
        {"id": "d2", "document": "Doc 2", "metadata": {}, "score": 0.4},
    ]

    mock_bm25 = MagicMock()
    mock_bm25.retrieve.return_value = [
        {"id": "d2", "document": "Doc 2", "metadata": {}, "score": 10.0},
        {"id": "d1", "document": "Doc 1", "metadata": {}, "score": 2.0},
    ]

    hybrid = HybridRAG.__new__(HybridRAG)
    hybrid.dense_rag = mock_dense
    hybrid.bm25_rag = mock_bm25
    hybrid.dense_weight = 0.5
    hybrid.bm25_weight = 0.5
    hybrid.top_k = 2

    results = hybrid.retrieve("test query")
    assert len(results) == 2
    assert all("score" in r for r in results)
    assert results[0]["retriever"] == "hybrid_bm25_dense"
