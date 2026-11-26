"""Unit tests for polarity/negation/exclusion scoring and lexical hybrid fusion."""

import numpy as np
import pytest

from src.core.polarity_scorer import (
    classify_clinical_polarity,
    extract_exclusion_terms,
    polarity_adjustments,
    pool_bm25_normalized,
)
from src.evaluation.polarity_benchmark import build_corpus, build_queries


def test_clinical_polarity_covers_all_curated_minimal_pairs():
    corpus = build_corpus()
    queries = build_queries()
    by_id = {d["id"]: d for d in corpus}

    for q in queries:
        gold = by_id[q["gold_id"]]
        opp = by_id[q["opposite_id"]]
        adj = polarity_adjustments(q["query"], [gold["text"], opp["text"]])
        assert adj[0] > adj[1], (
            f"Polarity scorer failed on {q['pair']} ({q['polarity_type']}/{q['phrasing']}): "
            f"gold={adj[0]:.3f} vs opp={adj[1]:.3f}"
        )


def test_metadata_polarity_overrides_ambiguous_text():
    assert classify_clinical_polarity("Drug X and Disease Y.", {"relation": "contraindication"}) == -1
    assert classify_clinical_polarity("Drug X and Disease Y.", {"relation": "indication"}) == 1
    assert classify_clinical_polarity("Metformin is not recommended for severe CKD.") == -1
    assert classify_clinical_polarity("Metformin is indicated for type 2 diabetes.") == 1


def test_exclusion_query_penalizes_affirmative_excluded_entity():
    q = "What therapy manages severe asthma without corticosteroids?"
    assert extract_exclusion_terms(q) == ["corticosteroids"]

    doc_pos = "Biologic therapy manages severe eosinophilic asthma in refractory patients."
    doc_neg = "Systemic corticosteroids manage severe asthma exacerbations."
    doc_explicit_neg = "Inhaled bronchodilators manage mild asthma without corticosteroids."
    adj = polarity_adjustments(q, [doc_pos, doc_neg, doc_explicit_neg])
    assert adj[0] > adj[1]
    assert adj[2] > adj[1]


def test_short_claim_negation_alignment_ignores_long_prose():
    q_neg = "Aspirin does not prevent this viral complication."
    c_neg = "Aspirin should not be given because it fails to prevent this complication."
    c_pos = "Aspirin is effective and prevents this viral complication."
    long_prose = " ".join(["clinical evidence sentence"] * 65 + ["without"] + ["details"] * 5)

    adj = polarity_adjustments(q_neg, [c_neg, c_pos, long_prose])
    assert adj[0] > 0
    assert adj[1] < 0
    assert adj[2] == 0.0


def test_pool_bm25_normalized_bounds_and_empty_match():
    docs = [
        "Metformin treats type 2 diabetes.",
        "Warfarin prevents thromboembolism.",
        "",
    ]
    scores = pool_bm25_normalized("metformin diabetes", docs)
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == 0.0
    assert scores[2] == 0.0
    assert np.all(pool_bm25_normalized("unseenxyz", docs) == 0.0)


def test_dense_anchors_skip_polarity_conflicting_vector_hit(monkeypatch):
    from src.core.hcad_retriever import HCADRetriever

    monkeypatch.setenv("HCAD_DENSE_ANCHOR_K", "1")
    monkeypatch.setenv("HCAD_DENSE_ANCHOR_MIN", "0.48")

    cands = [
        {"document": "Metformin is contraindicated in renal failure.", "source": "vector"},
        {"document": "Metformin is indicated for type 2 diabetes.", "source": "vector"},
    ]
    dense = np.array([0.92, 0.80], dtype=np.float32)
    phi = np.array([0.50, 0.78], dtype=np.float32)
    pol_adj = np.array([-0.35, 0.08], dtype=np.float32)

    anchors = HCADRetriever._dense_anchors(cands, dense, phi, pol_adj=pol_adj)
    assert [a["document"] for a in anchors] == [
        "Metformin is indicated for type 2 diabetes."
    ]


def test_config_toggles_and_anchor_floor(monkeypatch):
    from src.config import DENSE_ANCHOR_MIN, lex_weight, polarity_scoring_enabled

    monkeypatch.delenv("HCAD_DENSE_ANCHOR_MIN", raising=False)
    monkeypatch.delenv("HCAD_LEX_WEIGHT", raising=False)
    monkeypatch.delenv("HCAD_POLARITY_SCORING", raising=False)
    assert DENSE_ANCHOR_MIN == pytest.approx(0.48)
    assert lex_weight() == pytest.approx(0.20)
    assert polarity_scoring_enabled() is True

    monkeypatch.setenv("HCAD_LEX_WEIGHT", "0.0")
    monkeypatch.setenv("HCAD_POLARITY_SCORING", "0")
    assert lex_weight() == 0.0
    assert polarity_scoring_enabled() is False
