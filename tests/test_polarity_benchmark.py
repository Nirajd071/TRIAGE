"""Unit tests for Polarity Discrimination Benchmark."""

import pytest
from unittest.mock import MagicMock

from src.evaluation.polarity_benchmark import (
    PAIRED_CLINICAL_POLARITY_CASES,
    PolarityBenchmark,
)


def test_polarity_test_cases_count_and_structure():
    assert len(PAIRED_CLINICAL_POLARITY_CASES) >= 30, "Benchmark must contain at least 30 paired clinical cases"

    for pair in PAIRED_CLINICAL_POLARITY_CASES:
        assert "pair_id" in pair
        assert "entity" in pair
        assert "positive" in pair
        assert "negative" in pair

        pos = pair["positive"]
        neg = pair["negative"]
        assert pos["expected_polarity"] == "indicated"
        assert neg["expected_polarity"] == "contraindicated"
        assert len(pos["safety_keywords"]) > 0
        assert len(neg["safety_keywords"]) > 0
        assert "?" in pos["query"]
        assert "?" in neg["query"]


def test_polarity_fidelity_classifier():
    # Test contraindication awareness
    good_contra_ans = (
        "Metformin is strictly contraindicated in patients with severe renal failure "
        "(eGFR < 30) due to high risk of fatal lactic acidosis. Clinicians must avoid usage."
    )
    res = PolarityBenchmark._evaluate_polarity_fidelity(
        good_contra_ans, "contraindicated", ["lactic acidosis", "egfr", "contraindicated"]
    )
    assert res["correct_polarity"] is True
    assert res["inversion_error"] is False
    assert res["has_warning_language"] is True

    # Test dangerous inversion error (endorses without warning)
    bad_contra_ans = (
        "Metformin is indicated and highly recommended for all patients with diabetes, "
        "improving glycemic control effectively."
    )
    res_bad = PolarityBenchmark._evaluate_polarity_fidelity(
        bad_contra_ans, "contraindicated", ["lactic acidosis", "egfr"]
    )
    assert res_bad["inversion_error"] is True
    assert res_bad["correct_polarity"] is False


def test_safety_recall_scoring():
    contexts = [
        {"document": "Lactic acidosis is a rare but life-threatening complication of metformin in renal failure."},
        {"document": "Patients with eGFR under 30 should avoid metformin therapy."},
    ]
    keywords = ["lactic acidosis", "egfr", "avoid", "nonexistent_term"]
    recall = PolarityBenchmark._check_safety_recall(contexts, keywords)
    # 3 out of 4 found
    assert recall == 0.75
