"""Unit tests for Multi-Judge Evaluator and Inter-Annotator Agreement Metrics."""

import numpy as np
import pytest
from unittest.mock import MagicMock

from src.evaluation.base_paper_evaluator import BasePaperEvaluator, MultiJudgeEvaluator, ScoringResult
from src.evaluation.metrics import (
    bootstrap_ci,
    cohens_kappa,
    fleiss_kappa,
    wilcoxon_signed_rank_test,
)


def test_cohens_kappa():
    # Perfect agreement
    r1 = ["A", "B", "Tie", "A", "B"]
    r2 = ["A", "B", "Tie", "A", "B"]
    assert cohens_kappa(r1, r2) == 1.0

    # Total disagreement
    r1 = ["A", "A", "A", "A"]
    r2 = ["B", "B", "B", "B"]
    assert cohens_kappa(r1, r2) <= 0.0


def test_fleiss_kappa():
    # 5 subjects, 3 categories, 3 raters per subject
    # All raters agree on category 0 for subject 0, category 1 for subject 1, etc.
    perfect_matrix = np.array([
        [3, 0, 0],
        [0, 3, 0],
        [0, 0, 3],
        [3, 0, 0],
        [0, 3, 0],
    ])
    assert fleiss_kappa(perfect_matrix) == 1.0


def test_bootstrap_ci():
    data = [70.0, 72.0, 75.0, 73.0, 74.0, 71.0]
    low, high = bootstrap_ci(data, num_resamples=200, confidence_level=0.95)
    mean = np.mean(data)
    assert low <= mean <= high


def test_wilcoxon_signed_rank_test():
    x = [80.0, 85.0, 90.0, 78.0, 88.0, 92.0, 84.0, 86.0, 89.0, 91.0]
    y = [70.0, 72.0, 75.0, 68.0, 74.0, 78.0, 71.0, 73.0, 75.0, 77.0]
    res = wilcoxon_signed_rank_test(x, y)
    assert res["p_value"] < 0.05
    assert res["significant_at_05"] is True


def test_multi_judge_evaluator_scoring():
    # Mock two judges with different scores
    mock_eval1 = MagicMock(spec=BasePaperEvaluator)
    mock_eval1.judge = MagicMock()
    mock_eval1.judge.model = "gemma2:9b"
    mock_eval1.evaluate_scoring.return_value = ScoringResult(
        comprehensiveness=80.0, diversity=75.0, empowerment=85.0,
        logical=80.0, readability=90.0, overall_score=82.0, rationale="Judge 1"
    )

    mock_eval2 = MagicMock(spec=BasePaperEvaluator)
    mock_eval2.judge = MagicMock()
    mock_eval2.judge.model = "qwen2.5:14b"
    mock_eval2.evaluate_scoring.return_value = ScoringResult(
        comprehensiveness=90.0, diversity=85.0, empowerment=75.0,
        logical=80.0, readability=80.0, overall_score=82.0, rationale="Judge 2"
    )

    multi = MultiJudgeEvaluator(judges=[mock_eval1, mock_eval2])
    res = multi.evaluate_scoring(
        question="What is hypertension?", answer="High BP", reference_document="Ref"
    )

    assert res["num_judges"] == 2
    assert res["consensus"]["comprehensiveness"] == 85.0
    assert res["consensus"]["overall_score"] == 82.0
    assert "gemma2:9b" in res["per_judge"]
    assert "qwen2.5:14b" in res["per_judge"]


def test_multi_judge_evaluator_selection():
    mock_eval1 = MagicMock(spec=BasePaperEvaluator)
    mock_eval1.judge = MagicMock()
    mock_eval1.judge.model = "gemma2:9b"
    mock_eval1.evaluate_selection.return_value = {
        "criteria_winners": {crit: "HCAD-RAG" for crit in BasePaperEvaluator.CRITERIA_LIST},
        "overall_winner": "HCAD-RAG",
    }

    mock_eval2 = MagicMock(spec=BasePaperEvaluator)
    mock_eval2.judge = MagicMock()
    mock_eval2.judge.model = "qwen2.5:14b"
    mock_eval2.evaluate_selection.return_value = {
        "criteria_winners": {crit: "HCAD-RAG" for crit in BasePaperEvaluator.CRITERIA_LIST},
        "overall_winner": "HCAD-RAG",
    }

    multi = MultiJudgeEvaluator(judges=[mock_eval1, mock_eval2])
    sel_res = multi.evaluate_selection(
        question="Query", answer_a="Ans A", answer_b="Ans B", reference_document="Ref",
        label_a="HCAD-RAG", label_b="Vanilla RAG"
    )

    assert sel_res["overall_winner"] == "HCAD-RAG"
    assert sel_res["agreement"]["type"] == "cohens_kappa"
    assert sel_res["agreement"]["kappa"] == 1.0


def test_multi_judge_evaluator_selection_three_judges_fleiss_kappa():
    mock_eval1 = MagicMock(spec=BasePaperEvaluator)
    mock_eval1.judge = MagicMock()
    mock_eval1.judge.model = "qwen3:14b"
    mock_eval1.evaluate_selection.return_value = {
        "criteria_winners": {crit: "HCAD-RAG" for crit in BasePaperEvaluator.CRITERIA_LIST},
        "overall_winner": "HCAD-RAG",
    }

    mock_eval2 = MagicMock(spec=BasePaperEvaluator)
    mock_eval2.judge = MagicMock()
    mock_eval2.judge.model = "gemini-flash"
    mock_eval2.evaluate_selection.return_value = {
        "criteria_winners": {crit: "HCAD-RAG" for crit in BasePaperEvaluator.CRITERIA_LIST},
        "overall_winner": "HCAD-RAG",
    }

    mock_eval3 = MagicMock(spec=BasePaperEvaluator)
    mock_eval3.judge = MagicMock()
    mock_eval3.judge.model = "groq-qwen"
    mock_eval3.evaluate_selection.return_value = {
        "criteria_winners": {crit: "HCAD-RAG" for crit in BasePaperEvaluator.CRITERIA_LIST},
        "overall_winner": "HCAD-RAG",
    }

    multi = MultiJudgeEvaluator(judges=[mock_eval1, mock_eval2, mock_eval3])
    sel_res = multi.evaluate_selection(
        question="Query", answer_a="Ans A", answer_b="Ans B", reference_document="Ref",
        label_a="HCAD-RAG", label_b="Vanilla RAG"
    )

    assert sel_res["overall_winner"] == "HCAD-RAG"
    assert sel_res["n_valid_judges"] == 3
    assert sel_res["agreement"]["type"] == "fleiss_kappa"
    assert sel_res["agreement"]["kappa"] == 1.0

