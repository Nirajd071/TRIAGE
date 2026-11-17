"""Unit tests for HCADRetriever ablation toggles and AblationBenchmark."""

import pytest
from unittest.mock import MagicMock

from src.core.hcad_retriever import HCADRetriever
from src.evaluation.ablation_benchmark import AblationBenchmark


def test_hcad_ablation_toggles_initialization():
    r_full = HCADRetriever()
    assert r_full.enable_decoupling is True
    assert r_full.enable_decision_gate is True
    assert r_full.enable_aniso is True
    assert r_full.enable_hypergraph is True

    r_no_decouple = HCADRetriever(enable_decoupling=False)
    assert r_no_decouple.enable_decoupling is False

    r_no_gate = HCADRetriever(enable_decision_gate=False)
    assert r_no_gate.enable_decision_gate is False

    r_dense_only = HCADRetriever(enable_hypergraph=False)
    assert r_dense_only.enable_hypergraph is False


def test_ablation_benchmark_arms():
    mock_eval = MagicMock()
    bm = AblationBenchmark(evaluator=mock_eval)
    expected_arms = {
        "HCAD-Full",
        "HCAD-w/o Decoupling",
        "HCAD-w/o Decision Gate",
        "HCAD-w/o Anisotropic Tensor",
        "HCAD-Dense Only",
    }
    assert set(bm.arms.keys()) == expected_arms


def test_ablation_arms_execution():
    from unittest.mock import patch
    import numpy as np

    retriever = HCADRetriever(enable_decoupling=False, enable_aniso=False, enable_decision_gate=False)
    assert retriever.enable_decoupling is False
    assert retriever.enable_aniso is False
    assert retriever.enable_decision_gate is False

