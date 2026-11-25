"""Gate-mode provenance: the JEV gate's active mode is always local-calibrated."""

from src.core.decision_gate import DecisionGate
from src.core.hcad_retriever import HCADRetriever


def test_gate_mode_is_local_calibrated():
    gate = DecisionGate()
    assert gate.mode == "local-calibrated"


def test_retriever_gate_mode_uninitialized():
    retriever = HCADRetriever.__new__(HCADRetriever)
    retriever._decision_gate = None
    assert retriever.gate_mode == "uninitialized"


def test_retriever_gate_mode_delegates_to_gate():
    retriever = HCADRetriever.__new__(HCADRetriever)
    retriever._decision_gate = DecisionGate()
    assert retriever.gate_mode == "local-calibrated"


def test_retriever_gate_mode_tolerates_gate_without_mode():
    retriever = HCADRetriever.__new__(HCADRetriever)
    retriever._decision_gate = object()
    assert retriever.gate_mode == "unknown"
