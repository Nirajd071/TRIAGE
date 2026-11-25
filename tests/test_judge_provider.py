"""Tests for pure local Ollama LLM provider enforcement."""

import pytest

from src.config import judge_ensemble_specs
from src.core.llm_client import UnifiedLLMClient


@pytest.mark.parametrize("bad_provider", ["nvidia", "gemini", "groq", "openrouter", "huggingface"])
def test_unified_client_rejects_non_ollama_providers_when_strict(bad_provider):
    with pytest.raises(RuntimeError, match="disabled"):
        UnifiedLLMClient(model="qwen3:14b-q4_K_M", provider=bad_provider, strict=True)


def test_unified_client_coerces_non_ollama_provider_when_non_strict():
    client = UnifiedLLMClient(model="qwen3:14b-q4_K_M", provider="huggingface", strict=False)
    assert client.provider == "ollama"


def test_judge_ensemble_specs_returns_local_ollama_only(monkeypatch):
    monkeypatch.delenv("JUDGE_ENSEMBLE", raising=False)
    monkeypatch.setenv("JUDGE_MODEL_INDEX", "0,3")
    monkeypatch.setenv("JUDGE_QUANT", "q4_K_M")
    specs = judge_ensemble_specs()
    assert specs == [
        ("ollama", "qwen3:14b-q4_K_M"),
        ("ollama", "qwen2.5:14b-instruct-q4_K_M"),
    ]
    assert all(prov == "ollama" for prov, _ in specs)
