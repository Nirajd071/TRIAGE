"""Keep Kaggle long-running stages fail-fast and output-checked."""

import json
import re
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / "deploy" / "kaggle_triage.ipynb"


def _sources():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return {i: "".join(c["source"]) for i, c in enumerate(notebook["cells"])}


def test_notebook_checks_long_running_commands_and_full_test_suite():
    src = _sources()
    checked = (2, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24)
    for cell in checked:
        source = src[cell]
        assert "run_checked(" in source, f"cell {cell} bypasses checked subprocesses"
        assert not any(line.lstrip().startswith("!") for line in source.splitlines()), \
            f"cell {cell} contains an unchecked notebook shell command"
    assert "'pytest', 'tests', '-q'" in src[9]
    assert "'pull', '--rebase'" in src[2]
    assert "fresh results" in src[17]
    assert "run_manifest" in src[17]
    assert "fresh checkpoint" in src[14]
    assert "retrieval-eval" in src[15]
    assert "hub-eval" in src[16]


def test_settings_cell_exposes_only_the_judge_index():
    settings = _sources()[1]
    assert settings.startswith("# 0. EXPERIMENT SETTINGS")
    editable, _, fixed = settings.partition("Do NOT edit below this line.")
    assignments = re.findall(r"(?m)^([A-Z][A-Z0-9_]*) = ", editable)
    assert assignments == ["JUDGE_MODEL_INDEX"]
    for const in ("JUDGE_QUANT = 'q4_K_M'", "N_SAMPLES = 30",
                  "GENERATOR_TAG = 'llama3.1:8b-instruct-q4_K_M'",
                  "OLLAMA_KV_CACHE_TYPE = 'q8_0'"):
        assert const in fixed


def test_settings_cell_validates_the_index():
    settings = _sources()[1]

    def run(value):
        code = re.sub(r"(?m)^JUDGE_MODEL_INDEX = 0$", f"JUDGE_MODEL_INDEX = {value}",
                      settings, count=1)
        ns = {}
        exec(code, ns)
        return ns["JUDGE_INDICES"]

    assert run("2") == [2]
    assert run("[0, 3]") == [0, 3]
    for bad in ("True", "'2'", "[1, 1]", "[0, 1, 2]", "[]"):
        try:
            run(bad)
        except (TypeError, ValueError):
            continue
        raise AssertionError(f"settings cell accepted JUDGE_MODEL_INDEX = {bad}")


def test_ollama_cell_uses_registry_quantized_tags_and_memory_guards():
    src = _sources()
    ollama, env = src[5], src[6]
    assert "jm.parse_indices(JUDGE_INDICES)" in ollama
    assert "jm.resolve_judge_tag(i, JUDGE_QUANT)" in ollama
    assert "pull_with_retry(GENERATOR_TAG" in ollama
    assert "OLLAMA_KV_CACHE_TYPE" in ollama and "OLLAMA_NUM_PARALLEL" in ollama
    assert "OLLAMA_MAX_LOADED_MODELS" in ollama
    assert "100% GPU" in ollama
    # no hard-coded unquantized judge pulls
    assert "'pull', 'gemma2:9b'" not in ollama and "'pull', 'qwen2.5:14b'" not in ollama
    # .env carries the index, never a hard-coded ensemble
    assert "JUDGE_MODEL_INDEX=" in env and "JUDGE_ENSEMBLE=" not in env
    assert "judge_ensemble_specs" in env and "GENERATOR_TAG" in env


def test_notebook_covers_revision_stages():
    src = _sources()
    # benchmark checks the statistics and latency outputs on N_SAMPLES questions
    _ran_ok = any(key in src[17] for key in ("EXPECTED_JUDGES",))
    assert _ran_ok
    for key in ("statistical_tests", "inter_judge_agreement", "latency_breakdown.json",
                "str(N_SAMPLES)", "EXPECTED_JUDGES"):
        assert key in src[17]
    assert "--no-deps" in src[4]
    assert "'WANDB_MODE'" in src[6] and "'TYPESAFE_API_KEY'" in src[6]
    assert "'50'" not in src[17] and "'50'" not in src[18]
    assert "evidence_diagnostics" in src[19]
    assert "'polarity'" in src[20] and "fresh results" in src[20]
    assert "'medhallu'" in src[21]
    assert "'figures'" in src[22]
    assert "results/polarity_benchmark.json" in src[24]
