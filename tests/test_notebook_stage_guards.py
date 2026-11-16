"""Keep Kaggle long-running stages fail-fast and output-checked."""

import json
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / "deploy" / "kaggle_triage.ipynb"


def test_notebook_checks_long_running_commands_and_full_test_suite():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    sources = {i: "".join(notebook["cells"][i]["source"])
               for i in (1, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23)}

    for cell, source in sources.items():
        assert "run_checked(" in source, f"cell {cell} bypasses checked subprocesses"
        assert not any(line.lstrip().startswith("!") for line in source.splitlines()), \
            f"cell {cell} contains an unchecked notebook shell command"
    assert "'pytest', 'tests', '-q'" in sources[8]
    assert "'pull', '--rebase'" in sources[1]
    assert "fresh results" in sources[16]
    assert "run_manifest" in sources[16]
    assert "fresh checkpoint" in sources[13]
    assert "retrieval-eval" in sources[14]
    assert "hub-eval" in sources[15]


def test_notebook_covers_revision_stages():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    src = {i: "".join(c["source"]) for i, c in enumerate(notebook["cells"])}
    # multi-judge: both ensemble judges pulled and configured
    assert "'pull', 'gemma2:9b'" in src[4] and "'pull', 'qwen2.5:14b'" in src[4]
    assert "JUDGE_ENSEMBLE=ollama:gemma2:9b,ollama:qwen2.5:14b" in src[5]
    # benchmark checks the new statistics and latency outputs
    for key in ("statistical_tests", "inter_judge_agreement", "latency_breakdown.json"):
        assert key in src[16]
    assert "'polarity'" in src[19] and "fresh results" in src[19]
    assert "'medhallu'" in src[20]
    assert "'figures'" in src[21]
    assert "results/polarity_benchmark.json" in src[23]
