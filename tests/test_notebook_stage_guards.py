"""Keep Kaggle long-running stages fail-fast and output-checked."""

import json
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / "deploy" / "kaggle_triage.ipynb"


def test_notebook_checks_long_running_commands_and_full_test_suite():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    for i, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        # No cell may use unchecked notebook shell commands (!)
        assert not any(line.lstrip().startswith("!") for line in src.splitlines()), \
            f"cell {i} contains an unchecked notebook shell command"

        # Skip pure data/config cells that do not run subprocesses
        if "# 4. Environment file" in src or "# 10. Inspect the fresh benchmark results" in src:
            continue

        assert "run_checked(" in src, f"cell {i} bypasses checked subprocesses"

    all_sources = " ".join("".join(c.get("source", [])) for c in notebook["cells"])
    assert "'pytest', 'tests', '-q'" in all_sources
    assert "'pull', '--rebase'" in all_sources
    assert "fresh results" in all_sources
    assert "run_manifest" in all_sources
    assert "fresh checkpoint" in all_sources
    assert "retrieval-eval" in all_sources
    assert "hub-eval" in all_sources
    assert "OLLAMA_MAX_LOADED_MODELS" in all_sources
    assert "OLLAMA_KEEP_ALIVE'] = '5m'" in all_sources
    assert "record resident Ollama models" in all_sources
    assert "_run_checked_with_peak_rss" in all_sources
    assert "VmRSS:" in all_sources
    assert "CalledProcessError" in all_sources
