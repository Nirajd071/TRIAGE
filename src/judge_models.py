"""Judge-model registry: pick the LLM judge by index + quantization level.

This module is the ONE place that knows which judge models exist. Everything
else (src/config.py -> JUDGE_OLLAMA_MODEL -> UnifiedLLMClient ->
BasePaperEvaluator) reads the resolved tag, so switching the judge never
requires editing source files:

    JUDGE_MODEL_INDEX=0,1,2,3 python src/main.py base-paper ...

Judges (4 cross-vendor families, zero overlap with Llama-3.1-8B generator):
    0 = Qwen3-14B                 (Alibaba)
    1 = Gemma 3 12B IT            (Google)
    2 = Phi-4 14B                 (Microsoft)
    3 = Mistral 7B Instruct v0.3  (Mistral AI)

Judge ensemble used by the benchmark (config.judge_ensemble_specs):
    1. JUDGE_ENSEMBLE      - explicit "ollama:<model>,..." list
    2. JUDGE_MODEL_INDEX   - "0" = one judge, "0,1,2,3" = full 4-judge panel,
                             each quantized via JUDGE_QUANT (default q4_K_M)
    3. DEFAULT_JUDGE_ENSEMBLE in config.py

Legacy single-judge tag (config.JUDGE_OLLAMA_MODEL):
    JUDGE_OLLAMA_MODEL (exact tag) > first JUDGE_MODEL_INDEX > DEFAULT_JUDGE_TAG

Only the JUDGE is affected; the generator (llama3.1:8b-instruct-q4_K_M) is untouched.

Kept dependency-free on purpose: the Kaggle notebook kernel imports it
directly (different Python than the project env) to print the menu and
validate the selection before anything is pulled or run.

All tags below were verified against ollama.com/library/<model>/tags.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

DEFAULT_QUANT = "q4_K_M"
DEFAULT_JUDGE_TAG = "qwen3:14b-q4_K_M"

# Model families that emit chain-of-thought unless told not to. Thinking
# tokens eat the judge's token budget and break JSON parsing, so the client
# sends think=False for these.
THINKING_FAMILIES = ("qwen3",)


@dataclass(frozen=True)
class JudgeModel:
    name: str                      # human-readable, shown in the notebook menu
    quants: Dict[str, str]         # quant level -> exact Ollama tag
    sizes_gb: Dict[str, float] = field(default_factory=dict)  # weights on disk/VRAM
    note: str = ""

    def tag(self, quant: str) -> str:
        if quant not in self.quants:
            raise ValueError(
                f"Quantization {quant!r} is not available for judge "
                f"{self.name!r}. Available: {', '.join(self.quants)}"
            )
        return self.quants[quant]


JUDGE_MODELS: List[JudgeModel] = [
    JudgeModel(  # 0
        name="Qwen3-14B",
        quants={
            "q4_K_M": "qwen3:14b-q4_K_M",
            "q8_0": "qwen3:14b-q8_0",
            "fp16": "qwen3:14b-fp16",
        },
        sizes_gb={"q4_K_M": 9.3, "q8_0": 16.0, "fp16": 30.0},
        note="Alibaba; thinking disabled automatically",
    ),
    JudgeModel(  # 1
        name="Gemma 3 12B IT",
        quants={
            "qat": "gemma3:12b-it-qat",
            "q4_K_M": "gemma3:12b-it-q4_K_M",
            "q8_0": "gemma3:12b-it-q8_0",
            "fp16": "gemma3:12b-it-fp16",
        },
        sizes_gb={"qat": 8.9, "q4_K_M": 8.2, "q8_0": 13.0, "fp16": 24.0},
        note="Google; 'qat' = quantization-aware 4-bit",
    ),
    JudgeModel(  # 2
        name="Phi-4 14B",
        quants={
            "q4_K_M": "phi4:14b-q4_K_M",
            "q8_0": "phi4:14b-q8_0",
            "fp16": "phi4:14b-fp16",
        },
        sizes_gb={"q4_K_M": 9.1, "q8_0": 16.0, "fp16": 29.0},
        note="Microsoft; STEM/clinical reasoning",
    ),
    JudgeModel(  # 3
        name="Mistral 7B Instruct v0.3",
        quants={
            "q4_K_M": "mistral:7b-instruct-v0.3-q4_K_M",
            "q8_0": "mistral:7b-instruct-v0.3-q8_0",
        },
        sizes_gb={"q4_K_M": 4.37, "q8_0": 7.7},
        note="Mistral AI; compact 4th family (smallest/fastest)",
    ),
]


def get_judge_model(index: int) -> JudgeModel:
    if isinstance(index, bool) or not isinstance(index, int):
        raise ValueError(f"JUDGE_MODEL_INDEX must be an integer, got {index!r}")
    if not 0 <= index < len(JUDGE_MODELS):
        raise ValueError(
            f"JUDGE_MODEL_INDEX={index} is out of range. Valid indices:\n"
            + format_judge_menu()
        )
    return JUDGE_MODELS[index]


def resolve_judge_tag(index: int, quant: str = DEFAULT_QUANT) -> str:
    """Exact Ollama tag for registry entry `index` at quantization `quant`."""
    return get_judge_model(index).tag((quant or DEFAULT_QUANT).strip())


def _parse_index(raw: str) -> int:
    try:
        return int(raw.strip())
    except ValueError:
        raise ValueError(
            f"JUDGE_MODEL_INDEX must be an integer, got {raw!r}. Options:\n"
            + format_judge_menu()
        ) from None


def parse_indices(raw) -> List[int]:
    """JUDGE_MODEL_INDEX value -> list of registry indices.

    Accepts an int, "1", "0,3" or [0, 1, 2, 3]; one index = one judge, several =
    a multi-judge consensus panel. Duplicates are rejected.
    """
    if isinstance(raw, bool):
        raise ValueError(f"JUDGE_MODEL_INDEX must be an integer, got {raw!r}")
    if isinstance(raw, int):
        items = [raw]
    elif isinstance(raw, (list, tuple)):
        items = list(raw)
    else:
        items = [p for p in str(raw).replace(" ", "").split(",") if p != ""]
        items = [_parse_index(p) for p in items]
    if not items:
        raise ValueError("JUDGE_MODEL_INDEX is empty. Options:\n" + format_judge_menu())
    for i in items:
        get_judge_model(i)  # validates type and range, raises with the menu
    if len(set(items)) != len(items):
        raise ValueError(f"JUDGE_MODEL_INDEX lists a judge twice: {items}")
    return items


def resolve_judge_tags(env: Optional[Dict[str, str]] = None) -> Optional[List[str]]:
    """Ollama tags of the selected judge panel, or None when no index is set."""
    env = os.environ if env is None else env
    raw_index = (env.get("JUDGE_MODEL_INDEX") or "").strip()
    if not raw_index:
        return None
    quant = (env.get("JUDGE_QUANT") or DEFAULT_QUANT).strip()
    return [resolve_judge_tag(i, quant) for i in parse_indices(raw_index)]


def resolve_judge_ollama_model(env: Optional[Dict[str, str]] = None) -> str:
    """Resolve the primary judge tag from the environment (see module doc)."""
    env = os.environ if env is None else env
    explicit = (env.get("JUDGE_OLLAMA_MODEL") or "").strip()
    if explicit:
        return explicit
    tags = resolve_judge_tags(env)
    if tags:
        return tags[0]
    return DEFAULT_JUDGE_TAG


def is_thinking_model(tag: str) -> bool:
    base = (tag or "").split("/")[-1].lower()
    return base.startswith(THINKING_FAMILIES)


def judge_slug(tag: str) -> str:
    """Filesystem-safe id for per-judge checkpoint/result file names."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in tag)


def format_judge_menu() -> str:
    lines = []
    for i, m in enumerate(JUDGE_MODELS):
        quants = ", ".join(
            f"{q}" + (f" ({m.sizes_gb[q]:g} GB)" if q in m.sizes_gb else "")
            for q in m.quants
        )
        note = f"  - {m.note}" if m.note else ""
        lines.append(f"  [{i}] {m.name}: {quants}{note}")
    return "\n".join(lines)
