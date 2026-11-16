"""Single source of truth for the HCAD-RAG 5-stage methodology.

The same stage list drives three things, so they cannot drift apart:
  1. per-stage retrieval timings recorded by ``HCADRetriever.retrieve``
     (``_last_stage_timings`` keys == ``timing_key`` below),
  2. the latency breakdown written to ``results/latency_breakdown.json``,
  3. the publication methodology diagram rendered by
     ``src/evaluation/generate_figures.py`` (``methodology_pipeline.png``).

Stage 5 spans retrieval (the decision gate) and generation; its
generation part is timed separately by the benchmark (``t_generation``)
so retrieval latency is never confounded with decode length.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineStage:
    number: int
    name: str
    timing_key: str          # key in HCADRetriever._last_stage_timings
    modules: tuple[str, ...]  # implementing modules (for the diagram / paper)
    inputs: str
    outputs: str
    summary: str


PIPELINE_STAGES: tuple[PipelineStage, ...] = (
    PipelineStage(
        1, "Query Understanding", "query_understanding",
        ("core/ner_extractor.py", "core/relation_extractor.py",
         "core/entity_linker.py"),
        "clinical query q",
        "entities, relation cues, modifiers",
        "SciSpaCy NER + rule relation/negation extraction; SapBERT linking to PrimeKG",
    ),
    PipelineStage(
        2, "Candidate Pool Construction", "candidate_pool",
        ("data/build_vector_store.py", "core/hypergraph_builder.py",
         "core/fact_hypergraph.py"),
        "entities + relations",
        "candidate pool C (<= 80 items)",
        "dense vector search + relation-typed hyperedge expansion over PrimeKG",
    ),
    PipelineStage(
        3, "Semantic Decoupling", "semantic_decoupling",
        ("core/semantic_decoupler.py", "core/decoupler_cache.py"),
        "q and each c in C",
        "3 x 768 [E; R; M] matrices + presence masks",
        "entity / relation / modifier channels, soft Gram-Schmidt (eta = 0.70)",
    ),
    PipelineStage(
        4, "Hybrid Scoring & Adaptive Filtering", "scoring_and_filter",
        ("core/anisotropic_transformer.py", "core/statistical_filter.py"),
        "decoupled matrices",
        "filtered evidence (Heaviside gate)",
        "Phi = w_d cos(q,c) + (1-w_d) Phi_struct; Theta(Phi - (mu + k sigma))",
    ),
    PipelineStage(
        5, "Decision Gate & Grounded Generation", "decision_gate",
        ("core/decision_gate.py", "core/prompt_builder.py", "core/llm_client.py"),
        "filtered evidence",
        "cited clinical answer",
        "JEV groundedness gate, numbered evidence prompt, Llama-3.1-8B-Instruct",
    ),
)

STAGE_TIMING_KEYS: tuple[str, ...] = tuple(s.timing_key for s in PIPELINE_STAGES)
