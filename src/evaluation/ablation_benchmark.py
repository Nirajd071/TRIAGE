"""
Ablation Study Benchmark for HCAD-RAG.

Systematically evaluates the contribution of each architectural innovation:
1. HCAD-Full: Complete proposed architecture.
2. HCAD-w/o Decoupling: Raw sentence embeddings without semantic orthogonalization (E _|_ R _|_ M).
3. HCAD-w/o Decision Gate: Dual-stage Heaviside filtering without JEV / token pruning gate.
4. HCAD-w/o Anisotropic Tensor: Isotropic scalar cosine similarity without metric tensor projection.
5. HCAD-Dense Only: Pure dense vector search without multi-channel hyperedge expansion.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RESULTS_DIR
from src.core.hcad_retriever import HCADRetriever
from src.evaluation.base_paper_evaluator import BasePaperEvaluator

logger = logging.getLogger(__name__)


class AblationBenchmark:
    """Executes programmatic ablation sweeps over HCAD retriever configurations."""

    def __init__(
        self,
        evaluator: Optional[BasePaperEvaluator] = None,
        output_path: Optional[Path] = None,
    ):
        self.evaluator = evaluator or BasePaperEvaluator()
        self.output_path = Path(output_path or (RESULTS_DIR / "ablation_benchmark_results.json"))

        # Define the ablation arms
        self.arms: Dict[str, HCADRetriever] = {
            "HCAD-Full": HCADRetriever(
                fail_on_vector_error=False,
                enable_decoupling=True,
                enable_decision_gate=True,
                enable_aniso=True,
                enable_hypergraph=True,
            ),
            "HCAD-w/o Decoupling": HCADRetriever(
                fail_on_vector_error=False,
                enable_decoupling=False,
                enable_decision_gate=True,
                enable_aniso=True,
                enable_hypergraph=True,
            ),
            "HCAD-w/o Decision Gate": HCADRetriever(
                fail_on_vector_error=False,
                enable_decoupling=True,
                enable_decision_gate=False,
                enable_aniso=True,
                enable_hypergraph=True,
            ),
            "HCAD-w/o Anisotropic Tensor": HCADRetriever(
                fail_on_vector_error=False,
                enable_decoupling=True,
                enable_decision_gate=True,
                enable_aniso=False,
                enable_hypergraph=True,
            ),
            "HCAD-Dense Only": HCADRetriever(
                fail_on_vector_error=False,
                enable_decoupling=True,
                enable_decision_gate=True,
                enable_aniso=True,
                enable_hypergraph=False,
            ),
        }

    def run_ablation_sweep(
        self,
        samples: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Run all ablation arms over the evaluation samples."""
        logger.info("Starting ablation sweep across %d arms on %d samples...", len(self.arms), len(samples))

        scores_by_arm: Dict[str, List[Dict[str, float]]] = {arm: [] for arm in self.arms}
        latencies_by_arm: Dict[str, List[float]] = {arm: [] for arm in self.arms}
        per_sample_records = []

        for idx, sample in enumerate(samples, 1):
            q = sample["question"]
            ref = sample.get("reference", "")
            logger.info("Evaluating Sample %d/%d: %.60s...", idx, len(samples), q)

            sample_record: Dict[str, Any] = {"question": q, "reference": ref, "arms": {}}

            for arm_name, retriever in self.arms.items():
                t0 = time.perf_counter()
                contexts = retriever.retrieve(q)
                t_ret = time.perf_counter() - t0

                resp = retriever.retrieve_and_generate(q, contexts=contexts)
                score_res = self.evaluator.evaluate_scoring(
                    question=q, answer=resp, reference_document=ref, fallback_on_error=True
                )

                score_dict = score_res.to_dict()
                scores_by_arm[arm_name].append(score_dict)
                latencies_by_arm[arm_name].append(t_ret)

                sample_record["arms"][arm_name] = {
                    "score": score_dict["overall_score"],
                    "latency": round(t_ret, 3),
                    "answer_snippet": resp[:150],
                }

            per_sample_records.append(sample_record)

        # Aggregate summary metrics per arm
        summary: Dict[str, Any] = {}
        full_mean_score = (
            float(np.mean([s["overall_score"] for s in scores_by_arm["HCAD-Full"]]))
            if scores_by_arm["HCAD-Full"]
            else 0.0
        )

        for arm_name, score_list in scores_by_arm.items():
            if not score_list:
                continue
            mean_overall = float(np.mean([s["overall_score"] for s in score_list]))
            mean_comp = float(np.mean([s["comprehensiveness"] for s in score_list]))
            mean_div = float(np.mean([s["diversity"] for s in score_list]))
            mean_emp = float(np.mean([s["empowerment"] for s in score_list]))
            mean_log = float(np.mean([s["logical"] for s in score_list]))
            mean_read = float(np.mean([s["readability"] for s in score_list]))
            mean_lat = float(np.mean(latencies_by_arm[arm_name]))

            summary[arm_name] = {
                "overall_score": round(mean_overall, 2),
                "delta_vs_full": round(mean_overall - full_mean_score, 2),
                "comprehensiveness": round(mean_comp, 2),
                "diversity": round(mean_div, 2),
                "empowerment": round(mean_emp, 2),
                "logical": round(mean_log, 2),
                "readability": round(mean_read, 2),
                "mean_retrieval_latency_s": round(mean_lat, 3),
            }

        ablation_results = {
            "manifest": {
                "benchmark": "HCAD-RAG Component Ablation Study",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "num_samples": len(samples),
                "arms": list(self.arms.keys()),
            },
            "summary_by_arm": summary,
            "per_sample": per_sample_records,
        }

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(ablation_results, f, indent=2)

        logger.info("Ablation study successfully saved to %s", self.output_path)
        return ablation_results


if __name__ == "__main__":
    from src.evaluation.run_base_paper_benchmark import load_clinical_benchmark_samples
    samples = load_clinical_benchmark_samples(num_samples=3)
    runner = AblationBenchmark()
    res = runner.run_ablation_sweep(samples)
    print("\n=== ABLATION STUDY RESULTS ===")
    print(json.dumps(res["summary_by_arm"], indent=2))
