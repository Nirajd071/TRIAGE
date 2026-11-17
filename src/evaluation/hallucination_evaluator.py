"""
MedHallu Hallucination Detection Evaluator.

Evaluates hallucination detection on MedHallu benchmark (10K QA pairs
from PubMedQA with controlled hallucinations at easy/medium/hard tiers).

Two modes:
  detector        the classifier LLM labels the dataset's fixed ground-truth
                  and hallucinated answers -> precision / recall /
                  specificity / F1 / accuracy (a property of the CLASSIFIER).
  system_answers  the system under test (e.g. HCAD-RAG) GENERATES an answer
                  to every question, which the classifier labels ->
                  hallucination RATE (fraction labelled HALLUCINATED). F1 is
                  undefined here: generated answers have no gold label.

Both modes are stratified by difficulty (easy / medium / hard), checkpoint
every sample (resume-safe) and ALWAYS write
``results/hallucination/medhallu_<system>.json``: a classifier reply that is
neither HALLUCINATED nor FAITHFUL is counted as ``n_invalid`` instead of
aborting the run (an aborted run previously left no output file at all).

CLI:  python -m src.evaluation.hallucination_evaluator --system hcad --n 200
"""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import (
    DEFAULT_EVAL_SAMPLES,
    FIGURES_DIR,
    RESULTS_DIR,
    LLM_MODEL,
    MEDHALLU_CONFIG_LABELED,
    MEDHALLU_DATASET_ID,
    RANDOM_SEED,
    RAW_DATA_DIR,
    TABLES_DIR,
)
from src.wandb_tracking import log_hallucination_results, tracked_run

logger = logging.getLogger(__name__)

HALLU_DIR = RESULTS_DIR / "hallucination"
LABELS = ("HALLUCINATED", "FAITHFUL")


def parse_hallucination_label(text: str):
    """Map a classifier reply to True (hallucinated) / False / None (invalid)."""
    import re
    words = re.findall(r"[A-Z]+", (text or "").upper())
    if words and words[0] in LABELS:
        return words[0] == "HALLUCINATED"
    hits = {w for w in words if w in LABELS}
    if len(hits) == 1:
        return hits.pop() == "HALLUCINATED"
    return None

class HallucinationEvaluator:
    """
    Evaluates hallucination detection on the MedHallu benchmark.
    """

    DIFFICULTY_LEVELS = ["easy", "medium", "hard"]

    def __init__(self):
        self._data: Optional[list] = None

    # Data Loading

    def load_data(self, config: str = MEDHALLU_CONFIG_LABELED) -> list[dict]:
        """Load MedHallu dataset."""
        if self._data is not None:
            return self._data

        dataset_dir = RAW_DATA_DIR / "medhallu"
        legacy_path = dataset_dir / f"{config}.json"
        split_path = dataset_dir / config / "train.json"
        for save_path in (split_path, legacy_path):
            if save_path.exists():
                with open(save_path, encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, list):
                    raise ValueError(f"Expected a list of MedHallu rows in {save_path}")
                self._data = data
                logger.info("Loaded %d MedHallu samples from %s", len(data), save_path)
                return self._data

        logger.info("Downloading MedHallu (%s) from HuggingFace...", config)
        try:
            from datasets import load_dataset

            ds = load_dataset(MEDHALLU_DATASET_ID, config)
            split = "train" if "train" in ds else next(iter(ds))
            self._data = [dict(row) for row in ds[split]]
            split_path.parent.mkdir(parents=True, exist_ok=True)
            with open(split_path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2)
            logger.info("Saved %d MedHallu samples to %s", len(self._data), split_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to load MedHallu ({config}): {exc}") from exc

        return self._data

    # Hallucination Detection

    @tracked_run(
        name="medhallu-evaluation",
        job_type="evaluation",
        config=lambda self, system, num_samples=DEFAULT_EVAL_SAMPLES, seed=RANDOM_SEED, system_name="system", **_kw: {
            "system": system_name,
            "samples": num_samples,
            "seed": seed,
        },
        log_results=log_hallucination_results,
    )
    def evaluate(
        self,
        system,
        num_samples: Optional[int] = DEFAULT_EVAL_SAMPLES,
        seed: int = RANDOM_SEED,
        system_name: str = "system",
        mode: Optional[str] = None,
        resume: bool = True,
        classifier_model: Optional[str] = None,
    ) -> dict:
        """
        Evaluate hallucination detection on MedHallu.

        The system must have a `answer_mcq` or `retrieve_and_generate` method.
        We use a separate hallucination-detection prompt.

        Args:
            system: System with `answer_mcq(question, options)` or
                    `retrieve_and_generate(query)` method.
            num_samples: Number of samples (None = all).
            seed: Random seed.
            system_name: Label for output.

        Returns:
            Dict with precision, recall, f1 (overall + per difficulty).
        """
        import random
        from src.core.prompt_builder import PromptBuilder

        data = self.load_data()
        if not data:
            logger.warning("MedHallu data empty. Skipping hallucination evaluation.")
            return {"error": "No data available"}

        if num_samples is not None and num_samples <= 0:
            raise ValueError("num_samples must be positive or None")
        if num_samples is not None and num_samples < len(data):
            data = random.Random(seed).sample(data, num_samples)

        builder = PromptBuilder()
        model_name = (
            classifier_model
            or getattr(system, "model_name", None)
            or getattr(getattr(system, "llm", None), "model", None)
            or getattr(system, "model", None)
            or LLM_MODEL
        )
        # the system under test was previously
        # NEVER invoked - the evaluator classified fixed gt/hallucinated
        # answers with a raw ollama.chat() call, so every system sharing a
        # base LLM received identical scores. Two modes now exist:
        #   - "system_answers": the system generates answers to each question
        #     (through its retrieval pipeline); each generated answer is then
        #     classified - this measures RAG-mediated hallucination.
        #   - "detector": legacy paired classification of gt/hallu answers -
        #     this measures only the classifier LLM, and is labeled as such.
        rag_capable = hasattr(system, "retrieve_and_generate") and callable(
            getattr(system, "retrieve_and_generate")
        )
        eval_mode = mode or ("system_answers" if rag_capable else "detector")
        if eval_mode not in ("system_answers", "detector"):
            raise ValueError(f"Unknown MedHallu mode {eval_mode!r}")
        if eval_mode == "system_answers" and not rag_capable:
            raise ValueError("system_answers mode needs a system with retrieve_and_generate()")
        import ollama

        results_by_difficulty: dict[str, dict] = {
            level: {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "n_invalid": 0}
            for level in self.DIFFICULTY_LEVELS + ["all"]
        }
        safe_name = system_name.replace(" ", "_").lower()
        HALLU_DIR.mkdir(parents=True, exist_ok=True)
        ckpt_path = HALLU_DIR / f"medhallu_{safe_name}_{eval_mode}_checkpoint.jsonl"
        done: dict[int, dict] = {}
        if resume and ckpt_path.exists():
            for line in ckpt_path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (row.get("seed") == seed and row.get("n_requested") == num_samples
                        and row.get("classifier_model") == model_name):
                    done[row["sample_index"]] = row
        ckpt = open(ckpt_path, "a", encoding="utf-8")

        def classify(question: str, answer: str, knowledge: str, sample_index: int) -> bool:
            messages = builder.build_hallucination_prompt(question, answer, knowledge)
            try:
                response = ollama.chat(
                    model=model_name,
                    messages=messages,
                    options={"temperature": 0.0, "num_predict": 8},
                )
                prediction = response.message.content.strip().upper().rstrip(".!")
            except Exception as exc:
                raise RuntimeError(
                    f"MedHallu classifier failed for sample {sample_index}: {exc}"
                ) from exc

            label = parse_hallucination_label(prediction)
            if label is None:
                logger.warning("MedHallu classifier returned an invalid label for "
                               "sample %d: %r (counted as invalid)", sample_index, prediction)
            return label

        for sample_index, item in enumerate(data):
            question = item.get("Question", item.get("question", ""))
            knowledge_value = item.get("Knowledge", item.get("knowledge", ""))
            if isinstance(knowledge_value, (list, tuple)):
                knowledge = "\n".join(str(part) for part in knowledge_value if part)
            else:
                knowledge = str(knowledge_value or "")
            gt_answer = item.get("Ground Truth", item.get("ground_truth", ""))
            hallu_answer = item.get(
                "Hallucinated Answer", item.get("hallucinated_answer", "")
            )
            difficulty = str(
                item.get("Difficulty Level", item.get("difficulty_level", "medium"))
                or "medium"
            ).lower().strip()
            if not question or not knowledge or not gt_answer or not hallu_answer:
                raise ValueError(
                    f"MedHallu sample {sample_index} is missing question, knowledge, "
                    "ground-truth answer, or hallucinated answer"
                )
            if difficulty not in self.DIFFICULTY_LEVELS:
                raise ValueError(
                    f"MedHallu sample {sample_index} has unknown difficulty "
                    f"{difficulty!r}"
                )

            if sample_index in done:
                row = done[sample_index]
            elif eval_mode == "system_answers":
                # Measure the RAG pipeline: generate, then classify.
                try:
                    generated = system.retrieve_and_generate(question)
                except Exception as exc:
                    logger.warning("System generation failed for MedHallu sample %d: %s",
                                   sample_index, exc)
                    generated = ""
                if not isinstance(generated, str) or not generated.strip() \
                        or generated.strip().startswith("[Error"):
                    predictions = [None]
                else:
                    predictions = [classify(question, generated, knowledge, sample_index)]
                row = {"sample_index": sample_index, "difficulty": difficulty,
                       "predictions": predictions, "actual": [None]}
            else:
                predictions = [classify(question, str(gt_answer), knowledge, sample_index),
                               classify(question, str(hallu_answer), knowledge, sample_index)]
                row = {"sample_index": sample_index, "difficulty": difficulty,
                       "predictions": predictions, "actual": [False, True]}
            if sample_index not in done:
                ckpt.write(json.dumps({**row, "seed": seed, "n_requested": num_samples,
                                       "classifier_model": model_name}) + "\n")
                ckpt.flush()

            for predicted_hallu, actual_hallu in zip(row["predictions"], row["actual"]):
                for level_key in ("all", difficulty):
                    counts = results_by_difficulty[level_key]
                    if predicted_hallu is None:
                        counts["n_invalid"] += 1
                    elif actual_hallu is None:
                        # system_answers: no gold label; tp = labelled hallucinated
                        counts["tp" if predicted_hallu else "tn"] += 1
                    else:
                        if actual_hallu and predicted_hallu:
                            counts["tp"] += 1
                        elif not actual_hallu and predicted_hallu:
                            counts["fp"] += 1
                        elif not actual_hallu and not predicted_hallu:
                            counts["tn"] += 1
                        else:
                            counts["fn"] += 1
        ckpt.close()

        # Compute classification metrics for each difficulty.
        def _f1_from_counts(counts: dict) -> dict:
            if eval_mode == "system_answers":
                n_valid = counts["tp"] + counts["tn"]
                return {
                    "n": n_valid + counts["n_invalid"],
                    "n_hallucinated": counts["tp"],
                    "n_faithful": counts["tn"],
                    "n_invalid": counts["n_invalid"],
                    "hallucination_rate": round(counts["tp"] / n_valid, 4) if n_valid else None,
                }
            tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = (
                counts["tn"] / (counts["tn"] + fp)
                if (counts["tn"] + fp) > 0
                else 0.0
            )
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            total = tp + fp + counts["tn"] + fn
            accuracy = (tp + counts["tn"]) / total if total else 0.0
            out = {
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "specificity": round(specificity, 4),
                "f1": round(f1, 4),
                "accuracy": round(accuracy, 4),
                **{k: counts[k] for k in ("tp", "fp", "tn", "fn")},
            }
            if counts["n_invalid"]:
                out["n_invalid"] = counts["n_invalid"]
            return out

        n_classifications = len(data) if eval_mode == "system_answers" else 2 * len(data)
        output = {
            "system": system_name,
            "evaluation_mode": eval_mode,  # mode is now explicit
            "num_samples": len(data),
            "num_classifications": n_classifications,
            "classifier_model": model_name,
            "note": ("system_answers mode classifies SYSTEM-GENERATED answers "
                     "(RAG-mediated hallucination measurement).")
                    if eval_mode == "system_answers" else
                    ("detector mode measures only the classifier LLM on fixed "
                     "gt/hallucinated pairs - it does NOT evaluate the RAG pipeline."),
            "overall": _f1_from_counts(results_by_difficulty["all"]),
        }
        for level in self.DIFFICULTY_LEVELS:
            output[level] = _f1_from_counts(results_by_difficulty[level])

        key = "hallucination_rate" if eval_mode == "system_answers" else "f1"
        logger.info("MedHallu '%s' (%s): overall %s=%s, easy=%s, medium=%s, hard=%s",
                    system_name, eval_mode, key, output["overall"].get(key),
                    output["easy"].get(key), output["medium"].get(key), output["hard"].get(key))

        out_path = HALLU_DIR / f"medhallu_{safe_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        # legacy location consumed by `main.py base-paper --plot`
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        with open(TABLES_DIR / f"hallucination_{safe_name}.json", "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        output["output_path"] = str(out_path)

        return output


def main(argv=None) -> None:
    """CLI: run MedHallu for one system and save JSON + figure."""
    import argparse
    from types import SimpleNamespace

    parser = argparse.ArgumentParser(description="MedHallu hallucination evaluation")
    parser.add_argument("--system", default="detector",
                        choices=["detector", "closed_book", "dense", "hcad"],
                        help="'detector' scores the classifier only; the others "
                             "generate answers through that system")
    parser.add_argument("--n", type=int, default=DEFAULT_EVAL_SAMPLES)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--classifier-model", default=LLM_MODEL,
                        help="ollama model used to label answers")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    from src.config import ARM_LABELS

    if args.system == "detector":
        system, mode = SimpleNamespace(model=args.classifier_model), "detector"
        name = f"detector_{args.classifier_model.replace(':', '-')}"
    else:
        mode, name = "system_answers", ARM_LABELS[args.system]
        if args.system == "hcad":
            from src.core.hcad_retriever import HCADRetriever
            system = HCADRetriever()
        elif args.system == "dense":
            from src.baselines.vanilla_vector_rag import VanillaVectorRAG
            system = VanillaVectorRAG()
        else:
            from src.baselines.baseline_llm import BaselineLLM
            base = BaselineLLM()
            system = SimpleNamespace(retrieve_and_generate=base.generate)
    result = HallucinationEvaluator().evaluate(system, num_samples=args.n, seed=args.seed,
                                               system_name=name, mode=mode,
                                               classifier_model=args.classifier_model)
    from src.evaluation.plots import plot_hallucination_tiers
    fig = plot_hallucination_tiers(Path(result["output_path"]), FIGURES_DIR)
    print(f"saved {result['output_path']}" + (f" and {fig}" if fig else ""))


if __name__ == "__main__":
    main()
