import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.evaluation.hallucination_evaluator import HallucinationEvaluator

class TestHallucinationEvaluator(unittest.TestCase):
    def test_scores_faithful_and_hallucinated_answers_as_paired_examples(self):
        evaluator = HallucinationEvaluator()
        evaluator._data = [
            {
                "Question": "What does the evidence show?",
                "Knowledge": ["Evidence one.", "Evidence two."],
                "Ground Truth": "A faithful answer.",
                "Hallucinated Answer": "An unsupported answer.",
                "Difficulty Level": "easy",
            }
        ]
        ollama = types.ModuleType("ollama")
        ollama.chat = MagicMock(
            side_effect=[
                SimpleNamespace(
                    message=SimpleNamespace(content="FAITHFUL")
                ),
                SimpleNamespace(
                    message=SimpleNamespace(content="HALLUCINATED")
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(
            sys.modules, {"ollama": ollama}
        ), patch(
            "src.evaluation.hallucination_evaluator.TABLES_DIR", Path(temp_dir)
        ), patch(
            "src.evaluation.hallucination_evaluator.HALLU_DIR", Path(temp_dir) / "h"
        ), patch(
            "src.wandb_tracking.WANDB_MODE", "disabled"
        ):
            result = evaluator.evaluate(
                SimpleNamespace(model="test-model"),
                num_samples=None,
                system_name="test",
            )

        self.assertEqual(result["num_samples"], 1)
        self.assertEqual(result["num_classifications"], 2)
        self.assertEqual(
            result["overall"],
            {
                "precision": 1.0,
                "recall": 1.0,
                "specificity": 1.0,
                "f1": 1.0,
                "accuracy": 1.0,
                "tp": 1,
                "fp": 0,
                "tn": 1,
                "fn": 0,
            },
        )
        prompt = ollama.chat.call_args_list[0].kwargs["messages"][1]["content"]
        self.assertIn("Evidence one.\nEvidence two.", prompt)
        self.assertFalse(any(key in result for key in ("context_quality", "hallucination_probability")))

    def test_invalid_model_output_is_not_counted_as_a_prediction(self):
        evaluator = HallucinationEvaluator()
        evaluator._data = [
            {
                "Question": "Question",
                "Knowledge": ["Knowledge"],
                "Ground Truth": "Faithful answer",
                "Hallucinated Answer": "False answer",
                "Difficulty Level": "easy",
            }
        ]
        ollama = types.ModuleType("ollama")
        ollama.chat = MagicMock(
            return_value=SimpleNamespace(
                message=SimpleNamespace(content="uncertain")
            )
        )

        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(
            sys.modules, {"ollama": ollama}
        ), patch(
            "src.evaluation.hallucination_evaluator.TABLES_DIR", Path(temp_dir)
        ), patch(
            "src.evaluation.hallucination_evaluator.HALLU_DIR", Path(temp_dir) / "h"
        ), patch(
            "src.wandb_tracking.WANDB_MODE", "disabled"
        ):
            # An unparseable label must not abort the run (which previously
            # left NO output file); it is counted as invalid, never as a
            # prediction, and the result is still saved.
            result = evaluator.evaluate(SimpleNamespace(model="test-model"), num_samples=None)
            self.assertTrue(Path(result["output_path"]).exists())
        overall = result["overall"]
        self.assertEqual(overall["n_invalid"], 2)
        self.assertEqual((overall["tp"], overall["fp"], overall["tn"], overall["fn"]), (0, 0, 0, 0))

    def test_system_answers_mode_reports_hallucination_rate_not_f1(self):
        evaluator = HallucinationEvaluator()
        evaluator._data = [
            {"Question": f"Q{i}", "Knowledge": ["K"], "Ground Truth": "G",
             "Hallucinated Answer": "H", "Difficulty Level": lvl}
            for i, lvl in enumerate(["easy", "medium", "hard", "hard"])
        ]
        ollama = types.ModuleType("ollama")
        replies = iter(["HALLUCINATED", "FAITHFUL", "Faithful.", "HALLUCINATED"])
        ollama.chat = MagicMock(side_effect=lambda **kw: SimpleNamespace(
            message=SimpleNamespace(content=next(replies))))
        system = SimpleNamespace(retrieve_and_generate=lambda q: f"answer to {q}")
        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(
            sys.modules, {"ollama": ollama}
        ), patch(
            "src.evaluation.hallucination_evaluator.TABLES_DIR", Path(temp_dir)
        ), patch(
            "src.evaluation.hallucination_evaluator.HALLU_DIR", Path(temp_dir) / "h"
        ), patch("src.wandb_tracking.WANDB_MODE", "disabled"):
            result = evaluator.evaluate(system, num_samples=None, system_name="HCAD-RAG",
                                        classifier_model="clf")
        self.assertEqual(result["evaluation_mode"], "system_answers")
        self.assertEqual(result["overall"]["hallucination_rate"], 0.5)
        self.assertNotIn("f1", result["overall"])
        self.assertEqual(result["hard"]["n_hallucinated"], 1)
        self.assertEqual(ollama.chat.call_args.kwargs["model"], "clf")

    def test_closed_book_label_with_slash_sanitizes_filename(self):
        from src.config import ARM_LABELS

        evaluator = HallucinationEvaluator()
        evaluator._data = [
            {"Question": "Q1", "Knowledge": ["K"], "Ground Truth": "G",
             "Hallucinated Answer": "H", "Difficulty Level": "easy"}
        ]
        ollama = types.ModuleType("ollama")
        ollama.chat = MagicMock(return_value={"message": {"content": "FAITHFUL"}})
        system = SimpleNamespace(retrieve_and_generate=lambda q: "answer")
        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(
            sys.modules, {"ollama": ollama}
        ), patch(
            "src.evaluation.hallucination_evaluator.TABLES_DIR", Path(temp_dir)
        ), patch(
            "src.evaluation.hallucination_evaluator.HALLU_DIR", Path(temp_dir) / "h"
        ), patch("src.wandb_tracking.WANDB_MODE", "disabled"):
            result = evaluator.evaluate(
                system, num_samples=None, system_name=ARM_LABELS["closed_book"], classifier_model="clf"
            )
            self.assertTrue(Path(result["output_path"]).exists())
            self.assertNotIn("/_", result["output_path"])

if __name__ == "__main__":
    unittest.main()
