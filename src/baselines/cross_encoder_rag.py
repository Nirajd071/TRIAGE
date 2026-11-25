"""Biomedical Cross-Encoder Reranker baseline (ncbi/MedCPT-Cross-Encoder).

Retrieves a top-N candidate set via Hybrid BM25+Dense retrieval and reranks
it with a pretrained biomedical Cross-Encoder, giving a strong neural
reranker baseline for both the main RAG benchmark and polarity tests.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from src.config import VANILLA_TOP_K

logger = logging.getLogger(__name__)

DEFAULT_CROSS_ENCODER_MODEL = "ncbi/MedCPT-Cross-Encoder"


class MedCPTCrossEncoder:
    """Lazy wrapper around ncbi/MedCPT-Cross-Encoder for pairwise scoring."""

    def __init__(self, model_name: str = DEFAULT_CROSS_ENCODER_MODEL, max_length: int = 512) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self._tokenizer = None
        self._model = None
        self._device: Optional[str] = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from src.config import resolve_embed_device
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._device = resolve_embed_device()
        logger.info("Loading Cross-Encoder %s on %s...", self.model_name, self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._model.to(self._device)
        self._model.eval()

    def score_pairs(self, query: str, documents: Sequence[str], batch_size: int = 32) -> np.ndarray:
        """Score (query, document) pairs; higher logit = stronger relevance."""
        if not documents:
            return np.zeros(0, dtype=np.float32)
        self._ensure_loaded()
        import torch

        scores: List[float] = []
        pairs = [[query, str(doc or "")] for doc in documents]
        with torch.no_grad():
            for i in range(0, len(pairs), batch_size):
                batch = pairs[i : i + batch_size]
                encoded = self._tokenizer(
                    batch,
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self._device) for k, v in encoded.items()}
                logits = self._model(**encoded).logits.squeeze(-1)
                if logits.ndim == 0:
                    scores.append(float(logits.item()))
                else:
                    scores.extend(float(x) for x in logits.detach().cpu().tolist())
        return np.asarray(scores, dtype=np.float32)


class CrossEncoderRerankRAG:
    """Two-stage retriever: Hybrid candidate generation followed by Cross-Encoder reranking."""

    def __init__(
        self,
        candidate_fn: Callable[[str, int], List[Dict[str, Any]]],
        cross_encoder: Optional[MedCPTCrossEncoder] = None,
        candidate_k: int = 20,
        top_k: int = VANILLA_TOP_K,
    ) -> None:
        self.candidate_fn = candidate_fn
        self.cross_encoder = cross_encoder or MedCPTCrossEncoder()
        self.candidate_k = candidate_k
        self.top_k = top_k

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        k = top_k or self.top_k
        candidates = self.candidate_fn(query, max(k, self.candidate_k))
        if not candidates:
            return []
        docs = [
            str(c.get("text") or c.get("document") or "")
            for c in candidates
        ]
        scores = self.cross_encoder.score_pairs(query, docs)
        order = np.argsort(-scores)[:k]
        out: List[Dict[str, Any]] = []
        for idx in order:
            item = dict(candidates[int(idx)])
            item["rerank_score"] = float(scores[int(idx)])
            item["score"] = float(scores[int(idx)])
            out.append(item)
        return out
