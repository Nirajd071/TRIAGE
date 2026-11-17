"""
BM25 Sparse Lexical Retriever and Hybrid (BM25 + Dense) RAG Baselines.

Provides:
1. BM25RAG: Standard sparse retrieval using BM25Okapi over the biomedical corpus.
2. HybridRAG: Weighted convex combination of dense vector similarity (PubMedBERT)
   and sparse lexical matching (BM25Okapi).

Adheres strictly to the identical EVIDENCE_WORD_BUDGET (350 words) and sentence-boundary
clipping as VanillaVectorRAG and HCADRetriever for fair benchmarking.
"""

import json
import logging
import os
import pickle
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
try:
    from rank_bm25 import BM25Okapi
    HAS_RANK_BM25 = True
except ImportError:
    HAS_RANK_BM25 = False

from src.config import (
    BM25_B,
    BM25_INDEX_DIR,
    BM25_INDEX_PATH,
    BM25_K1,
    BM25_SYSTEM_NAME,
    BM25_TOP_K,
    EVIDENCE_WORD_BUDGET,
    HYBRID_BM25_WEIGHT,
    HYBRID_SYSTEM_NAME,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_SYSTEM_PROMPT,
    LLM_TEMPERATURE,
)
from src.core.llm_client import UnifiedLLMClient
from src.core.prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)

_STOPWORDS = frozenset(
    "the a an and or of to in for with on by is are was were be been this that "
    "these those it its as at from may can which who whom what when where how "
    "not no than then also such into during before after above below over under "
    "between within without about against each other more most less least any "
    "all some both few many much very own same so if while because through "
    "their there here they them we you he she his her our your my me us him".split()
)


def tokenize_corpus_text(text: str) -> List[str]:
    """Tokenize biomedical text into lowercased alphanumeric tokens, removing stopwords."""
    if not text:
        return []
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if len(t) >= 2 and t not in _STOPWORDS]


class BM25RAG:
    """Standard BM25Okapi lexical retrieval baseline."""

    def __init__(
        self,
        index_path: Optional[Path] = None,
        k1: float = BM25_K1,
        b: float = BM25_B,
        model: Optional[str] = None,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
    ):
        self.index_path = Path(index_path or BM25_INDEX_PATH)
        self.k1 = k1
        self.b = b
        self.model_name = model or LLM_MODEL
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_name = BM25_SYSTEM_NAME

        self.corpus_docs: List[Dict[str, Any]] = []
        self.bm25: Optional[Any] = None
        self._llm: Optional[UnifiedLLMClient] = None
        self._prompt_builder = PromptBuilder()

        self._load_or_build_index()

    @property
    def llm(self) -> UnifiedLLMClient:
        if self._llm is None:
            self._llm = UnifiedLLMClient(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        return self._llm

    def _load_or_build_index(self):
        """Load pickled BM25 index and corpus if available, otherwise build from ChromaDB."""
        if self.index_path.exists():
            try:
                logger.info("Loading persisted BM25 index from %s...", self.index_path)
                with open(self.index_path, "rb") as f:
                    data = pickle.load(f)
                    self.corpus_docs = data["corpus_docs"]
                    self.bm25 = data["bm25"]
                logger.info("BM25 index loaded successfully with %d documents.", len(self.corpus_docs))
                return
            except Exception as e:
                logger.warning("Failed to load BM25 index from %s (%s). Rebuilding...", self.index_path, e)

        # Build index from vector store collection or fallback corpus
        self._build_index_from_store()

        # Try direct sqlite3 extraction for fast ingestion without SQL variable limits
        sqlite_db = Path("data/vectordb/chroma.sqlite3")
        if sqlite_db.exists():
            try:
                import sqlite3
                conn = sqlite3.connect(sqlite_db)
                cur = conn.cursor()
                cur.execute("SELECT id, string_value FROM embedding_metadata WHERE key = 'chroma:document'")
                rows = cur.fetchall()
                if rows:
                    logger.info("Ingesting %d docs from Chroma SQLite store for BM25...", len(rows))
                    corpus_docs = []
                    tokenized_corpus = []
                    for doc_id, text in rows:
                        if not text:
                            continue
                        tokens = tokenize_corpus_text(text)
                        if not tokens:
                            continue
                        corpus_docs.append({"id": str(doc_id), "document": text, "metadata": {}})
                        tokenized_corpus.append(tokens)
                    if tokenized_corpus:
                        if HAS_RANK_BM25:
                            self.bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
                        else:
                            self.bm25 = _SimpleBM25(tokenized_corpus, k1=self.k1, b=self.b)
                        self.corpus_docs = corpus_docs
                        self._save_index()
                        logger.info("BM25 index successfully built with %d documents via direct SQLite.", len(self.corpus_docs))
                        return
            except Exception as e:
                logger.warning("Direct SQLite read failed: %s. Falling back to ChromaDB API...", e)

        try:
            collection = load_vector_store()
            total = collection.count()
            if total == 0:
                logger.warning("ChromaDB collection is empty. Initializing BM25 with empty index.")
                return

            logger.info("Building BM25 index from ChromaDB (%d documents)...", total)
            batch_size = 2000
            corpus_docs = []
            tokenized_corpus = []

            for offset in range(0, total, batch_size):
                chunk = collection.get(
                    limit=batch_size,
                    offset=offset,
                    include=["documents", "metadatas"],
                )
                docs = chunk.get("documents") or []
                metas = chunk.get("metadatas") or [{}] * len(docs)
                ids = chunk.get("ids") or [str(offset + i) for i in range(len(docs))]

                for doc, meta, doc_id in zip(docs, metas, ids):
                    if not doc:
                        continue
                    tokens = tokenize_corpus_text(doc)
                    if not tokens:
                        continue
                    corpus_docs.append({
                        "id": doc_id,
                        "document": doc,
                        "metadata": meta or {},
                    })
                    tokenized_corpus.append(tokens)

            if tokenized_corpus:
                if HAS_RANK_BM25:
                    self.bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
                else:
                    self.bm25 = _SimpleBM25(tokenized_corpus, k1=self.k1, b=self.b)
                self.corpus_docs = corpus_docs
                self._save_index()
                logger.info("BM25 index successfully built with %d documents.", len(self.corpus_docs))
        except Exception as e:
            logger.warning("Could not build BM25 index from vector store: %s", e)

    def _save_index(self):
        """Persist BM25 index and corpus to disk."""
        try:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.index_path, "wb") as f:
                pickle.dump({"corpus_docs": self.corpus_docs, "bm25": self.bm25}, f)
            logger.info("BM25 index persisted to %s", self.index_path)
        except Exception as e:
            logger.warning("Failed to persist BM25 index: %s", e)

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve top-K passages by BM25Okapi lexical score."""
        target_k = k if k is not None else (top_k if top_k is not None else BM25_TOP_K)
        if not self.bm25 or not self.corpus_docs:
            logger.warning("BM25 index is uninitialized; returning empty retrieval.")
            return []

        tokens = tokenize_corpus_text(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:target_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0.0:
                continue
            doc_entry = self.corpus_docs[idx]
            doc_text = doc_entry["document"]
            results.append({
                "id": doc_entry["id"],
                "text": doc_text,
                "document": doc_text,
                "metadata": doc_entry["metadata"],
                "score": score,
                "retriever": "bm25",
            })
        return results

    def _build_rag_prompt(self, query: str, contexts: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Format retrieved contexts respecting EVIDENCE_WORD_BUDGET at sentence boundaries."""
        evidence_lines = []
        remaining_words = EVIDENCE_WORD_BUDGET

        for i, ctx in enumerate(contexts, 1):
            text = ctx.get("document", "")
            words = text.split()
            if not words or remaining_words <= 0:
                break
            budget_slice = " ".join(words[:remaining_words])
            cut = max(budget_slice.rfind(". "), budget_slice.rfind("! "), budget_slice.rfind("? "))
            selected = budget_slice if cut == -1 else budget_slice[: cut + 1]
            evidence_lines.append(f"[Doc {i}] {selected.strip()}")
            remaining_words -= len(selected.split())

        formatted_evidence = "\n".join(evidence_lines) or "No relevant evidence retrieved."
        from src.core.prompt_builder import synthesis_instructions
        user_content = (
            f"Clinical Query:\n{query}\n\n"
            f"Retrieved Evidence:\n{formatted_evidence}\n\n"
            f"{synthesis_instructions(formatted_evidence)}"
        )
        return [
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def retrieve_and_generate(
        self,
        query: str,
        contexts: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """End-to-end BM25 retrieval and answer generation."""
        if contexts is None:
            contexts = self.retrieve(query)
        messages = self._build_rag_prompt(query, contexts)
        try:
            return self.llm.chat(messages)
        except Exception as exc:
            logger.error("BM25RAG generation failed: %s", exc)
            return f"[Error: {exc}]"


class HybridRAG:
    """Hybrid BM25 + Dense Vector RAG Baseline.

    Combines dense semantic vector retrieval (PubMedBERT) and sparse lexical matching (BM25Okapi)
    via normalized convex score fusion: S_hybrid = alpha * S_dense + (1 - alpha) * S_bm25.
    """

    def __init__(
        self,
        dense_weight: float = 1.0 - HYBRID_BM25_WEIGHT,
        bm25_weight: float = HYBRID_BM25_WEIGHT,
        top_k: int = BM25_TOP_K,
    ):
        from src.baselines.vanilla_vector_rag import VanillaVectorRAG

        self.dense_rag = VanillaVectorRAG()
        self.bm25_rag = BM25RAG()
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.top_k = top_k
        self.system_name = HYBRID_SYSTEM_NAME

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve candidates from both dense and sparse channels, fuse scores, and return top-K."""
        target_k = k if k is not None else (top_k if top_k is not None else self.top_k)
        dense_results = self.dense_rag.retrieve(query, top_k=target_k * 2)
        bm25_results = self.bm25_rag.retrieve(query, top_k=target_k * 2)

        # Min-max normalize scores for fair fusion
        def _normalize(scores: List[float]) -> List[float]:
            if not scores:
                return []
            s_min, s_max = min(scores), max(scores)
            if s_max == s_min:
                return [1.0] * len(scores)
            return [(s - s_min) / (s_max - s_min) for s in scores]

        dense_scores_norm = _normalize([c.get("score", 0.0) for c in dense_results])
        bm25_scores_norm = _normalize([c.get("score", 0.0) for c in bm25_results])

        doc_map: Dict[str, Dict[str, Any]] = {}

        for c, s_norm in zip(dense_results, dense_scores_norm):
            doc_text = c.get("text") or c.get("document", "")
            doc_id = c.get("id") or str(hash(doc_text))
            doc_map[doc_id] = {
                "id": doc_id,
                "text": doc_text,
                "document": doc_text,
                "metadata": c.get("metadata", {}),
                "dense_score": s_norm,
                "bm25_score": 0.0,
            }

        for c, s_norm in zip(bm25_results, bm25_scores_norm):
            doc_text = c.get("text") or c.get("document", "")
            doc_id = c.get("id") or str(hash(doc_text))
            if doc_id in doc_map:
                doc_map[doc_id]["bm25_score"] = s_norm
            else:
                doc_map[doc_id] = {
                    "id": doc_id,
                    "text": doc_text,
                    "document": doc_text,
                    "metadata": c.get("metadata", {}),
                    "dense_score": 0.0,
                    "bm25_score": s_norm,
                }

        # Calculate hybrid fused score
        for entry in doc_map.values():
            entry["score"] = (
                self.dense_weight * entry["dense_score"]
                + self.bm25_weight * entry["bm25_score"]
            )
            entry["retriever"] = "hybrid_bm25_dense"

        sorted_candidates = sorted(doc_map.values(), key=lambda x: x["score"], reverse=True)
        return sorted_candidates[:target_k]

    def retrieve_and_generate(
        self,
        query: str,
        contexts: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """End-to-end hybrid retrieval and answer generation."""
        if contexts is None:
            contexts = self.retrieve(query)
        return self.dense_rag.retrieve_and_generate(query, contexts=contexts)


class _SimpleBM25:
    """Lightweight fallback BM25Okapi implementation if rank_bm25 is unavailable."""

    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.avgdl = sum(len(doc) for doc in corpus) / max(1, self.corpus_size)
        self.doc_lens = np.array([len(doc) for doc in corpus], dtype=np.float32)

        # Term document frequencies
        self.df: Dict[str, int] = {}
        self.doc_freqs: List[Dict[str, int]] = []
        for doc in corpus:
            freqs: Dict[str, int] = {}
            for token in doc:
                freqs[token] = freqs.get(token, 0) + 1
            self.doc_freqs.append(freqs)
            for token in freqs:
                self.df[token] = self.df.get(token, 0) + 1

        # IDF
        self.idf: Dict[str, float] = {}
        for token, df in self.df.items():
            self.idf[token] = float(np.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1.0))

    def get_scores(self, query_tokens: List[str]) -> np.ndarray:
        scores = np.zeros(self.corpus_size, dtype=np.float32)
        for token in query_tokens:
            if token not in self.idf:
                continue
            idf_val = self.idf[token]
            for i, freqs in enumerate(self.doc_freqs):
                if token in freqs:
                    tf = freqs[token]
                    numerator = tf * (self.k1 + 1.0)
                    denominator = tf + self.k1 * (1.0 - self.b + self.b * (self.doc_lens[i] / self.avgdl))
                    scores[i] += idf_val * (numerator / denominator)
        return scores
