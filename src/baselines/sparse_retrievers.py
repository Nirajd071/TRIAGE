"""Standard retrieval baselines requested by the reviewers.

- ``BM25Index``        Okapi BM25 (k1 = 1.2, b = 0.75) over a document list,
                       backed by a scipy CSR term matrix.
- ``reciprocal_rank_fusion``  RRF (Cormack et al., 2009), k = 60.
- ``BM25RAG`` / ``HybridRAG``  RAG arms that share the dense baseline's
  prompt, evidence budget and generator, so the arms differ ONLY in the
  ranking function.

Every retriever returns contexts in the same shape as
``VanillaVectorRAG.retrieve``: ``{"id", "text", "score", "metadata"}``.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from src.config import HYBRID_RRF_K, VANILLA_TOP_K

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
# Small English stopword list; clinical polarity words ("not", "no",
# "contraindicated", "avoid") are deliberately KEPT - they carry meaning.
_BM25_STOPWORDS = frozenset(
    "a an the of and or to in for on by with as at from is are was were be been "
    "this that these those it its which who whom what when where how".split()
)


def bm25_tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall((text or "").lower())
            if t not in _BM25_STOPWORDS]


class BM25Index:
    """Okapi BM25 with Lucene-style non-negative IDF.

    idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5))
    score(q, d) = sum_t idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl))
    """

    def __init__(self, documents: Sequence[str], ids: Optional[Sequence[str]] = None,
                 metadatas: Optional[Sequence[dict]] = None,
                 k1: float = 1.2, b: float = 0.75):
        from scipy.sparse import csr_matrix

        self.k1, self.b = k1, b
        self.documents = list(documents)
        self.ids = [str(i) for i in (ids if ids is not None else range(len(self.documents)))]
        self.metadatas = list(metadatas) if metadatas is not None else [{} for _ in self.documents]
        if not (len(self.ids) == len(self.documents) == len(self.metadatas)):
            raise ValueError("documents, ids and metadatas must align")
        self.vocab: dict[str, int] = {}
        rows, cols, vals = [], [], []
        doc_len = np.zeros(len(self.documents), dtype=np.float32)
        for d, text in enumerate(self.documents):
            toks = bm25_tokenize(text)
            doc_len[d] = len(toks)
            tf: dict[int, int] = {}
            for t in toks:
                j = self.vocab.setdefault(t, len(self.vocab))
                tf[j] = tf.get(j, 0) + 1
            rows.extend([d] * len(tf))
            cols.extend(tf.keys())
            vals.extend(tf.values())
        n_docs = len(self.documents)
        self._tf = csr_matrix((np.asarray(vals, dtype=np.float32),
                               (np.asarray(rows, dtype=np.int64),
                                np.asarray(cols, dtype=np.int64))),
                              shape=(n_docs, max(1, len(self.vocab))))
        df = np.bincount(np.asarray(cols, dtype=np.int64),
                         minlength=max(1, len(self.vocab))).astype(np.float64)
        self.idf = np.log1p((n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        avgdl = float(doc_len.mean()) if n_docs else 0.0
        self._norm = (k1 * (1.0 - b + b * doc_len / avgdl)).astype(np.float32) if avgdl else \
            np.full(n_docs, k1, dtype=np.float32)
        self._tf_csc = self._tf.tocsc()

    def __len__(self) -> int:
        return len(self.documents)

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.documents), dtype=np.float32)
        q_terms = [self.vocab[t] for t in set(bm25_tokenize(query)) if t in self.vocab]
        for j in q_terms:
            col = self._tf_csc.getcol(j)
            docs = col.indices
            tf = col.data
            out[docs] += self.idf[j] * tf * (self.k1 + 1.0) / (tf + self._norm[docs])
        return out

    def search(self, query: str, top_k: int = VANILLA_TOP_K) -> list[dict]:
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if not self.documents:
            return []
        s = self.scores(query)
        k = min(top_k, len(s))
        top = np.argpartition(-s, k - 1)[:k]
        # deterministic: score desc, then index asc
        top = sorted(top.tolist(), key=lambda i: (-float(s[i]), i))
        return [{"id": self.ids[i], "text": self.documents[i],
                 "score": round(float(s[i]), 6), "metadata": self.metadatas[i] or {}}
                for i in top]

    # -- persistence for the 800k-document benchmark store --------------------
    @classmethod
    def from_collection(cls, collection, cache_path: Optional[Path] = None,
                        page_size: int = 5000) -> "BM25Index":
        """Build over every document of a Chroma collection (paged)."""
        if cache_path is not None and Path(cache_path).exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if cached.get("count") == collection.count():
                logger.info("BM25 corpus loaded from %s (%d docs)", cache_path, cached["count"])
                return cls(cached["documents"], cached["ids"], cached["metadatas"])
        docs, ids, metas = [], [], []
        total = collection.count()
        for offset in range(0, total, page_size):
            got = collection.get(limit=page_size, offset=offset,
                                 include=["documents", "metadatas"])
            docs.extend(got.get("documents") or [])
            ids.extend(got.get("ids") or [])
            metas.extend(got.get("metadatas") or [{}] * len(got.get("documents") or []))
        logger.info("BM25 index built over %d documents", len(docs))
        if cache_path is not None:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump({"count": total, "documents": docs, "ids": ids,
                           "metadatas": metas}, f)
        return cls(docs, ids, metas)


def reciprocal_rank_fusion(rankings: Sequence[Sequence[dict]], top_k: int,
                           k: int = HYBRID_RRF_K) -> list[dict]:
    """Fuse ranked context lists: score(d) = sum_r 1 / (k + rank_r(d))."""
    fused: dict[str, float] = {}
    first_seen: dict[str, dict] = {}
    order: dict[str, int] = {}
    for ranking in rankings:
        for rank, ctx in enumerate(ranking, 1):
            key = str(ctx.get("id"))
            fused[key] = fused.get(key, 0.0) + 1.0 / (k + rank)
            if key not in first_seen:
                first_seen[key] = ctx
                order[key] = len(order)
    ranked = sorted(fused, key=lambda key: (-fused[key], order[key]))[:top_k]
    return [{**first_seen[key], "score": round(fused[key], 6)} for key in ranked]


class _CorpusRAG:
    """Shared generation path: identical prompt/budget/generator to dense RAG."""

    name = "corpus-rag"

    def __init__(self, generator=None):
        # ``generator`` is a VanillaVectorRAG (or anything exposing
        # build_messages / llm); reused so all RAG arms share one prompt.
        self._generator = generator

    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[dict]:
        raise NotImplementedError

    def build_messages(self, query: str, contexts: list[dict]) -> list[dict]:
        from src.baselines.vanilla_vector_rag import build_rag_messages
        return build_rag_messages(query, contexts)

    def retrieve_and_generate(self, query: str, contexts: Optional[list[dict]] = None) -> str:
        from src.core.llm_client import UnifiedLLMClient
        from src.config import LLM_MAX_TOKENS, LLM_TEMPERATURE

        if contexts is None:
            contexts = self.retrieve(query)
        model = getattr(getattr(self._generator, "llm", None), "model", None)
        client = UnifiedLLMClient(model=model, temperature=LLM_TEMPERATURE,
                                  max_tokens=LLM_MAX_TOKENS)
        return client.chat(self.build_messages(query, contexts))


class BM25RAG(_CorpusRAG):
    name = "bm25"

    def __init__(self, index: BM25Index, top_k: int = VANILLA_TOP_K, generator=None):
        super().__init__(generator)
        self.index, self.top_k = index, top_k

    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[dict]:
        return self.index.search(query, top_k or self.top_k)


class HybridRAG(_CorpusRAG):
    """BM25 + dense fused with RRF over each arm's top-``pool`` list."""

    name = "hybrid"

    def __init__(self, index: BM25Index, dense_retrieve: Callable[[str, int], list[dict]],
                 top_k: int = VANILLA_TOP_K, pool: int = 50, generator=None):
        super().__init__(generator)
        self.index, self.dense_retrieve = index, dense_retrieve
        self.top_k, self.pool = top_k, pool

    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[dict]:
        k = top_k or self.top_k
        sparse = self.index.search(query, self.pool)
        dense = self.dense_retrieve(query, self.pool)
        return reciprocal_rank_fusion([sparse, dense], top_k=k)

