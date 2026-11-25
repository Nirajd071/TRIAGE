"""GraphRAG-style KG + Dense Passage baseline (source-matched control).

Retrieves both dense text passages (ChromaDB) and 1-2 hop PrimeKG
neighborhood triples (linked via SapBERT and serialized to natural-language
sentences), ranked purely by flat S-PubMedBERT cosine similarity.

Unlike HCAD-RAG, this baseline uses:
- NO E/R/M semantic decoupling or soft Gram-Schmidt orthogonalization,
- NO anisotropic weighting or dynamic tensor,
- NO Heaviside statistical filter or JEV abstention gate.

It answers the reviewer concern that Vanilla Dense RAG receives passages
only whereas HCAD receives passages + PrimeKG facts.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from src.config import VANILLA_TOP_K
from src.data.triple_text import triple_to_sentence

logger = logging.getLogger(__name__)


class GraphRAGBaseline:
    """Flat GraphRAG-style baseline combining dense passages + PrimeKG 2-hop triples."""

    def __init__(
        self,
        dense_retrieve_fn: Callable[[str, int], List[Dict[str, Any]]],
        encode_fn: Callable[[List[str]], np.ndarray],
        ner: Any,
        entity_linker: Any,
        hypergraph: Any,
        top_k: int = VANILLA_TOP_K,
        dense_keep: int = 2,
        max_kg_pool: int = 80,
    ) -> None:
        self.dense_retrieve_fn = dense_retrieve_fn
        self.encode_fn = encode_fn
        self.ner = ner
        self.entity_linker = entity_linker
        self.hypergraph = hypergraph
        self.top_k = top_k
        self.dense_keep = dense_keep
        self.max_kg_pool = max_kg_pool

    def _linked_entity_names(self, query: str) -> List[str]:
        entities = self.ner.extract_entities(query) or {}
        linked = self.entity_linker.link_entities(entities) if self.entity_linker else entities
        names: List[str] = []
        for items in linked.values():
            for item in items:
                if isinstance(item, dict):
                    cand = item.get("canonical_name") or item.get("text")
                else:
                    cand = str(item)
                if cand and cand not in names:
                    names.append(cand)
        return names

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        k = top_k or self.top_k
        dense_hits = list(self.dense_retrieve_fn(query, k) or [])

        seed_names = self._linked_entity_names(query)
        kg_candidates: List[Dict[str, Any]] = []
        if seed_names and self.hypergraph is not None:
            expanded: List[str] = list(seed_names)
            for name in seed_names[:3]:
                for nb in list(self.hypergraph.expand_neighborhood(name, hops=1))[:8]:
                    if str(nb) not in expanded:
                        expanded.append(str(nb))
            triples = self.hypergraph.get_triples_for_entities(expanded)[: self.max_kg_pool]
            if triples:
                texts = [triple_to_sentence(str(u), str(rel), str(v)) for u, rel, v in triples]
                embs = np.asarray(self.encode_fn([query] + texts), dtype=np.float32)
                norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
                embs = embs / norms
                sims = embs[1:] @ embs[0]
                for (u, rel, v), txt, sim in zip(triples, texts, sims):
                    kg_candidates.append({
                        "text": txt,
                        "document": txt,
                        "score": float(sim),
                        "source": "primekg_graphrag",
                        "metadata": {
                            "head_name": str(u),
                            "relation": str(rel),
                            "tail_name": str(v),
                        },
                    })
                kg_candidates.sort(key=lambda c: c["score"], reverse=True)

        # Keep dense_keep prose passages and fill remaining slots with top KG triples,
        # falling back to extra dense hits if KG is sparse (always returns top-k).
        chosen: List[Dict[str, Any]] = []
        seen_texts = set()
        for hit in dense_hits[: self.dense_keep]:
            txt = str(hit.get("text") or hit.get("document") or "").strip()
            if txt and txt not in seen_texts:
                chosen.append(dict(hit))
                seen_texts.add(txt)
        for hit in kg_candidates:
            if len(chosen) >= k:
                break
            txt = str(hit.get("text") or "").strip()
            if txt and txt not in seen_texts:
                chosen.append(hit)
                seen_texts.add(txt)
        for hit in dense_hits[self.dense_keep :]:
            if len(chosen) >= k:
                break
            txt = str(hit.get("text") or hit.get("document") or "").strip()
            if txt and txt not in seen_texts:
                chosen.append(dict(hit))
                seen_texts.add(txt)
        return chosen[:k]
