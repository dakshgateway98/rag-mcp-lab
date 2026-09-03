"""Hybrid search: BM25 + bag-of-words cosine vectors.

The vector side is a stable hash embedding so the lab runs without a model
download. Swap `_embed` for sentence-transformers in production.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from ragmcp.config import SearchConfig


@dataclass
class SearchResult:
    doc_id: str
    content: str
    score: float
    bm25_score: float
    vector_score: float
    metadata: dict


class HybridSearchEngine:
    def __init__(self, config: SearchConfig):
        self.config = config
        self.documents: dict = {}
        self.bm25 = None
        self.embeddings: dict[str, np.ndarray] = {}
        self.doc_ids: list[str] = []

    def index(self, documents: list[dict]) -> None:
        self.documents = {doc["id"]: doc for doc in documents}
        self.doc_ids = list(self.documents.keys())
        tokenized_docs = [doc["content"].lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        for doc in documents:
            self.embeddings[doc["id"]] = self._embed(doc["content"])

    def search(self, query: str, top_k: int | None = None, mode: str = "hybrid") -> list[SearchResult]:
        if top_k is None:
            top_k = self.config.top_k
        if not self.doc_ids or self.bm25 is None:
            return []

        query_tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        query_embedding = self._embed(query)
        vector_scores = self._compute_vector_scores(query_embedding)

        results: list[SearchResult] = []
        for idx, doc_id in enumerate(self.doc_ids):
            bm25_norm = max(0.0, min(1.0, float(bm25_scores[idx]) / 3.0))
            vector_norm = float(vector_scores.get(doc_id, 0.0))
            if mode == "bm25":
                hybrid_score = bm25_norm
            elif mode == "vector":
                hybrid_score = vector_norm
            else:
                hybrid_score = (
                    self.config.bm25_weight * bm25_norm
                    + self.config.vector_weight * vector_norm
                )
            if (
                bm25_norm >= self.config.bm25_threshold
                or vector_norm >= self.config.vector_similarity_threshold
            ):
                doc = self.documents[doc_id]
                results.append(
                    SearchResult(
                        doc_id=doc_id,
                        content=doc["content"],
                        score=hybrid_score,
                        bm25_score=bm25_norm,
                        vector_score=vector_norm,
                        metadata=doc.get("metadata", {}),
                    )
                )

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def _embed(self, text: str) -> np.ndarray:
        embedding = np.zeros(384)
        for word in text.lower().split():
            digest = hashlib.sha256(word.encode("utf-8")).digest()
            seed = int.from_bytes(digest[:2], "little") % 384
            embedding[seed] += 1.0
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    def _compute_vector_scores(self, query_embedding: np.ndarray) -> dict[str, float]:
        scores = {}
        for doc_id, doc_embedding in self.embeddings.items():
            scores[doc_id] = max(0.0, float(np.dot(query_embedding, doc_embedding)))
        return scores

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "documents": self.documents,
            "doc_ids": self.doc_ids,
            "embeddings": {k: v.tolist() for k, v in self.embeddings.items()},
        }
        path.write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, config: SearchConfig, path: Path) -> "HybridSearchEngine":
        engine = cls(config)
        data = json.loads(path.read_text(encoding="utf-8"))
        engine.documents = data["documents"]
        engine.doc_ids = data["doc_ids"]
        engine.embeddings = {k: np.array(v) for k, v in data["embeddings"].items()}
        tokenized_docs = [
            engine.documents[doc_id]["content"].lower().split()
            for doc_id in engine.doc_ids
        ]
        engine.bm25 = BM25Okapi(tokenized_docs)
        return engine
