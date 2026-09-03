"""Heuristic reranker (stand-in for a cross-encoder)."""

from dataclasses import dataclass

import numpy as np


@dataclass
class RerankerResult:
    doc_id: str
    content: str
    score: float
    rank: int
    metadata: dict


class CrossEncoderReranker:
    def __init__(self, model_name: str = "heuristic-overlap-reranker"):
        self.model_name = model_name
        self.cache: dict = {}

    def rerank(self, query: str, results: list, top_k: int = 5, threshold: float = 0.0) -> list[RerankerResult]:
        if not results:
            return []

        scores = np.array([self._score_pair(query, result.content) for result in results], dtype=float)
        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())
        else:
            scores = np.ones_like(scores) * 0.5

        reranked: list[RerankerResult] = []
        for result, score in zip(results, scores):
            if score >= threshold:
                reranked.append(
                    RerankerResult(
                        doc_id=result.doc_id,
                        content=result.content,
                        score=float(score),
                        rank=0,
                        metadata=result.metadata,
                    )
                )
        reranked.sort(key=lambda x: x.score, reverse=True)
        out = reranked[:top_k]
        for i, item in enumerate(out):
            item.rank = i + 1
        return out

    def _score_pair(self, query: str, document: str) -> float:
        cache_key = (query, document[:120])
        if cache_key in self.cache:
            return self.cache[cache_key]

        query_words = set(query.lower().split())
        doc_words = set(document.lower().split())
        union = len(query_words | doc_words)
        jaccard = (len(query_words & doc_words) / union) if union else 0.0
        all_terms = all(word in doc_words for word in query_words if len(word) > 3)
        bonus = 0.2 if all_terms else 0.0
        length_penalty = min(1.0, max(0.4, len(document.split()) / 400))
        score = (jaccard * 0.7 + bonus * 0.3) * length_penalty
        self.cache[cache_key] = score
        return score


class RerankerPipeline:
    def __init__(self, search_engine, reranker_config):
        self.search_engine = search_engine
        self.reranker = CrossEncoderReranker(reranker_config.model)
        self.config = reranker_config

    def search_and_rerank(self, query: str, mode: str = "hybrid") -> list[RerankerResult]:
        search_results = self.search_engine.search(query, top_k=self.config.top_k * 2, mode=mode)
        if not self.config.enabled:
            return [
                RerankerResult(
                    doc_id=r.doc_id,
                    content=r.content,
                    score=r.score,
                    rank=i + 1,
                    metadata=r.metadata,
                )
                for i, r in enumerate(search_results)
            ]
        return self.reranker.rerank(
            query=query,
            results=search_results,
            top_k=self.config.top_k,
            threshold=self.config.threshold,
        )
