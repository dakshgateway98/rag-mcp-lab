"""IR metrics used to compare retrievers."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class EvalMetrics:
    ndcg: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    hit_rate: dict[int, float] = field(default_factory=dict)
    precision: dict[int, float] = field(default_factory=dict)
    recall: dict[int, float] = field(default_factory=dict)
    map_score: float = 0.0

    def __str__(self) -> str:
        lines = ["Evaluation Metrics:", "-" * 40]
        for k, v in sorted(self.ndcg.items()):
            lines.append(f"NDCG@{k}: {v:.4f}")
        lines.append(f"MRR: {self.mrr:.4f}")
        for k, v in sorted(self.hit_rate.items()):
            lines.append(f"Hit Rate@{k}: {v:.4f}")
        for k, v in sorted(self.precision.items()):
            lines.append(f"Precision@{k}: {v:.4f}")
        lines.append(f"MAP: {self.map_score:.4f}")
        return "\n".join(lines)


class MetricsCalculator:
    @staticmethod
    def compute_ndcg(relevance_scores: list[float], k: int = 10) -> float:
        relevance_scores = relevance_scores[:k]
        dcg = sum((2 ** rel - 1) / np.log2(i + 2) for i, rel in enumerate(relevance_scores))
        sorted_relevance = sorted(relevance_scores, reverse=True)
        idcg = sum((2 ** rel - 1) / np.log2(i + 2) for i, rel in enumerate(sorted_relevance))
        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def compute_mrr(relevance_scores: list[float], threshold: float = 0.5) -> float:
        for i, score in enumerate(relevance_scores):
            if score >= threshold:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def compute_hit_rate(relevance_scores: list[float], k: int = 10, threshold: float = 0.5) -> float:
        return 1.0 if any(score >= threshold for score in relevance_scores[:k]) else 0.0

    @staticmethod
    def compute_precision(relevance_scores: list[float], k: int = 10, threshold: float = 0.5) -> float:
        top_k = relevance_scores[:k]
        if not top_k:
            return 0.0
        return sum(1 for score in top_k if score >= threshold) / len(top_k)

    @staticmethod
    def compute_map(relevance_scores: list[float], threshold: float = 0.5) -> float:
        precisions = []
        num_relevant = 0
        for i, score in enumerate(relevance_scores):
            if score >= threshold:
                num_relevant += 1
                precisions.append(num_relevant / (i + 1))
        return sum(precisions) / len(precisions) if precisions else 0.0


class EvaluationRunner:
    def __init__(self, rag_pipeline, k_values: list[int] | None = None):
        self.rag_pipeline = rag_pipeline
        self.k_values = k_values or [1, 5, 10]
        self.calculator = MetricsCalculator()

    def evaluate(self, eval_questions: list[dict]) -> EvalMetrics:
        all_ndcg = {k: [] for k in self.k_values}
        all_mrr = []
        all_hit_rate = {k: [] for k in self.k_values}
        all_precision = {k: [] for k in self.k_values}
        all_map = []

        for question in eval_questions:
            relevant = set(question.get("relevant_doc_ids", []))
            results = self.rag_pipeline.reranker_pipeline.search_and_rerank(question["query"])
            relevance_scores = [1.0 if r.doc_id in relevant else 0.0 for r in results]
            for k in self.k_values:
                all_ndcg[k].append(self.calculator.compute_ndcg(relevance_scores, k))
                all_hit_rate[k].append(self.calculator.compute_hit_rate(relevance_scores, k))
                all_precision[k].append(self.calculator.compute_precision(relevance_scores, k))
            all_mrr.append(self.calculator.compute_mrr(relevance_scores))
            all_map.append(self.calculator.compute_map(relevance_scores))

        return EvalMetrics(
            ndcg={k: float(np.mean(scores)) for k, scores in all_ndcg.items()},
            mrr=float(np.mean(all_mrr)),
            hit_rate={k: float(np.mean(scores)) for k, scores in all_hit_rate.items()},
            precision={k: float(np.mean(scores)) for k, scores in all_precision.items()},
            map_score=float(np.mean(all_map)),
        )
