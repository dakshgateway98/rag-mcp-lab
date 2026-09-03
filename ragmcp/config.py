"""Configuration for the RAG + MCP lab."""

from dataclasses import dataclass, field


@dataclass
class SearchConfig:
    bm25_weight: float = 0.4
    vector_weight: float = 0.6
    top_k: int = 10
    vector_similarity_threshold: float = 0.15
    bm25_threshold: float = 0.05


@dataclass
class RerankerConfig:
    enabled: bool = True
    model: str = "heuristic-overlap-reranker"
    top_k: int = 5
    threshold: float = 0.0
    batch_size: int = 32


@dataclass
class LLMConfig:
    model: str = "claude-sonnet-4-20250514"
    temperature: float = 0.3
    max_tokens: int = 1024
    system_prompt: str = (
        "You are a helpful assistant. Answer only from the provided context. "
        "If the context is insufficient, say so."
    )


@dataclass
class EvalConfig:
    metrics: list[str] = field(default_factory=lambda: ["ndcg", "mrr", "hit_rate", "precision"])
    k_values: list[int] = field(default_factory=lambda: [1, 5, 10])


@dataclass
class RAGConfig:
    search: SearchConfig
    reranker: RerankerConfig
    llm: LLMConfig
    eval: EvalConfig
    index_dir: str = "./data/indices"
    document_dir: str = "./data/documents"
    eval_dir: str = "./eval"
    log_level: str = "INFO"


def get_default_config() -> RAGConfig:
    return RAGConfig(
        search=SearchConfig(),
        reranker=RerankerConfig(),
        llm=LLMConfig(),
        eval=EvalConfig(),
    )
