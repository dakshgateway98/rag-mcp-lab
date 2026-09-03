from ragmcp.config import get_default_config
from ragmcp.indexer import create_sample_documents
from ragmcp.pipeline import RAGPipeline
from ragmcp.search import HybridSearchEngine
from ragmcp.eval_metrics import MetricsCalculator
from ragmcp.techniques import compare_techniques


def test_search_returns_results():
    engine = HybridSearchEngine(get_default_config().search)
    engine.index(create_sample_documents())
    results = engine.search("machine learning", top_k=5)
    assert results
    assert results[0].score > 0


def test_search_respects_top_k():
    engine = HybridSearchEngine(get_default_config().search)
    engine.index(create_sample_documents())
    results = engine.search("learning", top_k=3)
    assert len(results) <= 3


def test_vector_scores_bounded():
    engine = HybridSearchEngine(get_default_config().search)
    engine.index(create_sample_documents())
    results = engine.search("deep learning", top_k=5)
    assert all(0 <= r.vector_score <= 1 for r in results)


def test_pipeline_generate_without_api_key():
    pipeline = RAGPipeline(get_default_config())
    pipeline.index_documents(create_sample_documents())
    result = pipeline.generate("What is MCP?")
    assert result.answer
    assert result.generator in {"extractive", "none"} or result.sources


def test_compare_techniques():
    pipeline = RAGPipeline(get_default_config())
    pipeline.index_documents(create_sample_documents())
    payload = compare_techniques(pipeline, "hybrid RAG")
    ids = {run["technique_id"] for run in payload["runs"]}
    assert {"bm25", "vector", "hybrid", "hybrid_rerank", "multi_query", "hyde"} <= ids


def test_ndcg_perfect_ranking():
    assert MetricsCalculator.compute_ndcg([1, 0, 0], k=3) == 1.0
