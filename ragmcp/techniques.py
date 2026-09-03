"""Side-by-side RAG techniques on the same corpus and query."""

from __future__ import annotations

from dataclasses import dataclass, asdict

from ragmcp.pipeline import RAGPipeline
from ragmcp.reranker import CrossEncoderReranker


TECHNIQUES = [
    {
        "id": "bm25",
        "name": "BM25 only",
        "family": "lexical",
        "when": "IDs, error codes, exact phrases",
        "cost": "cheap",
        "weakness": "Paraphrases and synonyms miss",
        "pipeline": "tokenize → BM25 → top-k",
    },
    {
        "id": "vector",
        "name": "Naive vector RAG",
        "family": "semantic",
        "when": "Natural language questions",
        "cost": "cheap + embedding model",
        "weakness": "Misses rare tokens and strict keywords",
        "pipeline": "embed query → cosine kNN → generate",
    },
    {
        "id": "hybrid",
        "name": "Hybrid (BM25 + vector)",
        "family": "fusion",
        "when": "Mixed keyword + meaning workloads",
        "cost": "cheap",
        "weakness": "Fusion weights need tuning",
        "pipeline": "BM25 ∥ vectors → normalize → weighted sum",
    },
    {
        "id": "hybrid_rerank",
        "name": "Hybrid + rerank",
        "family": "two-stage",
        "when": "Quality matters more than extra latency",
        "cost": "medium",
        "weakness": "Reranker latency on large candidate sets",
        "pipeline": "hybrid recall → pair scorer → generate",
    },
    {
        "id": "multi_query",
        "name": "Multi-query",
        "family": "query expansion",
        "when": "Ambiguous or underspecified questions",
        "cost": "medium (N retrievals)",
        "weakness": "Can retrieve off-topic rewrites",
        "pipeline": "rewrite N queries → retrieve → merge/dedupe",
    },
    {
        "id": "hyde",
        "name": "HyDE-style",
        "family": "query transformation",
        "when": "Short queries that do not match corpus wording",
        "cost": "medium (extra generation in production)",
        "weakness": "A wrong hypothetical answer poisons retrieval",
        "pipeline": "expand query into a hypothetical passage → embed/search",
    },
    {
        "id": "agentic",
        "name": "Agentic RAG (described)",
        "family": "agent",
        "when": "Multi-hop research, tool choice, retries",
        "cost": "high",
        "weakness": "Latency, loops, harder eval",
        "pipeline": "plan → retrieve? → tool/MCP? → maybe retrieve again → answer",
    },
]


@dataclass
class TechniqueRun:
    technique_id: str
    name: str
    doc_ids: list[str]
    previews: list[dict]


def _previews(results, limit: int = 4) -> list[dict]:
    return [
        {
            "id": r.doc_id,
            "score": round(float(r.score), 4),
            "category": r.metadata.get("category", ""),
            "snippet": r.content[:180],
        }
        for r in results[:limit]
    ]


def _merge_by_id(result_lists: list[list]) -> list:
    best = {}
    for group in result_lists:
        for item in group:
            prev = best.get(item.doc_id)
            if prev is None or item.score > prev.score:
                best[item.doc_id] = item
    merged = sorted(best.values(), key=lambda x: x.score, reverse=True)
    return merged


def _hyde_passage(query: str) -> str:
    return (
        f"{query} This hypothetical note covers definitions, architecture, "
        f"trade-offs, and how practitioners apply {query} in production systems."
    )


def _multi_queries(query: str) -> list[str]:
    return [
        query,
        f"definition of {query}",
        f"how {query} works in practice",
    ]


def compare_techniques(pipeline: RAGPipeline, query: str) -> dict:
    engine = pipeline.search_engine
    reranker = CrossEncoderReranker()

    bm25 = engine.search(query, mode="bm25")
    vector = engine.search(query, mode="vector")
    hybrid = engine.search(query, mode="hybrid")
    reranked = reranker.rerank(query, hybrid, top_k=5, threshold=0.0)

    multi_hits = [engine.search(q, mode="hybrid") for q in _multi_queries(query)]
    multi = _merge_by_id(multi_hits)[:5]

    hyde = engine.search(_hyde_passage(query), mode="vector")[:5]

    runs = [
        TechniqueRun("bm25", "BM25 only", [r.doc_id for r in bm25[:5]], _previews(bm25)),
        TechniqueRun("vector", "Naive vector RAG", [r.doc_id for r in vector[:5]], _previews(vector)),
        TechniqueRun("hybrid", "Hybrid", [r.doc_id for r in hybrid[:5]], _previews(hybrid)),
        TechniqueRun("hybrid_rerank", "Hybrid + rerank", [r.doc_id for r in reranked], _previews(reranked)),
        TechniqueRun("multi_query", "Multi-query", [r.doc_id for r in multi], _previews(multi)),
        TechniqueRun("hyde", "HyDE-style", [r.doc_id for r in hyde], _previews(hyde)),
    ]

    return {
        "query": query,
        "catalog": TECHNIQUES,
        "runs": [asdict(run) for run in runs],
        "mcp": {
            "idea": "MCP does not retrieve chunks. It exposes tools/resources the host can call.",
            "this_lab": [
                "search_corpus — live retrieval tool",
                "compare_rag_techniques — same query, several retrievers",
                "explain_mcp — protocol primer as a resource/tool",
            ],
        },
    }
