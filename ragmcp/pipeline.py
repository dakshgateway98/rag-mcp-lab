"""Search → rerank → generate. LLM is optional."""

from __future__ import annotations

import os
from dataclasses import dataclass

from ragmcp.config import RAGConfig
from ragmcp.reranker import RerankerPipeline
from ragmcp.search import HybridSearchEngine

try:
    import anthropic
except ImportError:
    anthropic = None


@dataclass
class RAGResult:
    answer: str
    sources: list[dict]
    search_scores: list[float]
    rerank_scores: list[float]
    query: str
    generator: str


class RAGPipeline:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.search_engine = HybridSearchEngine(config.search)
        self.reranker_pipeline = RerankerPipeline(self.search_engine, config.reranker)
        self.llm_client = None
        if os.getenv("ANTHROPIC_API_KEY") and anthropic is not None:
            self.llm_client = anthropic.Anthropic()

    def index_documents(self, documents: list[dict]) -> None:
        self.search_engine.index(documents)

    def generate(self, query: str) -> RAGResult:
        reranked = self.reranker_pipeline.search_and_rerank(query)
        if not reranked:
            return RAGResult(
                answer="No relevant documents were retrieved.",
                sources=[],
                search_scores=[],
                rerank_scores=[],
                query=query,
                generator="none",
            )

        sources = [
            {"id": r.doc_id, "score": r.score, "metadata": r.metadata, "content": r.content}
            for r in reranked
        ]
        context = self._build_context(reranked)
        if self.llm_client:
            answer = self._llm_generate(query, context)
            generator = self.config.llm.model
        else:
            answer = self._extractive_generate(reranked)
            generator = "extractive"

        return RAGResult(
            answer=answer,
            sources=sources,
            search_scores=[s["score"] for s in sources],
            rerank_scores=[s["score"] for s in sources],
            query=query,
            generator=generator,
        )

    def _build_context(self, results: list) -> str:
        return "\n\n".join(f"[Source {i}]\n{r.content}" for i, r in enumerate(results, 1))

    def _extractive_generate(self, results: list) -> str:
        snippets = []
        for r in results[:3]:
            sentences = r.content.replace("\n", " ").split(". ")
            snippets.append(". ".join(sentences[:2]).strip())
        return "Based on retrieved sources:\n\n" + "\n\n".join(f"- {s}" for s in snippets if s)

    def _llm_generate(self, query: str, context: str) -> str:
        prompt = (
            "Use the following context to answer the question.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}"
        )
        response = self.llm_client.messages.create(
            model=self.config.llm.model,
            max_tokens=self.config.llm.max_tokens,
            temperature=self.config.llm.temperature,
            system=self.config.llm.system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
