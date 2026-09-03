"""MCP server: tools wrap the same RAG engine the HTTP UI uses.

Run:
  python -m ragmcp.mcp_server

Claude Desktop config (stdio):

{
  "mcpServers": {
    "rag-mcp-lab": {
      "command": "python",
      "args": ["-m", "ragmcp.mcp_server"],
      "cwd": "C:/Users/daksh/Downloads/all_repos/rag-mcp-lab"
    }
  }
}
"""

from __future__ import annotations

import json

from ragmcp.config import get_default_config
from ragmcp.indexer import create_sample_documents
from ragmcp.pipeline import RAGPipeline
from ragmcp.techniques import TECHNIQUES, compare_techniques

_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline(get_default_config())
        _pipeline.index_documents(create_sample_documents())
    return _pipeline


def create_mcp():
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP(
        "rag-mcp-lab",
        instructions=(
            "Lab server for comparing RAG retrievers and showing MCP tools. "
            "RAG fetches documents. These tools are live function calls."
        ),
    )

    @mcp.tool()
    def search_corpus(query: str) -> str:
        """Hybrid search + rerank over the demo corpus."""
        result = get_pipeline().generate(query)
        return json.dumps(
            {
                "answer": result.answer,
                "generator": result.generator,
                "sources": [
                    {"id": s["id"], "score": s["score"], "category": s["metadata"].get("category")}
                    for s in result.sources
                ],
            },
            indent=2,
        )

    @mcp.tool()
    def compare_rag_techniques(query: str) -> str:
        """Run BM25, vector, hybrid, rerank, multi-query, and HyDE-style retrieval on one query."""
        return json.dumps(compare_techniques(get_pipeline(), query), indent=2)

    @mcp.tool()
    def list_rag_techniques() -> str:
        """Catalog of RAG techniques and when to use each."""
        return json.dumps(TECHNIQUES, indent=2)

    @mcp.resource("lab://mcp/architecture")
    def mcp_architecture() -> str:
        return (
            "MCP roles\n"
            "- Host: Claude Desktop / Cursor / your agent app\n"
            "- Client: MCP client inside the host\n"
            "- Server: this process (tools + resources)\n\n"
            "Primitives\n"
            "- Tools: callable functions with JSON schemas (search_corpus)\n"
            "- Resources: readable data (this text)\n"
            "- Prompts: reusable templates (optional)\n\n"
            "RAG vs MCP\n"
            "- RAG: retrieve text, stuff into context, generate\n"
            "- MCP: typed capability boundary; the model decides to call a tool\n"
            "- Combined: an MCP tool can run RAG internally (search_corpus does)\n"
        )

    return mcp


def main() -> None:
    create_mcp().run()


if __name__ == "__main__":
    main()
