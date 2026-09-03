# RAG + MCP Lab

Interview demo that **compares RAG retrievers on one query** and shows how **MCP tools** wrap the same engine.

The hybrid search, heuristic reranker, eval metrics, and CLI flow started from the `Agentic_AI` lab. This repo packages that code, adds technique comparison, a UI, and an MCP server.

Retrieval embeddings here are **hash bag-of-words**, not a production encoder. BM25 is real (`rank-bm25`). The reranker is overlap-based, not MiniLM. That is intentional so the lab runs with `pip install` and no GPU.

## What you can show

| RAG | MCP |
| --- | --- |
| BM25 vs naive vector vs hybrid | Host / client / server |
| Hybrid + rerank (two-stage) | Tools: `search_corpus`, `compare_rag_techniques` |
| Multi-query merge | Resource: `lab://mcp/architecture` |
| HyDE-style query transform | RAG injects text; MCP calls capabilities |

## Run the UI

```bash
cd rag-mcp-lab
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m ragmcp.server
```

Open http://127.0.0.1:8000

Optional Claude answers: copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY`. Without a key, answers are extractive from retrieved chunks.

## CLI

```bash
python -m ragmcp compare --query "What is hybrid RAG?"
python -m ragmcp ask --query "When do I use MCP?"
python -m ragmcp evaluate
```

## MCP server

```bash
python -m ragmcp.mcp_server
```

Point Claude Desktop / Cursor at that stdio process. The tools call the same `RAGPipeline` as the website.

## Tests

```bash
pytest -q
```

## Layout

```text
ragmcp/          search, rerank, pipeline, techniques, MCP, HTTP API
web/             comparison UI
eval/            relevance labels
tests/
```
