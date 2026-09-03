"""HTTP API + comparison UI."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ragmcp.config import get_default_config
from ragmcp.eval_metrics import EvaluationRunner
from ragmcp.indexer import create_sample_documents
from ragmcp.pipeline import RAGPipeline
from ragmcp.techniques import TECHNIQUES, compare_techniques

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
EVAL = ROOT / "eval" / "questions.jsonl"

app = FastAPI(
    title="RAG + MCP Lab",
    description="Compare RAG retrievers and inspect an MCP tool surface.",
    version="1.0.0",
)

pipeline = RAGPipeline(get_default_config())
pipeline.index_documents(create_sample_documents())


class QueryBody(BaseModel):
    query: str


@app.get("/api/techniques")
def api_techniques():
    return {"techniques": TECHNIQUES}


@app.post("/api/compare")
def api_compare(body: QueryBody):
    if not body.query.strip():
        raise HTTPException(400, "query is required")
    return compare_techniques(pipeline, body.query.strip())


@app.post("/api/ask")
def api_ask(body: QueryBody):
    result = pipeline.generate(body.query.strip())
    return {
        "query": result.query,
        "answer": result.answer,
        "generator": result.generator,
        "sources": result.sources,
    }


@app.get("/api/eval")
def api_eval():
    questions = [json.loads(line) for line in EVAL.read_text(encoding="utf-8").splitlines() if line]
    metrics = EvaluationRunner(pipeline, k_values=[1, 5, 10]).evaluate(questions)
    return {
        "ndcg": metrics.ndcg,
        "mrr": metrics.mrr,
        "hit_rate": metrics.hit_rate,
        "precision": metrics.precision,
        "map": metrics.map_score,
    }


@app.get("/api/mcp")
def api_mcp():
    return {
        "protocol": "Model Context Protocol",
        "roles": {
            "host": "The agent app (Claude Desktop, Cursor, custom runtime)",
            "client": "Speaks MCP to one or more servers",
            "server": "This lab: python -m ragmcp.mcp_server",
        },
        "primitives": {
            "tools": [
                {"name": "search_corpus", "use": "Retrieve + answer from the demo corpus"},
                {"name": "compare_rag_techniques", "use": "Same query through several retrievers"},
                {"name": "list_rag_techniques", "use": "Catalog of RAG families"},
            ],
            "resources": [
                {"uri": "lab://mcp/architecture", "use": "Host / client / server primer"},
            ],
        },
        "vs_rag": [
            "RAG injects documents into the prompt.",
            "MCP exposes capabilities the model may call.",
            "A solid agent often uses MCP tools that run RAG inside the tool.",
        ],
        "run": "python -m ragmcp.mcp_server",
    }


@app.get("/health")
def health():
    return {"status": "ok", "documents": len(pipeline.search_engine.documents)}


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


app.mount("/static", StaticFiles(directory=str(WEB)), name="static")


def main() -> None:
    import uvicorn

    uvicorn.run("ragmcp.server:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
