"""Document indexing and the demo corpus."""

from pathlib import Path

from ragmcp.config import SearchConfig
from ragmcp.search import HybridSearchEngine


class DocumentIndexer:
    def __init__(self, config: SearchConfig, index_path: Path):
        self.config = config
        self.index_path = index_path
        self.search_engine = HybridSearchEngine(config)

    def index_from_directory(self, doc_dir: Path, file_pattern: str = "*.md") -> None:
        documents = []
        for file_path in Path(doc_dir).glob(file_pattern):
            documents.append(
                {
                    "id": file_path.stem,
                    "content": file_path.read_text(encoding="utf-8"),
                    "metadata": {"filename": file_path.name, "path": str(file_path)},
                }
            )
        self.search_engine.index(documents)
        self.search_engine.save(self.index_path)

    def index_from_list(self, documents: list[dict]) -> None:
        self.search_engine.index(documents)
        self.search_engine.save(self.index_path)


def create_sample_documents() -> list[dict]:
    return [
        {
            "id": "doc_ml",
            "content": "Machine learning is a subset of artificial intelligence that enables systems to learn from data without being explicitly programmed for every case.",
            "metadata": {"category": "ml", "source": "corpus"},
        },
        {
            "id": "doc_nn",
            "content": "Neural networks are computing systems of interconnected nodes inspired by biological brains. They learn weights with backpropagation.",
            "metadata": {"category": "ml", "source": "corpus"},
        },
        {
            "id": "doc_transformers",
            "content": "Transformers use self-attention to process sequences in parallel and are the foundation of modern large language models.",
            "metadata": {"category": "ml", "source": "corpus"},
        },
        {
            "id": "doc_embeddings",
            "content": "Embeddings are dense vectors that place similar meaning nearby in vector space. They power semantic search in RAG systems.",
            "metadata": {"category": "rag", "source": "corpus"},
        },
        {
            "id": "doc_naive_rag",
            "content": "Naive RAG embeds a query, retrieves top-k chunks by cosine similarity, and stuffs them into a prompt. It is simple but fails on keyword-heavy or multi-hop questions.",
            "metadata": {"category": "rag", "source": "lab"},
        },
        {
            "id": "doc_hybrid_rag",
            "content": "Hybrid RAG mixes BM25 lexical matching with vector search. BM25 catches rare identifiers and exact phrases. Vectors catch paraphrases. Scores are normalized then weighted.",
            "metadata": {"category": "rag", "source": "lab"},
        },
        {
            "id": "doc_rerank",
            "content": "A two-stage retriever recalls many candidates cheaply, then a reranker scores query-document pairs more carefully. Search optimizes recall. Reranking optimizes precision.",
            "metadata": {"category": "rag", "source": "lab"},
        },
        {
            "id": "doc_multiquery",
            "content": "Multi-query RAG rewrites one user question into several search queries, retrieves for each, then merges and deduplicates. It improves recall when wording is ambiguous.",
            "metadata": {"category": "rag", "source": "lab"},
        },
        {
            "id": "doc_hyde",
            "content": "HyDE (Hypothetical Document Embeddings) first generates a fake answer, embeds that answer, and searches with the hypothetical document instead of the raw question. It helps when queries are short.",
            "metadata": {"category": "rag", "source": "lab"},
        },
        {
            "id": "doc_agentic_rag",
            "content": "Agentic RAG lets a model decide whether to retrieve, what to retrieve, and whether to retrieve again. It is slower and costs more tokens but handles multi-step research better than a single retrieve-then-generate pass.",
            "metadata": {"category": "rag", "source": "lab"},
        },
        {
            "id": "doc_mcp",
            "content": "Model Context Protocol (MCP) is a standard way for an LLM host to call tools, read resources, and load prompts from a server. The host is the app. The client speaks MCP. The server exposes tools. RAG retrieves text. MCP invokes capabilities.",
            "metadata": {"category": "mcp", "source": "lab"},
        },
        {
            "id": "doc_mcp_tools",
            "content": "An MCP tool is a typed function the model can call: name, JSON schema, result. Typical tools are search_corpus, compare_rag, or create_ticket. Tools are live actions. RAG chunks are static context.",
            "metadata": {"category": "mcp", "source": "lab"},
        },
        {
            "id": "doc_mcp_vs_rag",
            "content": "Use RAG when the answer is in documents you own. Use MCP when the model must take an action or read a live system: databases, git, calendars, this search engine. Many production agents use both: MCP tools that themselves run RAG.",
            "metadata": {"category": "mcp", "source": "lab"},
        },
        {
            "id": "doc_supervised",
            "content": "Supervised learning trains on labeled input-output pairs. Unsupervised learning finds structure in unlabeled data such as clusters or embeddings.",
            "metadata": {"category": "ml", "source": "corpus"},
        },
    ]
