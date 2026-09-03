"""CLI: index, search, ask, evaluate, serve."""

from pathlib import Path

import click

from ragmcp.config import get_default_config
from ragmcp.eval_metrics import EvaluationRunner
from ragmcp.indexer import DocumentIndexer, create_sample_documents
from ragmcp.pipeline import RAGPipeline
from ragmcp.techniques import compare_techniques


def _load_pipeline(index_dir: str) -> RAGPipeline:
    config = get_default_config()
    pipeline = RAGPipeline(config)
    index_path = Path(index_dir) / "index.json"
    if index_path.exists():
        pipeline.search_engine = pipeline.search_engine.load(config.search, index_path)
        pipeline.reranker_pipeline.search_engine = pipeline.search_engine
    else:
        pipeline.index_documents(create_sample_documents())
    return pipeline


@click.group()
def cli():
    """RAG technique lab + MCP demo."""


@cli.command()
@click.option("--index-dir", default="./data/indices")
def index(index_dir: str):
    path = Path(index_dir) / "index.json"
    indexer = DocumentIndexer(get_default_config().search, path)
    indexer.index_from_list(create_sample_documents())
    click.echo(f"Indexed {len(indexer.search_engine.documents)} docs → {path}")


@cli.command()
@click.option("--query", prompt=True)
@click.option("--index-dir", default="./data/indices")
def search(query: str, index_dir: str):
    pipeline = _load_pipeline(index_dir)
    results = pipeline.reranker_pipeline.search_and_rerank(query)
    for r in results:
        click.echo(f"[{r.rank}] {r.doc_id}  {r.score:.3f}\n    {r.content[:120]}...\n")


@cli.command()
@click.option("--query", prompt=True)
@click.option("--index-dir", default="./data/indices")
def ask(query: str, index_dir: str):
    result = _load_pipeline(index_dir).generate(query)
    click.echo(result.answer)
    click.echo("\nSources: " + ", ".join(s["id"] for s in result.sources))


@cli.command("compare")
@click.option("--query", prompt=True)
@click.option("--index-dir", default="./data/indices")
def compare(query: str, index_dir: str):
    payload = compare_techniques(_load_pipeline(index_dir), query)
    for run in payload["runs"]:
        click.echo(f"\n{run['name']}: {', '.join(run['doc_ids']) or '(none)'}")


@cli.command()
@click.option("--eval-set", default="./eval/questions.jsonl")
@click.option("--index-dir", default="./data/indices")
def evaluate(eval_set: str, index_dir: str):
    import json

    questions = [json.loads(line) for line in Path(eval_set).read_text(encoding="utf-8").splitlines() if line]
    metrics = EvaluationRunner(_load_pipeline(index_dir)).evaluate(questions)
    click.echo(str(metrics))


@cli.command()
def serve():
    from ragmcp.server import main

    main()


if __name__ == "__main__":
    cli()
