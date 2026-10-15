"""Command line entry point: `qt --help`."""

import uuid
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from query_tool.config.settings import get_settings
from query_tool.core.exceptions import QueryToolError
from query_tool.core.logging import configure_logging
from query_tool.core.schemas import QueryRequest, QueryResponse

app = typer.Typer(help="Multi-agent RAG query tool.", no_args_is_help=True)
console = Console()


def _service():
    from query_tool.services.pipeline import QueryService

    settings = get_settings()
    configure_logging("WARNING")
    return QueryService.from_settings(settings)


@app.command()
def ingest(paths: Annotated[list[Path], typer.Argument(help="Files or folders to index.")]):
    """Index documents (.pdf, .docx, .txt, .md)."""
    result = _service().ingestion.ingest_paths(paths)
    for doc in result.indexed:
        console.print(f"[green]indexed[/] {doc.source} ({doc.chunks} chunks)")
    for reason in result.skipped:
        console.print(f"[yellow]skipped[/] {reason}")


@app.command()
def seed():
    """Index the bundled sample documents."""
    ingest([get_settings().sample_docs_dir])


@app.command()
def docs():
    """List indexed documents."""
    table = Table("doc_id", "source", "chunks", "ingested_at")
    for doc in _service().store.list_documents():
        table.add_row(doc.doc_id, doc.source, str(doc.chunks), doc.ingested_at)
    console.print(table)


@app.command()
def delete(doc_id: str):
    """Remove a document from the index."""
    removed = _service().store.delete_document(doc_id)
    console.print(f"Removed {removed} chunks." if removed else "[red]Document not found.[/]")


@app.command()
def ask(
    question: str,
    top_k: Annotated[int | None, typer.Option(help="Chunks to retrieve.")] = None,
    max_revisions: Annotated[int | None, typer.Option(help="Reviewer loop limit.")] = None,
    session: Annotated[
        str | None, typer.Option(help="Conversation id; follow-ups use its earlier turns.")
    ] = None,
    doc: Annotated[
        list[str] | None,
        typer.Option(help="Answer only from this document (file name or id; repeatable)."),
    ] = None,
    retry_of: Annotated[
        str | None, typer.Option(help="Request id of an answer you weren't satisfied with.")
    ] = None,
    why: Annotated[str | None, typer.Option(help="What was wrong with that answer.")] = None,
    trace: Annotated[bool, typer.Option(help="Show per-node timings.")] = False,
):
    """Ask a question and print the reviewed answer with sources."""
    service = _service()
    request = QueryRequest(
        question=question, top_k=top_k, max_revisions=max_revisions, session_id=session,
        doc_ids=_doc_ids(service, doc), retry_of=retry_of, retry_feedback=why,
    )
    if not _answer(service, request, trace):
        raise typer.Exit(1)


@app.command()
def chat(
    session: Annotated[
        str | None, typer.Option(help="Resume a conversation by id.")
    ] = None,
    doc: Annotated[
        list[str] | None,
        typer.Option(help="Answer only from this document (file name or id; repeatable)."),
    ] = None,
):
    """Interactive conversation; follow-up questions keep their context. Empty line exits.

    Type `/retry [what was wrong]` to have the last answer redone with a wider search.
    """
    service = _service()
    doc_ids = _doc_ids(service, doc)
    session = session or uuid.uuid4().hex[:12]
    scope = f", answering from {', '.join(doc)}" if doc else ""
    console.print(f"Session [bold]{session}[/]{scope}. Ask a question, `/retry [why]` to redo "
                  "the last answer, or press Enter to quit.")
    last: QueryResponse | None = None
    while line := console.input("[bold cyan]> [/]").strip():
        if line.startswith("/retry"):
            if last is None:
                console.print("[yellow]Nothing to retry yet.[/]")
                continue
            why = line.removeprefix("/retry").strip() or None
            request = QueryRequest(question=last.question, session_id=session,
                                   doc_ids=doc_ids, retry_of=last.request_id,
                                   retry_feedback=why)
        elif len(line) < 3:
            console.print("[yellow]Ask a longer question.[/]")
            continue
        else:
            request = QueryRequest(question=line, session_id=session, doc_ids=doc_ids)
        last = _answer(service, request, trace=False) or last


def _doc_ids(service, refs: list[str] | None) -> list[str] | None:
    if not refs:
        return None
    try:
        return service.resolve_documents(refs)
    except QueryToolError as exc:
        console.print(f"[red]{exc}[/] See `qt docs` for what is indexed.")
        raise typer.Exit(1) from exc


def _answer(service, request: QueryRequest, trace: bool) -> QueryResponse | None:
    """Run one question through the pipeline and print it. None on failure."""
    response: QueryResponse | None = None
    try:
        with console.status("Running pipeline...") as status:
            for event in service.stream(request):
                if event.event == "step":
                    status.update(f"Finished [bold]{event.node}[/]...")
                elif event.event == "result":
                    response = QueryResponse.model_validate(event.data)
                elif event.event == "error":
                    console.print(f"[red]Error:[/] {event.data['message']}")
                    return None
    except QueryToolError as exc:
        console.print(f"[red]{exc}[/]")
        return None
    assert response is not None

    if response.retry:
        verdict = ("[green]improved[/]" if response.retry.improved
                   else "[yellow]no better than before[/]")
        console.print(f"Attempt {response.attempt}, {verdict}. What changed:")
        for change in response.retry.changes:
            console.print(f"  [dim]- {change}[/]")
        if response.retry.note:
            console.print(f"  [yellow]{response.retry.note}[/]")
    if response.standalone_question != response.question:
        console.print(f"[dim]Understood as: {response.standalone_question}[/]")
    if len(response.search_queries) > 1:
        console.print(f"[dim]Searched for: {' | '.join(response.search_queries)}[/]")
    badge = "[green]approved[/]" if response.approved else "[yellow]not approved[/]"
    console.print(Panel(Markdown(response.answer), title=f"Answer ({badge})", expand=False))
    stats = [f"revisions {response.revisions}", f"{response.latency_ms / 1000:.1f}s",
             "cached, $0" if response.cached else f"${response.usage.cost_usd:.4f}"]
    if response.review:
        r = response.review
        stats = [f"grounding {r.grounding_score:.2f}", f"accuracy {r.accuracy_score:.2f}",
                 f"completeness {r.completeness_score:.2f}", *stats]
    retries = (f"{response.retries_left} retr{'y' if response.retries_left == 1 else 'ies'} "
               "left" if response.retries_left else "no retries left")
    console.print(" | ".join(stats) + f" | request {response.request_id} | {retries}")
    for citation in response.citations:
        page = f" p.{citation.page}" if citation.page else ""
        console.print(f"[cyan][{citation.index}][/] {citation.source}{page}")
    for warning in response.warnings:
        console.print(f"[yellow]! {warning}[/]")
    if trace:
        table = Table("node", "ms")
        for step in response.trace:
            table.add_row(step.node, f"{step.duration_ms:.0f}")
        console.print(table)
    return response


@app.command("eval")
def evaluate(
    retrieval_only: Annotated[
        bool, typer.Option("--retrieval-only", help="Score retrieval only: no LLM calls, free.")
    ] = False,
    dataset: Annotated[Path | None, typer.Option(help="Eval cases (.jsonl).")] = None,
    case: Annotated[
        list[str] | None, typer.Option("--case", help="Run only this case id (repeatable).")
    ] = None,
    limit: Annotated[int | None, typer.Option(help="Run only the first N cases.")] = None,
    top_k: Annotated[int | None, typer.Option(help="Chunks to retrieve.")] = None,
    baseline: Annotated[
        str | None, typer.Option(help="Report to compare against: a path, or 'latest'.")
    ] = None,
    delay: Annotated[float, typer.Option(help="Seconds to wait between cases.")] = 0.0,
):
    """Score the pipeline on a labelled question set and save a JSON report."""
    from query_tool.evaluation.dataset import load_cases
    from query_tool.evaluation.runner import EvalReport, EvalRunner, latest_report

    settings = get_settings()
    mode = "retrieval" if retrieval_only else "full"
    path = dataset or settings.eval_dataset
    cases = load_cases(path)
    if case:
        cases = [c for c in cases if c.id in set(case)]
    cases = cases[:limit] if limit else cases
    if not cases:
        console.print("[red]No cases selected.[/]")
        raise typer.Exit(1)

    previous: EvalReport | None = None
    if baseline:
        baseline_path = (
            latest_report(settings.eval_dir, mode) if baseline == "latest" else Path(baseline)
        )
        if baseline_path is None:
            console.print(f"[yellow]No earlier {mode} report to compare against.[/]")
        else:
            previous = EvalReport.load(baseline_path)
            console.print(f"Comparing against [bold]{baseline_path.name}[/]")

    service = _service()
    service.seed_if_empty()
    by_id = {c.id: c for c in cases}
    console.print(f"Running {len(cases)} cases ({mode})...")
    if mode == "full":
        console.print("[dim]Full mode calls the LLM: roughly 4-9 calls per answerable case.[/]")
    try:
        report = EvalRunner(service).run(
            cases, mode, path.name, top_k=top_k, delay_s=delay,
            on_result=lambda r: console.print(_case_line(r, by_id[r.id])),
        )
    except QueryToolError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc

    _print_summary(report, previous)
    if report.skipped:
        console.print(f"[dim]Skipped {len(report.skipped)} follow-up case(s); resolving them "
                      f"needs the LLM: {', '.join(report.skipped)}[/]")
    console.print(f"Report saved to {report.save(settings.eval_dir)}")


def _case_line(result, case) -> str:
    mark = {True: "[green]PASS[/]", False: "[red]FAIL[/]", None: "[dim]  - [/]"}[result.passed]
    line = f"{mark} {result.id:<28}"
    if result.error:
        return f"{line} [red]{result.error[:100]}[/]"
    if result.answerable and result.recall is not None:
        missing = [p for p, r in zip(case.evidence, result.evidence_ranks, strict=True) if not r]
        line += f" recall {result.recall:.2f}"
        if missing:
            line += f" [yellow]missing {missing}[/]"
    else:
        line += f" chunks {result.retrieved}"
    if result.answer is not None:
        if result.answerable:
            missed = [f for f, hit in zip(case.facts, result.fact_hits, strict=True) if not hit]
            line += f" | facts {sum(result.fact_hits)}/{len(result.fact_hits)}"
            if missed:
                line += f" [yellow]missed {missed}[/]"
        line += " | abstained" if result.abstained else ""
        line += " | searched again" if result.researched else ""
        line += f" | {'approved' if result.approved else 'not approved'}"
    return f"{line} | {result.latency_ms / 1000:.1f}s"


def _format_metric(value: float | None, metric) -> str:
    if value is None:
        return "-"
    if metric.money:
        return f"${value:.4f}"
    if metric.percent:
        return f"{value:.0%}"
    return f"{value:,.2f}" if value < 100 else f"{value:,.0f}"


def _format_delta(delta: float, metric) -> str:
    if metric.money:
        return f"{'+' if delta > 0 else '-'}${abs(delta):.4f}"
    if metric.percent:
        return f"{delta * 100:+.0f} pts"
    return f"{delta:+,.2f}" if abs(delta) < 100 else f"{delta:+,.0f}"


def _print_summary(report, previous) -> None:
    from query_tool.evaluation.runner import METRICS

    table = Table("metric", "value", title=f"Summary ({report.mode})", title_justify="left")
    if previous:
        table.add_column("baseline")
        table.add_column("change")
    for key, value in report.summary.items():
        metric = METRICS[key]
        row = [metric.label, _format_metric(value, metric)]
        if previous:
            old = previous.summary.get(key)
            row.append(_format_metric(old, metric))
            change = ""
            if value is not None and old is not None and value != old:
                better = (value > old) == metric.higher_is_better
                change = f"[{'green' if better else 'red'}]{_format_delta(value - old, metric)}[/]"
            row.append(change)
        table.add_row(*row)
    console.print(table)

    columns = ["pass_rate", "retrieval_recall", "answer_accuracy"]
    by_category = Table("category", "cases", *(METRICS[c].label for c in columns),
                        title="By category", title_justify="left")
    for name, summary in report.by_category.items():
        count = sum(r.category == name for r in report.cases)
        by_category.add_row(name, str(count),
                            *(_format_metric(summary.get(c), METRICS[c]) for c in columns))
    console.print(by_category)


def _database():
    from query_tool.storage.database import Database

    return Database(get_settings().app_db)


@app.command()
def usage(days: Annotated[int, typer.Option(help="Days to include, counting today.")] = 1):
    """LLM calls, tokens and cost, against the daily budget."""
    from datetime import timedelta

    from query_tool.services.usage import UsageLedger, start_of_today

    settings = get_settings()
    ledger = UsageLedger(_database())
    spent = ledger.spent_today()
    budget = settings.daily_budget_usd
    limit = f" of ${budget:.2f} daily budget ({spent / budget:.0%})" if budget else ""
    console.print(f"Spent today (UTC): [bold]${spent:.4f}[/]{limit}")

    since = start_of_today() - timedelta(days=days - 1)
    table = Table("provider", "model", "calls", "input tokens", "output tokens", "cost",
                  title=f"Last {days} day(s)", title_justify="left")
    for row in ledger.by_model(since):
        table.add_row(row["provider"], row["model"], str(row["calls"]),
                      f"{row['input_tokens']:,}", f"{row['output_tokens']:,}",
                      f"${row['cost_usd']:.4f}")
    console.print(table)


@app.command()
def feedback(
    down: Annotated[bool, typer.Option("--down", help="Only thumbs-down ratings.")] = False,
    export: Annotated[
        Path | None, typer.Option(help="Write eval case stubs (.jsonl) for these questions.")
    ] = None,
):
    """Ratings users gave answers; export the bad ones as eval cases to fill in."""
    import json

    from query_tool.services.query_log import QueryLog

    items = QueryLog(_database()).feedback(rating="down" if down else None)
    if not items:
        console.print("No feedback yet.")
        return
    table = Table("rating", "question", "comment", "request")
    for item in items:
        mark = "[green]up[/]" if item.rating == "up" else "[red]down[/]"
        table.add_row(mark, item.question, item.comment or "", item.request_id)
    console.print(table)

    if export:
        # Stubs fail validation until evidence or facts are added, so they can't be run
        # half-finished by accident.
        lines = [json.dumps({"id": f"feedback-{i.request_id}", "category": "feedback",
                             "question": i.question, "evidence": [], "facts": [],
                             "_answer_given": i.answer, "_comment": i.comment})
                 for i in items if i.rating == "down"]
        export.write_text("\n".join(lines) + "\n", encoding="utf-8")
        console.print(f"Wrote {len(lines)} case stub(s) to {export}. Fill in evidence or facts, "
                      "then add them to the eval dataset.")


@app.command("clear-cache")
def clear_cache():
    """Forget all cached answers."""
    from query_tool.services.cache import AnswerCache

    removed = AnswerCache(_database(), get_settings().cache_ttl_hours).clear()
    console.print(f"Removed {removed} cached answer(s).")


@app.command()
def models(
    provider: Annotated[
        str | None, typer.Argument(help="groq, gemini or openai (default: the configured one).")
    ] = None,
):
    """List the chat models your API key can use. Uses no tokens or generation quota."""
    from query_tool.services.llm import list_chat_models

    settings = get_settings()
    provider = provider or settings.llm_provider
    try:
        names = list_chat_models(settings, provider)
    except QueryToolError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc

    in_use: set[str] = set()
    if provider == settings.llm_provider:
        in_use |= {settings.resolved_primary_model, settings.resolved_fast_model}
    reviewer_provider, reviewer_model = settings.resolved_reviewer
    if provider == reviewer_provider:
        in_use.add(reviewer_model)
    for name in names:
        console.print(f"[green]{name}  (configured)[/]" if name in in_use else name)
    for name in sorted(in_use - set(names)):
        console.print(f"[red]{name}  (configured, but your key can't use it)[/]")


@app.command()
def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    reload: Annotated[bool, typer.Option(help="Auto-reload on code changes.")] = False,
    workers: Annotated[
        int, typer.Option(help="Worker processes. All state is shared safely between them.")
    ] = 1,
):
    """Start the FastAPI server."""
    import uvicorn

    uvicorn.run("query_tool.api.main:app", host=host, port=port, reload=reload,
                workers=None if reload else workers)


if __name__ == "__main__":
    app()
