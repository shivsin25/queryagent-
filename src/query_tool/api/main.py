import logging
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    HTTPException,
    Path,
    Request,
    UploadFile,
    status,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from query_tool import __version__
from query_tool.config.settings import get_settings
from query_tool.core.exceptions import (
    BudgetExceededError,
    DocumentError,
    LLMNotConfiguredError,
    NotFoundError,
    QueryToolError,
    RetryLimitError,
)
from query_tool.core.logging import configure_logging, request_id_var
from query_tool.core.schemas import (
    SESSION_ID_PATTERN,
    DocumentInfo,
    FeedbackRequest,
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    StreamEvent,
    Turn,
)
from query_tool.services.pipeline import QueryService
from query_tool.services.usage import start_of_today

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")


def get_service(request: Request) -> QueryService:
    return request.app.state.service


ServiceDep = Annotated[QueryService, Depends(get_service)]


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health(service: ServiceDep) -> HealthResponse:
    return service.health()


@router.post("/query", response_model=QueryResponse, tags=["query"])
def query(body: QueryRequest, service: ServiceDep) -> QueryResponse:
    return service.run(body)


@router.post("/query/stream", tags=["query"], response_class=StreamingResponse)
def query_stream(body: QueryRequest, service: ServiceDep):
    """NDJSON stream: a `step` event per pipeline node, then a `result` (or `error`) event."""
    service.check_request(body)  # fail with 503, 404 or 429 before the stream starts

    def lines(events: Iterator[StreamEvent]) -> Iterator[str]:
        for event in events:
            yield event.model_dump_json() + "\n"

    return StreamingResponse(lines(service.stream(body)), media_type="application/x-ndjson")


@router.get("/documents", response_model=list[DocumentInfo], tags=["documents"])
def list_documents(service: ServiceDep) -> list[DocumentInfo]:
    return service.store.list_documents()


@router.post("/documents", response_model=IngestResponse, tags=["documents"])
async def upload_documents(
    files: list[UploadFile], service: ServiceDep
) -> IngestResponse:
    limit = service.settings.max_upload_mb * 1024 * 1024
    response = IngestResponse()
    for upload in files:
        name = upload.filename or "upload"
        data = await upload.read()
        if len(data) > limit:
            response.skipped.append(f"{name}: larger than {service.settings.max_upload_mb} MB")
            continue
        try:
            info = await run_in_threadpool(service.ingestion.ingest_bytes, name, data)
        except DocumentError as exc:
            response.skipped.append(str(exc))
            continue
        if info:
            response.indexed.append(info)
        else:
            response.skipped.append(f"{name}: already indexed")
    return response


@router.delete(
    "/documents/{doc_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["documents"]
)
def delete_document(doc_id: str, service: ServiceDep) -> None:
    if not service.store.delete_document(doc_id):
        raise DocumentError(f"Document {doc_id} not found.")


SessionId = Annotated[str, Path(pattern=SESSION_ID_PATTERN)]


@router.get("/sessions/{session_id}", response_model=list[Turn], tags=["sessions"])
def get_session(session_id: SessionId, service: ServiceDep) -> list[Turn]:
    """The stored turns of a conversation, oldest first. Unknown ids return an empty list."""
    return service.conversations.history(session_id)


@router.delete(
    "/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["sessions"]
)
def delete_session(session_id: SessionId, service: ServiceDep) -> None:
    service.conversations.delete(session_id)


@router.post("/feedback", status_code=status.HTTP_204_NO_CONTENT, tags=["feedback"])
def add_feedback(body: FeedbackRequest, service: ServiceDep) -> None:
    """Rate an answer by its request_id. Rating again replaces the earlier rating."""
    if not service.query_log.add_feedback(body.request_id, body.rating, body.comment):
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Request {body.request_id} not found.")


@router.get("/usage", tags=["system"])
def usage(service: ServiceDep) -> dict:
    """Today's LLM calls and cost by model (UTC day), against the daily budget."""
    return {
        "spent_today_usd": round(service.usage.spent_today(), 6),
        "daily_budget_usd": service.settings.daily_budget_usd,
        "by_model": service.usage.by_model(start_of_today()),
    }


def create_app(service: QueryService | None = None) -> FastAPI:
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if service is None:
            configure_logging(settings.log_level)
            app.state.service = QueryService.from_settings(settings)
            if settings.auto_seed_samples:
                await run_in_threadpool(app.state.service.seed_if_empty)
        else:
            app.state.service = service
        yield

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description="Self-correcting multi-agent RAG with grounded citations.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        token = request_id_var.set(request_id)
        start = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        elapsed = (time.perf_counter() - start) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{elapsed:.0f}"
        logger.info("%s %s -> %s (%.0f ms)", request.method, request.url.path,
                    response.status_code, elapsed)
        return response

    @app.exception_handler(LLMNotConfiguredError)
    async def llm_not_configured(_: Request, exc: LLMNotConfiguredError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(BudgetExceededError)
    async def budget_exceeded(_: Request, exc: BudgetExceededError) -> JSONResponse:
        return JSONResponse(status_code=429, content={"detail": str(exc)})

    @app.exception_handler(RetryLimitError)
    async def retry_limit(_: Request, exc: RetryLimitError) -> JSONResponse:
        return JSONResponse(status_code=429, content={"detail": str(exc)})

    @app.exception_handler(NotFoundError)
    async def not_found(_: Request, exc: NotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(DocumentError)
    async def document_error(_: Request, exc: DocumentError) -> JSONResponse:
        code = 404 if "not found" in str(exc) else 422
        return JSONResponse(status_code=code, content={"detail": str(exc)})

    @app.exception_handler(QueryToolError)
    async def app_error(_: Request, exc: QueryToolError) -> JSONResponse:
        logger.exception("Unhandled application error")
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    app.include_router(router)
    return app


app = create_app()
