"""LLM token usage and cost: recorded per call, summed per request, capped per day.

A LangChain callback attached to each graph run sees every LLM call made inside it, so no
agent needs to know about cost.
"""

import logging
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

from query_tool.core.schemas import CallUsage
from query_tool.storage.database import Database

logger = logging.getLogger(__name__)

# USD per million (input, output) tokens, from the providers' price lists (2025). Check
# https://openai.com/api/pricing and override with QT_MODEL_PRICES if they change.
DEFAULT_PRICES: dict[str, tuple[float, float]] = {
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-5": (1.25, 10.00),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5-nano": (0.05, 0.40),
}
# Free tiers cost nothing. If you pay for one of these, add its prices to QT_MODEL_PRICES.
FREE_PROVIDERS = frozenset({"groq", "gemini", "ollama"})
# A paid model with no known price is assumed to be expensive, so the budget errs safe.
UNKNOWN_PAID_PRICE = (5.00, 20.00)
# LangChain's provider names -> ours.
_PROVIDER_ALIASES = {"google_genai": "gemini", "google": "gemini"}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL,
    node TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cost_usd REAL NOT NULL,
    estimated INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS llm_calls_by_time ON llm_calls (created_at);
"""


class PriceTable:
    def __init__(self, overrides: dict[str, tuple[float, float]] | None = None) -> None:
        self._prices = {**DEFAULT_PRICES, **(overrides or {})}

    def price(self, provider: str, model: str) -> tuple[float, float]:
        # Dated snapshots ("gpt-4.1-mini-2025-04-14") use their base model's price; the
        # longest matching name wins, so "gpt-4.1-mini" is not priced as "gpt-4.1".
        matches = [name for name in self._prices if model == name or model.startswith(name + "-")]
        if matches:
            return self._prices[max(matches, key=len)]
        if provider in FREE_PROVIDERS:
            return 0.0, 0.0
        logger.warning("No price known for %s/%s; assuming %s per 1M tokens",
                       provider, model, UNKNOWN_PAID_PRICE)
        return UNKNOWN_PAID_PRICE

    def cost(self, provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
        price_in, price_out = self.price(provider, model)
        return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


class UsageLedger:
    """Every LLM call ever made, so spend can be capped across processes and restarts."""

    def __init__(self, db: Database) -> None:
        self._db = db
        db.migrate(_SCHEMA)

    def record(self, request_id: str, call: CallUsage) -> None:
        with self._db.connect() as db:
            db.execute(
                "INSERT INTO llm_calls (request_id, node, provider, model, input_tokens, "
                "output_tokens, cost_usd, estimated, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (request_id, call.node, call.provider, call.model, call.input_tokens,
                 call.output_tokens, call.cost_usd, int(call.estimated),
                 datetime.now(UTC).isoformat(timespec="seconds")),
            )

    def spent_since(self, since: datetime) -> float:
        with self._db.connect() as db:
            (total,) = db.execute(
                "SELECT COALESCE(SUM(cost_usd), 0) FROM llm_calls WHERE created_at >= ?",
                (since.isoformat(timespec="seconds"),),
            ).fetchone()
        return float(total)

    def spent_today(self) -> float:
        return self.spent_since(start_of_today())

    def by_model(self, since: datetime) -> list[dict[str, Any]]:
        with self._db.connect() as db:
            rows = db.execute(
                "SELECT provider, model, COUNT(*), SUM(input_tokens), SUM(output_tokens), "
                "SUM(cost_usd) FROM llm_calls WHERE created_at >= ? "
                "GROUP BY provider, model ORDER BY SUM(cost_usd) DESC, COUNT(*) DESC",
                (since.isoformat(timespec="seconds"),),
            ).fetchall()
        keys = ("provider", "model", "calls", "input_tokens", "output_tokens", "cost_usd")
        return [dict(zip(keys, row, strict=True)) for row in rows]


def start_of_today() -> datetime:
    return datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)


def _text_length(content: Any) -> int:
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        return sum(len(p.get("text", "")) if isinstance(p, dict) else len(str(p))
                   for p in content)
    return 0


class UsageTracker(BaseCallbackHandler):
    """Collects the usage of every LLM call in one request and writes it to the ledger."""

    def __init__(self, ledger: UsageLedger | None, prices: PriceTable, request_id: str) -> None:
        self._ledger = ledger
        self._prices = prices
        self._request_id = request_id
        self._runs: dict[UUID, tuple[str, str, str, int]] = {}
        self.calls: list[CallUsage] = []

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        meta = metadata or {}
        provider = str(meta.get("ls_provider", "unknown"))
        provider = _PROVIDER_ALIASES.get(provider, provider)
        model = str(meta.get("ls_model_name", "unknown"))
        node = str(meta.get("langgraph_node", "unknown"))
        # ~4 characters per token; used only if the provider reports no usage.
        estimate = sum(_text_length(m.content) for batch in messages for m in batch) // 4
        self._runs[run_id] = (node, provider, model, estimate)

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        run = self._runs.pop(run_id, None)
        if run is None:
            return
        node, provider, model, input_estimate = run
        generation = response.generations[0][0] if response.generations else None
        message = getattr(generation, "message", None)
        usage = getattr(message, "usage_metadata", None)
        if usage:
            input_tokens, output_tokens = usage["input_tokens"], usage["output_tokens"]
        else:
            text = getattr(generation, "text", "") or ""
            input_tokens, output_tokens = input_estimate, len(text) // 4
        call = CallUsage(
            node=node,
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=round(self._prices.cost(provider, model, input_tokens, output_tokens), 8),
            estimated=not usage,
        )
        self.calls.append(call)
        if self._ledger is not None:
            try:
                self._ledger.record(self._request_id, call)
            except Exception:  # never fail an answer because bookkeeping failed
                logger.exception("Could not record LLM usage")

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._runs.pop(run_id, None)
