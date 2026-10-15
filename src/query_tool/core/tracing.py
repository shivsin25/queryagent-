import functools
import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from query_tool.core.schemas import TraceEvent
from query_tool.core.state import GraphState

logger = logging.getLogger(__name__)

NodeFn = Callable[[GraphState], dict[str, Any]]


def traced(node: str) -> Callable[[NodeFn], NodeFn]:
    """Time a graph node and append a TraceEvent.

    A node may return a ``_detail`` dict; it is moved into the trace event
    instead of being written to the graph state.
    """

    def decorator(fn: NodeFn) -> NodeFn:
        @functools.wraps(fn)
        def wrapper(state: GraphState) -> dict[str, Any]:
            started_at = datetime.now(UTC)
            start = time.perf_counter()
            update = dict(fn(state) or {})
            detail = update.pop("_detail", {})
            event = TraceEvent(
                node=node,
                started_at=started_at,
                duration_ms=round((time.perf_counter() - start) * 1000, 1),
                detail=detail,
            )
            logger.info("node=%s duration_ms=%.0f", node, event.duration_ms)
            return {**update, "trace": [event]}

        return wrapper

    return decorator
