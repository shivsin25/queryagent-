"""Deterministic scoring functions. No LLM judge, so results are cheap and repeatable."""

import math
import re
from collections.abc import Iterable, Sequence

from query_tool.agents.common import looks_like_refusal

# Nodes that make one LLM call each (the reviewer may make a second on fallback).
LLM_NODES = frozenset({"preprocess", "writer", "reviewer", "feedback", "enhancer"})



def normalize(text: str) -> str:
    """Lower-case, drop Markdown emphasis and collapse whitespace so phrases match chunks."""
    return " ".join(text.replace("*", "").lower().split())


def evidence_ranks(evidence: Sequence[str], chunks: Sequence[str]) -> list[int | None]:
    """1-based rank of the first chunk containing each evidence phrase, None if absent."""
    normalized = [normalize(c) for c in chunks]
    ranks: list[int | None] = []
    for phrase in evidence:
        needle = normalize(phrase)
        ranks.append(next((i for i, c in enumerate(normalized, 1) if needle in c), None))
    return ranks


def chunk_precision(evidence: Sequence[str], chunks: Sequence[str]) -> float | None:
    """Share of retrieved chunks containing any evidence phrase.

    A lower bound on true precision: a chunk can be relevant without quoting the evidence.
    It still moves the right way when irrelevant chunks stop being retrieved.
    """
    if not evidence or not chunks:
        return None
    needles = [normalize(e) for e in evidence]
    return sum(any(n in normalize(c) for n in needles) for c in chunks) / len(chunks)


def recall(ranks: Sequence[int | None]) -> float | None:
    return sum(r is not None for r in ranks) / len(ranks) if ranks else None


def reciprocal_rank(ranks: Sequence[int | None]) -> float | None:
    if not ranks:
        return None
    found = [r for r in ranks if r is not None]
    return 1.0 / min(found) if found else 0.0


def fact_hits(facts: Sequence[str], answer: str) -> list[bool]:
    return [re.search(fact, answer, re.IGNORECASE) is not None for fact in facts]


def is_abstention(answer: str, nodes: Iterable[str]) -> bool:
    """The pipeline declined to answer, either by routing or in the writer's own words."""
    return "no_context" in nodes or looks_like_refusal(answer)


def llm_calls(nodes: Iterable[str]) -> int:
    return sum(node in LLM_NODES for node in nodes)


def mean(values: Iterable[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def rate(flags: Iterable[bool | None]) -> float | None:
    return mean(None if f is None else float(f) for f in flags)


def percentile(values: Sequence[float], q: float) -> float | None:
    """Nearest-rank percentile, q in (0, 100]."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q / 100 * len(ordered)) - 1)]
