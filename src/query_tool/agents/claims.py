"""Deterministic citation check: is each cited sentence about what its sources say?

Every cited sentence is scored against each source it cites with the local cross-encoder.
This reliably catches a sentence pinned to the wrong source, and costs no LLM call. It does
not catch a wrong number in an otherwise on-topic sentence; that is the LLM reviewer's job.
"""

import re

from query_tool.agents.common import extract_citations, strip_citations
from query_tool.core.schemas import ClaimCheck, RetrievedChunk
from query_tool.retrieval.embeddings import Reranker

# A sentence ends at . ! or ? (not inside 99.9), or after a citation marker that follows one.
_SENTENCE_BREAK_RE = re.compile(r"(?<=[.!?])\s+(?=[^\s\[])|(?<=\])\s+(?=[A-Z])")
_LINE_PREFIX_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)]|#{1,6})\s+")
# Shorter fragments are headings or lead-ins ("In short:"), not factual claims.
MIN_CLAIM_WORDS = 4


def _paragraphs(text: str) -> list[str]:
    """Join soft-wrapped lines; blank lines, list items, headings and table rows start anew."""
    blocks: list[list[str]] = [[]]
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or _LINE_PREFIX_RE.match(line) or stripped.startswith("|"):
            blocks.append([])
        if stripped:
            blocks[-1].append(_LINE_PREFIX_RE.sub("", line).strip())
    return [" ".join(block) for block in blocks if block]


def split_claims(text: str) -> list[tuple[str, list[int]]]:
    """Return (sentence without citation markers, cited source numbers) pairs."""
    claims: list[tuple[str, list[int]]] = []
    for paragraph in _paragraphs(text):
        for sentence in _SENTENCE_BREAK_RE.split(paragraph):
            plain = " ".join(strip_citations(sentence).replace("**", "").split())
            if len(plain.split()) >= MIN_CLAIM_WORDS and not plain.endswith(":"):
                claims.append((plain, sorted(extract_citations(sentence))))
    return claims


def claim_support(claims: list[ClaimCheck]) -> float | None:
    checked = [c.supported for c in claims if c.supported is not None]
    return sum(checked) / len(checked) if checked else None


def claim_issues(claims: list[ClaimCheck]) -> list[str]:
    return [
        f'"{c.text}" cites {c.citations}, but that source does not appear to support it. '
        "Cite the source that does, or remove the sentence."
        for c in claims
        if c.supported is False
    ]


class ClaimChecker:
    def __init__(self, reranker: Reranker, threshold: float) -> None:
        self._reranker = reranker
        self._threshold = threshold

    def check(self, draft: str, documents: list[RetrievedChunk]) -> list[ClaimCheck]:
        claims = split_claims(draft)
        pairs: list[tuple[str, str]] = []
        owners: list[int] = []
        for i, (text, cited) in enumerate(claims):
            for n in cited:
                if 1 <= n <= len(documents):  # out-of-range numbers are reported separately
                    pairs.append((text, documents[n - 1].text))
                    owners.append(i)

        # A sentence citing several sources is supported if any one of them supports it.
        scores = self._reranker.score_pairs(pairs) if pairs else []
        best: dict[int, float] = {}
        for owner, score in zip(owners, scores, strict=True):
            best[owner] = max(score, best.get(owner, score))

        return [
            ClaimCheck(
                text=text,
                citations=cited,
                support=round(best[i], 2) if i in best else None,
                supported=best[i] >= self._threshold if i in best else None,
            )
            for i, (text, cited) in enumerate(claims)
        ]
