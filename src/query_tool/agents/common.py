import re

from query_tool.core.schemas import RetrievedChunk

# Matches [1], [2][3] and [2, 3].
_CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_SPACED_CITATION_RE = re.compile(r"\s*" + _CITATION_RE.pattern)
# Phrases with which an answer declines, or says the sources don't contain something.
_REFUSAL_RE = re.compile(
    r"couldn't find|could not find|cannot find|can't help with|no information|"
    r"not enough information|"
    r"(?:do|does|did)(?: not|n't) (?:mention|say|specify|state|include|contain|cover|"
    r"provide|list|name|indicate)|"
    r"(?:is|are)(?: not|n't) (?:mentioned|stated|specified|provided|listed|covered|named|"
    r"included)|"
    r"no (?:mention|details?|data) (?:of|about|on)",
    re.IGNORECASE,
)


def looks_like_refusal(text: str) -> bool:
    """The answer declines, or says the sources lack something (in its own words)."""
    return _REFUSAL_RE.search(text) is not None


def format_context(documents: list[RetrievedChunk]) -> str:
    blocks = []
    for i, doc in enumerate(documents, start=1):
        location = f"{doc.source}, page {doc.page}" if doc.page else doc.source
        blocks.append(f"[{i}] ({location})\n{doc.text}")
    return "\n\n".join(blocks)


def extract_citations(text: str) -> set[int]:
    cited: set[int] = set()
    for group in _CITATION_RE.findall(text):
        cited.update(int(n) for n in group.split(","))
    return cited


def strip_citations(text: str) -> str:
    return _SPACED_CITATION_RE.sub("", text)


def check_citations(draft: str, source_count: int) -> list[str]:
    """Deterministic checks the LLM reviewer can miss."""
    invalid = sorted(n for n in extract_citations(draft) if n < 1 or n > source_count)
    if not invalid:
        return []
    return [
        f"Citation(s) {invalid} refer to sources that do not exist "
        f"(valid range is 1-{source_count})."
    ]
