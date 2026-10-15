"""Deterministic guardrails: no LLM call, so they can't be talked out of their job.

Prompt injection is caught where it can be recognised cheaply. A question that addresses the
assistant's instructions instead of the documents is refused before any LLM sees it.
Documents are only flagged at upload, because their text is never obeyed anyway: the
writer is told to treat sources as data.
"""

import re

# Each pattern addresses the assistant itself ("ignore your previous instructions"), so a
# question about instructions in the documents ("can Nimbus ignore customer instructions?")
# is not caught.
_INJECTION_PATTERNS: dict[str, re.Pattern[str]] = {
    "override instructions": re.compile(
        r"\b(ignore|disregard|forget|override|bypass)\s+(all\s+|any\s+|the\s+)?"
        r"(previous|prior|above|earlier|preceding|your|system)\s+"
        r"(instructions?|rules?|prompts?|guidelines?|directions?)",
        re.IGNORECASE,
    ),
    "reveal the prompt": re.compile(
        r"\b(reveal|show|print|repeat|output|leak|tell me)\b.{0,40}"
        r"\b(system|hidden|initial|original|your)\s+(prompt|instructions?)",
        re.IGNORECASE,
    ),
    "role change": re.compile(
        r"\byou are (now|no longer)\b|\bpretend (that )?you (are|have) no\b|"
        r"\b(jailbreak|developer mode|DAN mode)\b",
        re.IGNORECASE,
    ),
}

BLOCKED_ANSWER = (
    "I can't help with that. I answer questions from the indexed documents, and I can't "
    "change or reveal my instructions. Ask me something about the documents instead."
)


def injection_signals(text: str) -> list[str]:
    """Names of the injection patterns found in the text."""
    return [name for name, pattern in _INJECTION_PATTERNS.items() if pattern.search(text)]
