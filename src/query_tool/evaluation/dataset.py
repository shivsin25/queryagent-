"""Evaluation cases: one JSON object per line in a .jsonl file."""

import json
import re
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from query_tool.core.schemas import Turn


class EvalCase(BaseModel):
    id: str
    question: str
    category: str = "general"
    answerable: bool = True
    history: list[Turn] = Field(
        default_factory=list, description="Earlier turns; the question is a follow-up to them."
    )
    documents: list[str] = Field(
        default_factory=list, description="File names to restrict the search to (all if empty)."
    )
    search_queries: list[str] = Field(
        default_factory=list,
        description="The sub-queries the planner is expected to split a multi-part question "
        "into. Used only in retrieval-only mode, which has no planner; full runs use the "
        "real planner.",
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="Phrases quoted from the documents; a chunk containing one is relevant.",
    )
    facts: list[str] = Field(
        default_factory=list,
        description="Case-insensitive regexes a correct answer must match.",
    )

    @field_validator("facts")
    @classmethod
    def _valid_regexes(cls, facts: list[str]) -> list[str]:
        for fact in facts:
            re.compile(fact)
        return facts

    @model_validator(mode="after")
    def _answerable_needs_checks(self) -> "EvalCase":
        if self.answerable and not (self.evidence or self.facts):
            raise ValueError("an answerable case needs evidence or facts")
        return self


def load_cases(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            cases.append(EvalCase.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ValueError(f"{path.name}:{number}: {exc}") from exc

    duplicates = [case_id for case_id, n in Counter(c.id for c in cases).items() if n > 1]
    if duplicates:
        raise ValueError(f"{path.name}: duplicate case ids {sorted(duplicates)}")
    return cases
