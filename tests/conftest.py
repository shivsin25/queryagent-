import hashlib
import json
import math
from collections.abc import Callable, Sequence
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from query_tool.agents.claims import ClaimChecker
from query_tool.config.settings import Settings
from query_tool.retrieval.retriever import HybridRetriever, RerankPolicy, tokenize
from query_tool.retrieval.store import VectorStore
from query_tool.services.pipeline import QueryService

Reply = str | Exception | Callable[[list[BaseMessage]], str]


class ScriptedChatModel(BaseChatModel):
    """Returns scripted replies in order; the last reply repeats.

    Reports itself as `provider`/`model` and, if `usage` is set, that many (input, output)
    tokens per call, so cost tracking can be tested. The default is a free provider.
    """

    replies: list[Any]
    calls: list[list[BaseMessage]] = []
    provider: str = "groq"
    model: str = "scripted"
    usage: tuple[int, int] | None = None

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _get_ls_params(self, stop=None, **kwargs) -> dict:
        return {**super()._get_ls_params(stop=stop, **kwargs),
                "ls_provider": self.provider, "ls_model_name": self.model}

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.calls.append(messages)
        reply = self.replies[min(len(self.calls), len(self.replies)) - 1]
        if isinstance(reply, Exception):
            raise reply
        text = reply(messages) if callable(reply) else reply
        usage = None
        if self.usage:
            usage = {"input_tokens": self.usage[0], "output_tokens": self.usage[1],
                     "total_tokens": sum(self.usage)}
        message = AIMessage(content=text, usage_metadata=usage)
        return ChatResult(generations=[ChatGeneration(message=message)])


class FakeLLMProvider:
    def __init__(
        self, configured: bool = True, paid: bool = False, **replies: list[Reply]
    ) -> None:
        """paid=True makes every call report 1,000 input and 200 output tokens of
        gpt-4.1-mini ($0.00072 per call)."""
        self._configured = configured
        defaults: dict[str, list[Reply]] = {
            "planner": [plan_json("refund policy annual plan")],
            "writer": ["Annual plans can be refunded within 30 days [1]."],
            "reviewer": [review_json()],
            "feedback": ["1. Fix the citation."],
            "enhancer": [lambda msgs: msgs[-1].content.split("Answer to polish:\n", 1)[1]],
        }
        defaults.update(replies)
        billing = {"provider": "openai", "model": "gpt-4.1-mini", "usage": (1000, 200)}
        self.models = {
            role: ScriptedChatModel(replies=r, calls=[], **(billing if paid else {}))
            for role, r in defaults.items()
        }

    def get(self, role: str) -> BaseChatModel:
        return self.models[role]

    def is_configured(self) -> bool:
        return self._configured

    def missing_key_message(self) -> str:
        return "No API key for provider 'groq': add GROQ_API_KEY to the .env file."


def plan_json(*queries: str, standalone: str = "(ignored without history)") -> str:
    return json.dumps({"standalone_question": standalone, "search_queries": list(queries)})


def review_json(
    verdict: str = "approve", score: float = 0.95, issues=(), missing=()
) -> str:
    return json.dumps(
        {
            "grounding_score": score,
            "accuracy_score": score,
            "completeness_score": score,
            "verdict": verdict,
            "issues": list(issues),
            "unsupported_claims": [],
            "missing_information": list(missing),
        }
    )


class HashEmbedder:
    """Deterministic bag-of-words vectors so tests need no model download."""

    dim = 256

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in tokenize(text):
            vec[int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class KeywordReranker:
    """Scores by keyword overlap on the cross-encoder's logit scale: -10 (none) to +10 (all)."""

    def _score(self, query: str, text: str) -> float:
        wanted = set(tokenize(query))
        found = wanted & set(tokenize(text))
        return 20 * len(found) / max(len(wanted), 1) - 10

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        return [self._score(q, t) for q, t in pairs]


SAMPLE_DOC = b"""# Billing

Annual plans can be refunded within 30 days of purchase.
Monthly plans are not refundable.

# Support

Enterprise customers get a 1 hour response time for critical incidents.
"""


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,  # tests must not depend on the developer's .env
        llm_provider="groq",
        storage_dir=tmp_path / "storage",
        rerank_enabled=False,
        auto_seed_samples=False,
        chunk_size=120,
        chunk_overlap=0,
        groq_api_key="test",
    )


@pytest.fixture
def store(settings) -> VectorStore:
    return VectorStore(settings.chroma_dir, settings.collection_name)


@pytest.fixture
def make_service(settings, store):
    def factory(
        llm: FakeLLMProvider | None = None, seed: bool = True, rerank: bool = False
    ) -> QueryService:
        """rerank=True adds the keyword reranker, the relevance gate and the claim checker.
        The precision cutoffs have their own tests, so they're off here."""
        embedder = HashEmbedder()
        reranker = KeywordReranker() if rerank else None
        retriever = HybridRetriever(
            store, embedder, reranker, settings.candidate_k,
            RerankPolicy(min_score=settings.min_relevance) if rerank else None,
        )
        checker = ClaimChecker(reranker, settings.claim_support_threshold) if reranker else None
        service = QueryService(
            settings, llm or FakeLLMProvider(), store, embedder, retriever, checker
        )
        if seed:
            service.ingestion.ingest_bytes("billing.md", SAMPLE_DOC)
        return service

    return factory
