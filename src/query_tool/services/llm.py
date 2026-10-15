"""Single place where chat models are created and invoked."""

import json
import logging
import re
import warnings
from typing import Literal, Protocol, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel, SecretStr

from query_tool.config.settings import Settings
from query_tool.core.exceptions import LLMNotConfiguredError

logger = logging.getLogger(__name__)

# Some Gemini models ignore temperature and warn on every call; the setting is still right
# for the other models, so silence just that warning.
warnings.filterwarnings("ignore", message=r".*uses fixed sampling defaults.*")

AgentRole = Literal["planner", "writer", "reviewer", "feedback", "enhancer"]
# Roles where answer quality matters most get the larger model.
PRIMARY_ROLES: frozenset[str] = frozenset({"writer", "reviewer"})

# provider -> (Settings field, .env variable, where to get a key). Ollama needs no key.
API_KEYS: dict[str, tuple[str, str, str]] = {
    "groq": ("groq_api_key", "GROQ_API_KEY", "https://console.groq.com/keys"),
    "gemini": ("google_api_key", "GOOGLE_API_KEY", "https://aistudio.google.com/apikey"),
    "openai": ("openai_api_key", "OPENAI_API_KEY", "https://platform.openai.com/api-keys"),
}

T = TypeVar("T", bound=BaseModel)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


class LLMProvider(Protocol):
    def get(self, role: AgentRole) -> BaseChatModel: ...

    def is_configured(self) -> bool: ...

    def missing_key_message(self) -> str: ...


class ChatModelFactory:
    """Builds and caches one chat model per (provider, model) pair."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._cache: dict[tuple[str, str, bool], BaseChatModel] = {}

    def _providers(self) -> set[str]:
        return {self._settings.llm_provider, self._settings.resolved_reviewer[0]}

    def _has_key(self, provider: str) -> bool:
        return provider not in API_KEYS or api_key(self._settings, provider) is not None

    def is_configured(self) -> bool:
        return all(self._has_key(p) for p in self._providers())

    def missing_key_message(self) -> str:
        missing = sorted(p for p in self._providers() if not self._has_key(p))
        return " ".join(
            f"No API key for provider '{p}': add {API_KEYS[p][1]} to the .env file "
            f"(get one at {API_KEYS[p][2]}) and restart."
            for p in missing
        )

    def route(self, role: AgentRole) -> tuple[str, str]:
        """(provider, model) used for an agent role."""
        s = self._settings
        if role == "reviewer":
            return s.resolved_reviewer
        if role in PRIMARY_ROLES:
            return s.llm_provider, s.resolved_primary_model
        return s.llm_provider, s.resolved_fast_model

    def get(self, role: AgentRole) -> BaseChatModel:
        provider, model = self.route(role)
        # Only the writer's text is streamed to users; every other call returns whole,
        # which keeps structured output on the path the providers handle best.
        key = (provider, model, role == "writer")
        if key not in self._cache:
            self._cache[key] = self._build(provider, model, streaming=role == "writer")
        return self._cache[key]

    def _build(self, provider: str, model: str, streaming: bool) -> BaseChatModel:
        s = self._settings
        if not self._has_key(provider):
            raise LLMNotConfiguredError(self.missing_key_message())
        common = {
            "model": model,
            "temperature": s.temperature,
            "disable_streaming": not streaming,
        }
        hosted = {**common, "max_retries": s.llm_max_retries, "timeout": s.llm_timeout_s}
        try:
            if provider == "groq":
                from langchain_groq import ChatGroq

                return ChatGroq(**hosted, api_key=s.groq_api_key, max_tokens=s.max_output_tokens)
            if provider == "gemini":
                from langchain_google_genai import ChatGoogleGenerativeAI

                return ChatGoogleGenerativeAI(
                    **hosted, google_api_key=s.google_api_key,
                    max_output_tokens=s.max_output_tokens,
                )
            if provider == "openai":
                from langchain_openai import ChatOpenAI

                return ChatOpenAI(
                    **hosted, api_key=s.openai_api_key, max_tokens=s.max_output_tokens,
                    stream_usage=True,  # streamed answers still report tokens, so cost is known
                )
            from langchain_ollama import ChatOllama

            return ChatOllama(**common, base_url=s.ollama_base_url,
                              num_predict=s.max_output_tokens)
        except ImportError as exc:
            raise LLMNotConfiguredError(
                f"Provider '{provider}' needs an extra package: uv sync --extra {provider}"
            ) from exc


def api_key(settings: Settings, provider: str) -> SecretStr | None:
    return getattr(settings, API_KEYS[provider][0]) if provider in API_KEYS else None


# Models that can't chat (speech, images, embeddings, moderation) are hidden from `qt models`.
_NON_CHAT_RE = re.compile(
    r"embed|tts|audio|transcribe|whisper|realtime|image|dall-e|moderation|search|guard|aqa",
    re.IGNORECASE,
)


def list_chat_models(settings: Settings, provider: str) -> list[str]:
    """Models the configured key can use. Listing makes no generation calls and costs nothing."""
    key = api_key(settings, provider)
    if provider in API_KEYS and key is None:
        raise LLMNotConfiguredError(f"Set {API_KEYS[provider][1]} in .env first.")
    if provider == "openai":
        from openai import OpenAI

        names = [m.id for m in OpenAI(api_key=key.get_secret_value()).models.list()]
        names = [n for n in names if n.startswith(("gpt-", "o"))]
    elif provider == "gemini":
        from google import genai

        client = genai.Client(api_key=key.get_secret_value())
        names = [m.name.removeprefix("models/") for m in client.models.list()
                 if "generateContent" in (m.supported_actions or [])]
    elif provider == "groq":
        from groq import Groq

        names = [m.id for m in Groq(api_key=key.get_secret_value()).models.list().data]
    else:
        raise LLMNotConfiguredError("For Ollama, run `ollama list` instead.")
    return sorted(n for n in names if not _NON_CHAT_RE.search(n))


def content_text(content: str | list) -> str:
    """Message content as plain text, unchanged otherwise (safe for streamed chunks)."""
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return content


def message_text(message: BaseMessage) -> str:
    return _THINK_RE.sub("", content_text(message.content)).strip()


def invoke_text(llm: BaseChatModel, messages: list[BaseMessage]) -> str:
    return message_text(llm.invoke(messages))


def extract_json(text: str) -> dict:
    match = _FENCED_JSON_RE.search(text)
    if match:
        return json.loads(match.group(1))
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("No JSON object found in model output")
    return json.loads(text[start : end + 1])


def invoke_structured(llm: BaseChatModel, messages: list[BaseMessage], schema: type[T]) -> T:
    """Native structured output first; fall back to prompting for JSON and parsing it."""
    try:
        result = llm.with_structured_output(schema).invoke(messages)
        if isinstance(result, schema):
            return result
        if isinstance(result, dict):
            return schema.model_validate(result)
    except Exception as exc:  # provider without tool calling, or a malformed tool call
        logger.debug("Structured output failed (%s); falling back to JSON parsing", exc)

    instruction = HumanMessage(
        content="Respond with only a JSON object matching this JSON schema, no prose:\n"
        + json.dumps(schema.model_json_schema())
    )
    raw = invoke_text(llm, [*messages, instruction])
    return schema.model_validate(extract_json(raw))
