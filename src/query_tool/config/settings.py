from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]

LLMProviderName = Literal["groq", "gemini", "openai", "ollama"]

# (primary, fast) model per provider, used when QT_PRIMARY_MODEL / QT_FAST_MODEL are unset.
# Providers retire models; `qt models` lists what your key can use.
DEFAULT_MODELS: dict[str, tuple[str, str]] = {
    "groq": ("llama-3.3-70b-versatile", "llama-3.1-8b-instant"),
    "gemini": ("gemini-3.8-flash", "gemini-3.5-flash-lite"),
    "openai": ("gpt-4.1-mini", "gpt-4.1-nano"),
    "ollama": ("llama3.1:8b", "llama3.1:8b"),
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_prefix="QT_",
        extra="ignore",
        populate_by_name=True,
    )

    app_name: str = "Query Tool"
    log_level: str = "INFO"

    # LLM
    llm_provider: LLMProviderName = "groq"
    groq_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("GROQ_API_KEY", "QT_GROQ_API_KEY")
    )
    google_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("GOOGLE_API_KEY", "QT_GOOGLE_API_KEY")
    )
    openai_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("OPENAI_API_KEY", "QT_OPENAI_API_KEY")
    )
    ollama_base_url: str = "http://localhost:11434"
    primary_model: str | None = None
    fast_model: str | None = None
    # The reviewer can run on a different provider so one model doesn't grade its own work.
    # The model defaults to that provider's primary model.
    reviewer_provider: LLMProviderName | None = None
    reviewer_model: str | None = None
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 3
    # Caps every LLM reply. Output tokens cost the most, so this bounds the worst case per call.
    max_output_tokens: int = Field(default=2048, ge=256)

    # Cost control. New queries are refused once today's (UTC) recorded spend reaches the
    # budget; a request already running finishes. Unset to disable.
    daily_budget_usd: float | None = Field(default=0.50, ge=0)
    # USD per million (input, output) tokens, added to or overriding the built-in prices,
    # e.g. QT_MODEL_PRICES='{"gpt-4.1-mini": [0.4, 1.6]}'.
    model_prices: dict[str, tuple[float, float]] = Field(default_factory=dict)
    # Approved answers are reused for this long, until documents, settings or prompts change.
    # 0 disables the cache.
    cache_ttl_hours: float = Field(default=168, ge=0)

    # Retrieval
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    rerank_enabled: bool = True
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    storage_dir: Path = PROJECT_ROOT / "storage"
    collection_name: str = "documents"
    # Parent-child chunking: children of chunk_size characters are searched and reranked; the
    # writer reads their parent, the section they belong to, split at parent_chunk_size if
    # longer. Parents stay within the cross-encoder's 512-token input for claim checks.
    chunk_size: int = 400
    chunk_overlap: int = 80
    parent_chunk_size: int | None = 1500
    top_k: int = Field(default=5, ge=1, le=20)
    candidate_k: int = Field(default=20, ge=1, le=100)
    # Cross-encoder scores are logits: off-topic chunks score about -11, relevant ones above 0.
    # If even the best chunk scores below this, the question is answered as out of scope
    # without calling the writer. Only applies with the reranker on. Unset to disable.
    min_relevance: float | None = -5.0
    # Which further chunks reach the writer, beyond the best one for each query (the
    # question, plus each part of a multi-part question). See README "Retrieval tuning".
    min_chunk_score: float | None = -7.0  # absolute floor; unset to disable
    rerank_margin: float | None = 3.0  # within this of its query's best; unset to disable
    # Fused candidates scored by the cross-encoder, the slowest retrieval stage. 12 is the
    # smallest value that kept every piece of evidence on the eval set (-18% latency vs 20).
    rerank_k: int = Field(default=12, ge=1, le=100)

    # Agent loop
    max_revisions: int = Field(default=2, ge=0, le=5)
    approval_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    # Each cited sentence is scored against its sources with the cross-encoder. A sentence
    # whose best score is below the threshold blocks approval as a likely miscitation.
    claim_check_enabled: bool = True
    claim_support_threshold: float = -4.0
    # When the reviewer reports facts missing from the sources, search again (locally, no LLM
    # call) up to this many times, adding at most research_k new chunks each time.
    max_research_rounds: int = Field(default=1, ge=0, le=3)
    research_k: int = Field(default=3, ge=1, le=10)

    # Conversations
    history_turns: int = Field(default=3, ge=0, le=20)
    # How many times one question may be answered again after the first answer. Each retry
    # searches wider and is told what the earlier answers lacked.
    max_retries_per_question: int = Field(default=3, ge=0, le=10)

    # Services
    api_url: str = "http://localhost:8000"
    cors_origins: list[str] = ["http://localhost:8501"]
    max_upload_mb: int = 20
    auto_seed_samples: bool = True
    sample_docs_dir: Path = PROJECT_ROOT / "data" / "sample_docs"
    eval_dataset: Path = PROJECT_ROOT / "data" / "eval" / "nimbus.jsonl"

    @property
    def resolved_primary_model(self) -> str:
        return self.primary_model or DEFAULT_MODELS[self.llm_provider][0]

    @property
    def resolved_fast_model(self) -> str:
        return self.fast_model or DEFAULT_MODELS[self.llm_provider][1]

    @property
    def resolved_reviewer(self) -> tuple[LLMProviderName, str]:
        provider = self.reviewer_provider or self.llm_provider
        if self.reviewer_model:
            return provider, self.reviewer_model
        if provider == self.llm_provider:
            return provider, self.resolved_primary_model
        return provider, DEFAULT_MODELS[provider][0]

    @property
    def app_db(self) -> Path:
        return self.storage_dir / "app.sqlite3"

    @property
    def chroma_dir(self) -> Path:
        return self.storage_dir / "chroma"

    @property
    def model_cache_dir(self) -> Path:
        return self.storage_dir / "models"

    @property
    def eval_dir(self) -> Path:
        return self.storage_dir / "eval"


@lru_cache
def get_settings() -> Settings:
    return Settings()
