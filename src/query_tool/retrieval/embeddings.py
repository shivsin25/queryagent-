"""Local ONNX models via fastembed: free, CPU-only, no API calls."""

import os
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

# Windows without Developer Mode can't symlink; the HF cache copies files instead and says so.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


class Embedder(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class Reranker(Protocol):
    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        """Relevance of each text to its query, as cross-encoder logits."""
        ...


class _LazyModel:
    """Loads (and on first use downloads) the model only when it is needed."""

    def __init__(self, model_name: str, cache_dir: Path) -> None:
        self.model_name = model_name
        self._cache_dir = cache_dir
        self._model: Any = None
        self._lock = threading.Lock()

    def _load(self) -> Any:
        raise NotImplementedError

    @property
    def model(self) -> Any:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._cache_dir.mkdir(parents=True, exist_ok=True)
                    self._model = self._load()
        return self._model


class FastEmbedEmbedder(_LazyModel):
    def _load(self) -> Any:
        from fastembed import TextEmbedding

        return TextEmbedding(model_name=self.model_name, cache_dir=str(self._cache_dir))

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self.model.passage_embed(list(texts))]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self.model.query_embed(text))).tolist()


class FastEmbedReranker(_LazyModel):
    def _load(self) -> Any:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        return TextCrossEncoder(model_name=self.model_name, cache_dir=str(self._cache_dir))

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        return [float(s) for s in self.model.rerank_pairs(list(pairs))]
