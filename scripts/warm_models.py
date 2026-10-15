"""Download the embedding and reranking models into storage/models.

Run when building the Docker image, so a container starts without downloading anything.
"""

from query_tool.config.settings import get_settings
from query_tool.retrieval.embeddings import FastEmbedEmbedder, FastEmbedReranker

settings = get_settings()
FastEmbedEmbedder(settings.embedding_model, settings.model_cache_dir).embed_query("warm up")
FastEmbedReranker(settings.rerank_model, settings.model_cache_dir).score_pairs([("warm", "up")])
print(f"Models ready in {settings.model_cache_dir}")
