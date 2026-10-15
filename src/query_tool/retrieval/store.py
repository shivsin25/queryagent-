"""Persistent Chroma collection holding chunk text, embeddings and metadata."""

import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from query_tool.core.schemas import DocumentInfo


@dataclass(frozen=True)
class Chunk:
    """A child chunk: the unit that is searched and reranked. The writer reads its parent,
    the surrounding section, which is kept in the child's own metadata so the two can never
    get out of step."""

    chunk_id: str
    doc_id: str
    source: str
    text: str
    chunk_index: int
    page: int | None
    ingested_at: str
    parent_id: str | None = None
    parent_text: str | None = None

    def metadata(self) -> dict:
        meta = {
            "doc_id": self.doc_id,
            "source": self.source,
            "chunk_index": self.chunk_index,
            "ingested_at": self.ingested_at,
        }
        # Chroma rejects None metadata values
        if self.page is not None:
            meta["page"] = self.page
        if self.parent_id is not None:
            meta["parent_id"] = self.parent_id
        if self.parent_text is not None:
            meta["parent_text"] = self.parent_text
        return meta

    @classmethod
    def from_record(cls, chunk_id: str, text: str, meta: dict) -> "Chunk":
        return cls(
            chunk_id=chunk_id,
            doc_id=meta["doc_id"],
            source=meta["source"],
            text=text,
            chunk_index=int(meta.get("chunk_index", 0)),
            page=meta.get("page"),
            ingested_at=meta.get("ingested_at", ""),
            parent_id=meta.get("parent_id"),
            parent_text=meta.get("parent_text"),
        )


class VectorStore:
    def __init__(self, persist_dir: Path, collection_name: str) -> None:
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(persist_dir), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self._name = collection_name
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
            configuration={"hnsw": {"space": "cosine"}},
        )

    @property
    def version(self) -> str:
        """Changes on every write, in any process (it lives in the collection's metadata), so
        caches like BM25 and the answer cache see documents added by the CLI or another
        API worker. Reading it costs well under a millisecond."""
        metadata = self._client.get_collection(self._name, embedding_function=None).metadata
        return (metadata or {}).get("index_version", "0")

    def _bump_version(self) -> None:
        self._collection.modify(metadata={"index_version": uuid.uuid4().hex})

    def count(self) -> int:
        return self._collection.count()

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        self._collection.add(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embeddings,
            metadatas=[c.metadata() for c in chunks],
        )
        self._bump_version()

    def query(
        self, embedding: list[float], k: int, doc_ids: frozenset[str] | None = None
    ) -> list[tuple[Chunk, float]]:
        n = min(k, self.count())
        if n == 0:
            return []
        res = self._collection.query(
            query_embeddings=[embedding],
            n_results=n,
            where={"doc_id": {"$in": sorted(doc_ids)}} if doc_ids else None,
            include=["documents", "metadatas", "distances"],
        )
        return [
            (Chunk.from_record(cid, text, meta), 1.0 - dist)
            for cid, text, meta, dist in zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0],
                strict=True,
            )
        ]

    def all_chunks(self) -> list[Chunk]:
        res = self._collection.get(include=["documents", "metadatas"])
        return [
            Chunk.from_record(cid, text, meta)
            for cid, text, meta in zip(res["ids"], res["documents"], res["metadatas"], strict=True)
        ]

    def doc_ids_for_source(self, source: str) -> set[str]:
        res = self._collection.get(where={"source": source}, include=["metadatas"])
        return {meta["doc_id"] for meta in res["metadatas"]}

    def has_document(self, doc_id: str) -> bool:
        return bool(self._collection.get(where={"doc_id": doc_id}, limit=1)["ids"])

    def delete_document(self, doc_id: str) -> int:
        ids = self._collection.get(where={"doc_id": doc_id})["ids"]
        if ids:
            self._collection.delete(ids=ids)
            self._bump_version()
        return len(ids)

    def list_documents(self) -> list[DocumentInfo]:
        res = self._collection.get(include=["metadatas"])
        grouped: dict[str, list[dict]] = defaultdict(list)
        for meta in res["metadatas"]:
            grouped[meta["doc_id"]].append(meta)
        docs = [
            DocumentInfo(
                doc_id=doc_id,
                source=metas[0]["source"],
                chunks=len(metas),
                ingested_at=metas[0].get("ingested_at", ""),
            )
            for doc_id, metas in grouped.items()
        ]
        return sorted(docs, key=lambda d: d.source.lower())
