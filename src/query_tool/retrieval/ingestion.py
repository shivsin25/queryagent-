"""Turn uploaded files into embedded chunks in the vector store."""

import bisect
import hashlib
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from query_tool.core.exceptions import DocumentError
from query_tool.core.guardrails import injection_signals
from query_tool.core.schemas import DocumentInfo, IngestResponse
from query_tool.retrieval.embeddings import Embedder
from query_tool.retrieval.extraction import (
    SUPPORTED_EXTENSIONS,
    Section,
    extract_document,
    markdown_sections,
    tidy_text,
)
from query_tool.retrieval.store import Chunk, VectorStore

logger = logging.getLogger(__name__)

__all__ = ["SUPPORTED_EXTENSIONS", "IngestionService", "Piece", "markdown_sections",
           "tidy_text"]

# Part of every doc_id, so a change to chunking re-indexes a file the next time it is ingested.
# 3: PDF pages joined, headers/footers removed, clause headings; Word headings and tables.
# 4: parent-child chunks.
CHUNKING_VERSION = 4


@dataclass(frozen=True)
class Piece:
    page: int | None  # the page the child's text starts on
    text: str  # the child: searched and reranked
    parent: int  # index of its parent within the document
    parent_text: str  # what the writer reads


def _section_text(section: Section) -> tuple[str, list[int], list[int | None]]:
    """The section's text, with the offset and page where each of its lines starts."""
    offsets: list[int] = []
    pages: list[int | None] = []
    position = 0
    for page, line in section.lines:
        offsets.append(position)
        pages.append(page)
        position += len(line) + 1
    return "\n".join(line for _, line in section.lines), offsets, pages


class IngestionService:
    """Parent-child chunking, chosen per document.

    Children (chunk_size) are small, so a question matches precisely and the cross-encoder
    judges them well. Parents give the writer the surrounding context: the whole section for
    structured documents (Markdown and Word headings, PDF clause headings), or recursive
    character windows of parent_size where there is no structure or a section is too long.
    """

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        chunk_size: int,
        chunk_overlap: int,
        parent_size: int | None = None,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._children = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
        )
        self._parent_size = max(parent_size or chunk_size, chunk_size)
        self._parents = RecursiveCharacterTextSplitter(
            chunk_size=self._parent_size, chunk_overlap=0, add_start_index=True
        )
        self._fingerprint = (
            f"v{CHUNKING_VERSION}:{chunk_size}:{chunk_overlap}:{self._parent_size}:".encode()
        )

    def split(self, filename: str, data: bytes) -> list[Piece]:
        """Every child chunk with its page and parent.

        Both start with the document title and section heading ("Agreement > 7. Security
        Incidents"), so they keep their context once retrieved on their own. Chunks may run
        across page breaks.
        """
        document = extract_document(filename, data)
        pieces: list[Piece] = []
        parent_index = 0
        for section in document.sections:
            header = " > ".join(part for part in (document.title, section.heading) if part)
            text, offsets, pages = _section_text(section)
            parents = (
                [(0, text)] if len(text) <= self._parent_size
                else [(p.metadata["start_index"], p.page_content)
                      for p in self._parents.create_documents([text])]
            )
            for parent_start, parent in parents:
                parent_text = f"{header}\n\n{parent}" if header else parent
                for child in self._children.create_documents([parent]):
                    start = parent_start + child.metadata["start_index"]
                    line = max(bisect.bisect_right(offsets, start) - 1, 0)
                    body = child.page_content
                    pieces.append(Piece(
                        page=pages[line],
                        text=f"{header}\n\n{body}" if header else body,
                        parent=parent_index,
                        parent_text=parent_text,
                    ))
                parent_index += 1
        return pieces

    def ingest_bytes(self, filename: str, data: bytes) -> DocumentInfo | None:
        """Index one file. Returns None when identical content is already indexed."""
        source = Path(filename).name
        doc_id = hashlib.sha256(self._fingerprint + data).hexdigest()[:16]
        if self._store.has_document(doc_id):
            return None

        pieces = self.split(source, data)
        if not pieces:
            raise DocumentError(f"{source}: no text found outside headings.")
        ingested_at = datetime.now(UTC).isoformat(timespec="seconds")
        chunks = [
            Chunk(
                chunk_id=f"{doc_id}-{i:04d}",
                doc_id=doc_id,
                source=source,
                text=piece.text,
                chunk_index=i,
                page=piece.page,
                ingested_at=ingested_at,
                parent_id=f"{doc_id}-p{piece.parent:04d}",
                # Stored only when it adds context, i.e. differs from the child.
                parent_text=piece.parent_text if piece.parent_text != piece.text else None,
            )
            for i, piece in enumerate(pieces)
        ]

        embeddings = self._embedder.embed_documents([c.text for c in chunks])
        # A changed file with the same name replaces its previous version.
        for old_id in self._store.doc_ids_for_source(source):
            self._store.delete_document(old_id)
        self._store.add(chunks, embeddings)
        logger.info("Indexed %s (%d chunks)", source, len(chunks))
        return DocumentInfo(
            doc_id=doc_id, source=source, chunks=len(chunks), ingested_at=ingested_at,
            warnings=_injection_warnings(chunks),
        )

    def ingest_paths(self, paths: Iterable[Path]) -> IngestResponse:
        response = IngestResponse()
        for path in _expand(paths):
            info = self.ingest_bytes(path.name, path.read_bytes())
            if info:
                response.indexed.append(info)
            else:
                response.skipped.append(f"{path.name}: already indexed")
        return response


def _injection_warnings(chunks: list[Chunk]) -> list[str]:
    """Flag text that tries to instruct an AI. It is still indexed: the writer treats every
    source as data, so the text is harmless, but the uploader should know it is there."""
    flagged = [c for c in chunks if injection_signals(c.text)]
    if not flagged:
        return []
    where = ", ".join(sorted({f"page {c.page}" if c.page else f"chunk {c.chunk_index + 1}"
                              for c in flagged}))
    logger.warning("%s contains instruction-like text (%s)", chunks[0].source, where)
    return [f"Contains text that looks like instructions to an AI assistant ({where}). It is "
            "indexed, but answers treat it as plain text and never follow it."]


def _expand(paths: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files += sorted(
                p for p in path.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS
            )
        elif path.is_file():
            files.append(path)
        else:
            raise DocumentError(f"{path}: not found.")
    return files
