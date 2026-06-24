"""
MemoryNav — Long-Term Spatial Memory (ChromaDB + sentence-transformers)
backend/app/memory_modules/long_term.py

Module 3 (Memory System): persistent, semantically-searchable memory of
what's been seen and where — "where did I last see my keys" rather
than "what's in front of me right now" (that's the Risk Engine's job,
not this one). Embeds and stores free-text context (object + location
+ optional metadata) in ChromaDB, retrieves by meaning rather than
exact keyword match.

Distinct from Short-Term Memory (a separate module, not yet built):
that's a small rolling window (settings.SHORT_TERM_MEMORY_WINDOW_SECONDS)
of recent frames used for motion tracking; this is unbounded, persisted
to disk, and meant to answer questions across sessions, not frames.

Usage:

    from app.memory_modules.long_term import LongTermMemory

    memory = LongTermMemory()
    memory.add_context("saw your keys on the kitchen counter near the fridge")

    results = memory.retrieve("where are my keys")
    for r in results:
        print(r.text, r.similarity, r.metadata)

Dependencies: chromadb, sentence-transformers.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.utils import embedding_functions

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MemoryResult:
    """One retrieved memory, ranked by semantic similarity to the query."""

    id: str
    text: str
    metadata: Dict[str, Any]
    distance: float  # raw ChromaDB distance (lower = more similar, cosine space)

    @property
    def similarity(self) -> float:
        """
        Cosine similarity, roughly in [0, 1] (1 - distance) — convenience
        for display/sorting. `distance` from ChromaDB is the source of
        truth; this is just a friendlier number to print.
        """
        return 1.0 - self.distance


class LongTermMemory:
    """
    ChromaDB-backed semantic memory store. Loads the embedding model and
    opens the persistent collection once; reused across add_context()/
    retrieve() calls.
    """

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self.persist_dir = persist_dir or settings.CHROMA_PERSIST_DIR
        self.collection_name = collection_name or settings.CHROMA_COLLECTION_NAME
        self.embedding_model = embedding_model or settings.SENTENCE_TRANSFORMER_MODEL
        self.device = device or settings.device

        logger.info(
            "Opening ChromaDB at '%s', collection '%s', embeddings '%s' on '%s'",
            self.persist_dir,
            self.collection_name,
            self.embedding_model,
            self.device,
        )

        self._client = chromadb.PersistentClient(path=self.persist_dir)
        self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model,
            device=self.device,
        )
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def add_context(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        memory_id: Optional[str] = None,
    ) -> str:
        """
        Stores a piece of spatial/contextual memory — e.g. "saw keys on
        the kitchen counter near the fridge". Returns the id used (a
        fresh UUID unless you pass memory_id, e.g. to overwrite an
        existing entry).

        A timestamp is always stamped into metadata so retrieve() callers
        can sort/filter by recency without re-parsing the text.
        """
        if not text or not text.strip():
            raise ValueError("add_context() requires non-empty text.")

        doc_id = memory_id or str(uuid.uuid4())
        meta = dict(metadata or {})
        meta.setdefault("timestamp", datetime.now(timezone.utc).isoformat())

        self._collection.add(documents=[text], metadatas=[meta], ids=[doc_id])
        logger.debug("Stored memory %s: %r", doc_id, text)
        return doc_id

    def retrieve(
        self,
        query: str,
        n_results: int = 3,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[MemoryResult]:
        """
        Semantic search over stored context — finds memories by meaning,
        not exact wording (a query like "where are my keys" can match a
        stored memory that never used the word "where").

        `where` is an optional ChromaDB metadata filter (e.g.
        {"room": "kitchen"}) applied before similarity ranking.

        Returns an empty list (not an error) if nothing's been stored
        yet or nothing matches — callers don't need a try/except for
        the common "no memories yet" case.
        """
        if not query or not query.strip():
            return []

        count = self._collection.count()
        if count == 0:
            return []

        results = self._collection.query(
            query_texts=[query],
            n_results=min(n_results, count),
            where=where,
            include=["documents", "distances", "metadatas"],
        )

        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        return [
            MemoryResult(id=doc_id, text=text, metadata=meta or {}, distance=distance)
            for doc_id, text, distance, meta in zip(ids, documents, distances, metadatas)
        ]

    def delete(self, memory_id: str) -> None:
        """Removes a single memory by id."""
        self._collection.delete(ids=[memory_id])

    def clear(self) -> None:
        """Wipes the entire collection. Mainly for tests/dev resets."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self._collection.count()


if __name__ == "__main__":
    # Quick manual check: `python -m app.memory_modules.long_term`
    import tempfile

    logging.basicConfig(level=logging.INFO)
    with tempfile.TemporaryDirectory() as tmp:
        memory = LongTermMemory(persist_dir=tmp, collection_name="smoke_test")

        memory.add_context(
            "saw your keys on the kitchen counter near the fridge",
            metadata={"room": "kitchen", "object": "keys"},
        )
        memory.add_context(
            "your shoes are by the front door",
            metadata={"room": "entryway", "object": "shoes"},
        )
        memory.add_context(
            "wallet is on the bedside table",
            metadata={"room": "bedroom", "object": "wallet"},
        )

        print(f"Stored {memory.count()} memories.")

        results = memory.retrieve("where are my keys")
        for r in results:
            print(f"  [{r.similarity:.2f}] {r.text} ({r.metadata})")

        assert results, "Expected at least one match."
        assert "keys" in results[0].text
        print("Retrieval sanity check OK — top match mentions keys.")