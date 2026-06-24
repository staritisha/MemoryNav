"""
MemoryNav — Long-Term Memory REST Router
backend/app/api/memory_router.py

Module 3 / Backend API (Phase 6): CRUD for home context in ChromaDB —
the same store that seed_memory.py populates and the WebSocket pipeline
can query. Provides a REST interface so the frontend (Phase 7) can let
users add/browse/delete contextual memories without touching the CLI.

Routes
------
GET    /memory              list all memories (unranked)
GET    /memory?q=...        semantic search (ranked by similarity)
GET    /memory/{id}         fetch one memory by id
POST   /memory              create a new memory entry
DELETE /memory/{id}         delete one entry by id
DELETE /memory?confirm=true wipe the entire collection

Mount in main.py:

    from app.api.memory_router import router as memory_router
    app.include_router(memory_router)

The LongTermMemory instance is initialized once at app startup and
shared across all requests via a FastAPI dependency. It can safely
coexist with the WebSocket pipeline writing to the same ChromaDB file —
both use the same PersistentClient path, and ChromaDB's WAL mode
handles concurrent reads and writes.

Dependencies: fastapi, chromadb, sentence-transformers.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.memory_modules.long_term import LongTermMemory, MemoryResult

logger = logging.getLogger(__name__)

# Sentence-transformers inference is blocking — one dedicated thread so
# it doesn't freeze the event loop, same pattern as ws_stream.py.
_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="memory-embed")

# --------------------------------------------------------------------------- #
# Singleton — initialized once at app startup via init_memory()
# --------------------------------------------------------------------------- #

_memory: Optional[LongTermMemory] = None


def init_memory(instance: Optional[LongTermMemory] = None) -> LongTermMemory:
    """
    Initializes the module-level LongTermMemory singleton. Call this
    from your FastAPI lifespan (or main.py startup) before requests
    arrive. Pass an existing instance to share one across routers; omit
    to create a fresh one from config.
    """
    global _memory
    _memory = instance or LongTermMemory()
    logger.info("LongTermMemory ready (%d entries).", _memory.count())
    return _memory


def get_memory() -> LongTermMemory:
    """FastAPI dependency. Raises 503 if init_memory() hasn't been called yet."""
    if _memory is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Memory store not initialized. App may still be starting up.",
        )
    return _memory


# --------------------------------------------------------------------------- #
# Pydantic schemas
# --------------------------------------------------------------------------- #

class MemoryEntryIn(BaseModel):
    """Request body for POST /memory."""

    text: str = Field(..., min_length=1, description="Context to store, e.g. 'step at kitchen entrance'")
    room: Optional[str] = Field(None, description="Room tag stored as metadata, e.g. 'kitchen'")
    type: Optional[str] = Field(None, description="Category tag, e.g. 'hazard' or 'landmark'")
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Arbitrary extra metadata merged with room/type if both provided.",
    )

    def build_metadata(self) -> Dict[str, Any]:
        meta: Dict[str, Any] = dict(self.metadata or {})
        if self.room:
            meta["room"] = self.room
        if self.type:
            meta["type"] = self.type
        return meta


class MemoryEntryOut(BaseModel):
    """One memory entry returned by any endpoint."""

    id: str
    text: str
    metadata: Dict[str, Any]
    similarity: Optional[float] = Field(
        None,
        description="Cosine similarity to query [0–1]. Present only on search results; null on list/get.",
    )

    @classmethod
    def from_result(cls, r: MemoryResult, include_similarity: bool = False) -> "MemoryEntryOut":
        return cls(
            id=r.id,
            text=r.text,
            metadata=r.metadata,
            similarity=round(r.similarity, 4) if include_similarity else None,
        )


class ListMemoryResponse(BaseModel):
    entries: List[MemoryEntryOut]
    count: int
    query: Optional[str] = None
    total_stored: int


class CreateMemoryResponse(BaseModel):
    id: str
    text: str
    metadata: Dict[str, Any]
    message: str = "Memory stored."


class DeleteResponse(BaseModel):
    deleted_id: Optional[str] = None
    deleted_count: Optional[int] = None
    message: str


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #

router = APIRouter(prefix="/memory", tags=["memory"])


@router.get("/spatial-map", summary="Current spatial room map")
async def get_spatial_map_route() -> dict:
    """Returns the live spatial room map from the running pipeline."""
    try:
        from app.api.ws_stream import get_state
        state = get_state()
        return state.spatial_map.snapshot()
    except Exception:
        return {"current_room": "Unknown", "rooms": {}}


@router.get("", response_model=ListMemoryResponse, summary="List or search memories")
async def list_or_search_memories(
    q: Optional[str] = Query(None, description="Semantic search query. Omit to list everything."),
    n: int = Query(5, ge=1, le=50, description="Max results for semantic search (ignored when listing all)."),
    room: Optional[str] = Query(None, description="Filter by room metadata tag."),
    memory: LongTermMemory = Depends(get_memory),
) -> ListMemoryResponse:
    """
    Without `q`: returns every stored memory (unranked, no similarity score).
    With `q`: returns the top `n` semantically similar memories, ranked by
    cosine similarity. Either way, `room` filters by metadata tag.
    """
    loop = asyncio.get_running_loop()
    where = {"room": room} if room else None

    if q:
        results: List[MemoryResult] = await loop.run_in_executor(
            _EXECUTOR, lambda: memory.retrieve(q, n_results=n, where=where)
        )
        entries = [MemoryEntryOut.from_result(r, include_similarity=True) for r in results]
    else:
        results = await loop.run_in_executor(_EXECUTOR, lambda: memory.list_all())
        if room:
            results = [r for r in results if r.metadata.get("room") == room]
        entries = [MemoryEntryOut.from_result(r, include_similarity=False) for r in results]

    total = await loop.run_in_executor(_EXECUTOR, memory.count)
    return ListMemoryResponse(entries=entries, count=len(entries), query=q, total_stored=total)


@router.get("/{memory_id}", response_model=MemoryEntryOut, summary="Fetch one memory by ID")
async def get_memory_by_id(
    memory_id: str,
    memory: LongTermMemory = Depends(get_memory),
) -> MemoryEntryOut:
    """
    Fetches a single entry by its UUID. Returns 404 if not found.
    ChromaDB doesn't expose a single-get-by-id method cleanly, so this
    uses list_all() and scans — only called on explicit lookups, not
    in the hot pipeline path, so the linear scan is acceptable.
    """
    loop = asyncio.get_running_loop()
    all_results: List[MemoryResult] = await loop.run_in_executor(_EXECUTOR, memory.list_all)
    for r in all_results:
        if r.id == memory_id:
            return MemoryEntryOut.from_result(r)
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Memory '{memory_id}' not found.")


@router.post("", response_model=CreateMemoryResponse, status_code=status.HTTP_201_CREATED, summary="Store a new memory")
async def create_memory(
    body: MemoryEntryIn,
    memory: LongTermMemory = Depends(get_memory),
) -> CreateMemoryResponse:
    """
    Stores `text` as a new memory entry. `room` and `type` are
    convenience shortcuts stored as metadata; use `metadata` for any
    other tags. All three can be combined.

    Example body:
        {"text": "step at kitchen entrance", "room": "kitchen", "type": "hazard"}
    """
    meta = body.build_metadata()
    loop = asyncio.get_running_loop()
    try:
        memory_id: str = await loop.run_in_executor(
            _EXECUTOR, lambda: memory.add_context(body.text, metadata=meta)
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))

    logger.info("POST /memory created [%s]: %r", memory_id[:8], body.text)
    return CreateMemoryResponse(id=memory_id, text=body.text, metadata=meta)


@router.delete(
    "/{memory_id}",
    response_model=DeleteResponse,
    status_code=status.HTTP_200_OK,
    summary="Delete one memory by ID",
)
async def delete_memory(
    memory_id: str,
    memory: LongTermMemory = Depends(get_memory),
) -> DeleteResponse:
    """Deletes a single entry. Returns 404 if the id doesn't exist."""
    loop = asyncio.get_running_loop()

    # Verify it exists before deleting — ChromaDB's delete() is a no-op
    # on missing ids (no error), so we check first to return a proper 404.
    all_results: List[MemoryResult] = await loop.run_in_executor(_EXECUTOR, memory.list_all)
    if not any(r.id == memory_id for r in all_results):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Memory '{memory_id}' not found.")

    await loop.run_in_executor(_EXECUTOR, lambda: memory.delete(memory_id))
    logger.info("DELETE /memory/%s", memory_id[:8])
    return DeleteResponse(deleted_id=memory_id, message="Memory deleted.")


@router.delete(
    "",
    response_model=DeleteResponse,
    status_code=status.HTTP_200_OK,
    summary="Wipe the entire collection",
)
async def clear_all_memories(
    confirm: bool = Query(False, description="Must be true to wipe all memories."),
    memory: LongTermMemory = Depends(get_memory),
) -> DeleteResponse:
    """
    Deletes every stored memory. Requires `?confirm=true` as a
    deliberate speed-bump — a DELETE /memory without it returns 400.
    """
    if not confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Pass ?confirm=true to wipe all memories. This cannot be undone.",
        )
    loop = asyncio.get_running_loop()
    count_before: int = await loop.run_in_executor(_EXECUTOR, memory.count)
    await loop.run_in_executor(_EXECUTOR, memory.clear)
    logger.warning("DELETE /memory?confirm=true — wiped %d entries.", count_before)
    return DeleteResponse(deleted_count=count_before, message=f"Cleared {count_before} memories.")