"""
scripts/seed_memory.py
-----------------------
CLI developer tool to seed home context into ChromaDB (long-term spatial memory).

Usage:
    # Add a single entry interactively
    python scripts/seed_memory.py

    # Add a single entry via argument
    python scripts/seed_memory.py --add "step down at kitchen entrance"

    # Add from a JSON file (batch seeding)
    python scripts/seed_memory.py --file scripts/home_context.json

    # List all stored memories
    python scripts/seed_memory.py --list

    # Query memory (test retrieval)
    python scripts/seed_memory.py --query "kitchen"

    # Delete all memories (full reset)
    python scripts/seed_memory.py --reset

Matches Phase 3 architecture:
    - ChromaDB local vector store  → backend/app/memory_modules/long_term.py
    - sentence-transformers embeds → all-MiniLM-L6-v2
    - Persists to ./chroma_db/     → survives across sessions
"""

import argparse
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path

# ── ChromaDB + embeddings ──────────────────────────────────────────────────
try:
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer
except ImportError:
    print(
        "\n[ERROR] Required packages not found.\n"
        "Run:  pip install chromadb sentence-transformers\n"
    )
    sys.exit(1)


# ── Constants ──────────────────────────────────────────────────────────────
CHROMA_PATH = Path(__file__).resolve().parent.parent / "chroma_db"
COLLECTION_NAME = "home_context"
EMBED_MODEL = "all-MiniLM-L6-v2"  # lightweight, fast, offline


# ── Client + collection helpers ────────────────────────────────────────────
def get_collection() -> chromadb.Collection:
    """Return (or create) the persistent ChromaDB collection."""
    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def get_embedder() -> SentenceTransformer:
    """Load the sentence-transformer model (cached after first run)."""
    print(f"[INFO] Loading embedding model: {EMBED_MODEL} …")
    return SentenceTransformer(EMBED_MODEL)


# ── Core operations ────────────────────────────────────────────────────────
def add_context(text: str, room: str = "unknown", embedder=None, collection=None) -> str:
    """
    Embed and store a single home-context string.

    Args:
        text      : Natural language description, e.g. 'step at kitchen entrance'
        room      : Optional room tag for filtering, e.g. 'kitchen'
        embedder  : SentenceTransformer instance (reuse across calls)
        collection: ChromaDB collection (reuse across calls)

    Returns:
        doc_id    : The UUID assigned to this entry
    """
    if embedder is None:
        embedder = get_embedder()
    if collection is None:
        collection = get_collection()

    doc_id = str(uuid.uuid4())
    embedding = embedder.encode(text).tolist()
    metadata = {
        "room": room,
        "created_at": datetime.now().isoformat(),
        "source": "seed_script",
    }

    collection.add(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[text],
        metadatas=[metadata],
    )
    return doc_id


def query_context(query_text: str, n_results: int = 5, embedder=None, collection=None) -> list[dict]:
    """
    Retrieve the top-N most relevant memory entries for a query.

    Mirrors backend/app/memory_modules/long_term.py → retrieve(query)
    """
    if embedder is None:
        embedder = get_embedder()
    if collection is None:
        collection = get_collection()

    if collection.count() == 0:
        return []

    embedding = embedder.encode(query_text).tolist()
    results = collection.query(
        query_embeddings=[embedding],
        n_results=min(n_results, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append(
            {
                "text": doc,
                "room": meta.get("room", "unknown"),
                "created_at": meta.get("created_at", ""),
                "similarity": round(1 - dist, 4),   # cosine distance → similarity
            }
        )
    return hits


def list_all(collection=None) -> list[dict]:
    """Return all stored memories (no embedding needed)."""
    if collection is None:
        collection = get_collection()

    if collection.count() == 0:
        return []

    result = collection.get(include=["documents", "metadatas"])
    entries = []
    for doc_id, doc, meta in zip(result["ids"], result["documents"], result["metadatas"]):
        entries.append(
            {
                "id": doc_id,
                "text": doc,
                "room": meta.get("room", "unknown"),
                "created_at": meta.get("created_at", ""),
            }
        )
    return entries


def reset_collection() -> int:
    """Delete ALL entries from the collection. Returns count deleted."""
    collection = get_collection()
    count = collection.count()
    if count == 0:
        return 0

    all_ids = collection.get()["ids"]
    collection.delete(ids=all_ids)
    return count


# ── Pretty printers ────────────────────────────────────────────────────────
def print_separator(char: str = "─", width: int = 60) -> None:
    print(char * width)


def print_entry(entry: dict, index: int | None = None) -> None:
    prefix = f"  [{index}]" if index is not None else "  •"
    print(f"{prefix}  \"{entry['text']}\"")
    print(f"       room={entry['room']}  |  added={entry.get('created_at', '')[:19]}")
    if "similarity" in entry:
        print(f"       similarity={entry['similarity']}")


# ── Batch seeding from JSON ────────────────────────────────────────────────
EXAMPLE_JSON = """[
  {"text": "step down at kitchen entrance",     "room": "kitchen"},
  {"text": "loose rug near sofa in living room","room": "living_room"},
  {"text": "low coffee table beside armchair",  "room": "living_room"},
  {"text": "bathroom door opens outward",        "room": "bathroom"},
  {"text": "two steps up to bedroom hallway",   "room": "hallway"}
]"""


def seed_from_file(filepath: str) -> None:
    """Load a JSON array of {text, room} objects and add them all."""
    path = Path(filepath)
    if not path.exists():
        print(f"\n[ERROR] File not found: {filepath}")
        print("\nExpected JSON format:")
        print(EXAMPLE_JSON)
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    if not isinstance(entries, list):
        print("[ERROR] JSON must be an array of objects: [{text, room}, …]")
        sys.exit(1)

    embedder = get_embedder()
    collection = get_collection()

    print(f"\n[INFO] Seeding {len(entries)} entries from {filepath} …\n")
    print_separator()
    for i, item in enumerate(entries, 1):
        text = item.get("text", "").strip()
        room = item.get("room", "unknown")
        if not text:
            print(f"  [{i}] SKIPPED — empty text")
            continue
        doc_id = add_context(text, room, embedder, collection)
        print(f"  [{i}] ADDED  {doc_id[:8]}…  \"{text}\"  (room={room})")
    print_separator()
    print(f"\n✅  Done. Collection now has {collection.count()} entries.")


# ── Interactive prompt ─────────────────────────────────────────────────────
ROOM_OPTIONS = [
    "kitchen", "living_room", "bedroom", "bathroom",
    "hallway", "dining_room", "study", "garage", "other",
]


def interactive_add(embedder, collection) -> None:
    """Walk the user through adding a single entry."""
    print("\n  Enter the home-context description.")
    print("  Example: 'step down at kitchen entrance'")
    text = input("\n  > ").strip()
    if not text:
        print("[WARN] Empty input — nothing added.")
        return

    print(f"\n  Room options: {', '.join(ROOM_OPTIONS)}")
    room = input("  Room (press Enter to skip): ").strip().lower() or "unknown"

    doc_id = add_context(text, room, embedder, collection)
    print(f"\n  ✅  Stored!  id={doc_id[:8]}…  |  \"{text}\"  (room={room})")
    print(f"      Collection size: {collection.count()} entries\n")


# ── CLI entry point ────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="seed_memory",
        description=(
            "MemoryNav — Seed ChromaDB with home-context descriptions.\n"
            "Used during development to pre-populate long-term spatial memory."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/seed_memory.py\n"
            "  python scripts/seed_memory.py --add \"step at kitchen entrance\"\n"
            "  python scripts/seed_memory.py --add \"loose rug near sofa\" --room living_room\n"
            "  python scripts/seed_memory.py --file scripts/home_context.json\n"
            "  python scripts/seed_memory.py --list\n"
            "  python scripts/seed_memory.py --query \"kitchen\"\n"
            "  python scripts/seed_memory.py --reset\n"
        ),
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--add", "-a",
        metavar="TEXT",
        help="Add a single context description (e.g. 'step at kitchen entrance')",
    )
    group.add_argument(
        "--file", "-f",
        metavar="PATH",
        help="Seed from a JSON file — array of {text, room} objects",
    )
    group.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all stored memory entries",
    )
    group.add_argument(
        "--query", "-q",
        metavar="TEXT",
        help="Test memory retrieval with a query string",
    )
    group.add_argument(
        "--reset",
        action="store_true",
        help="Delete ALL entries from ChromaDB (irreversible)",
    )
    parser.add_argument(
        "--room", "-r",
        metavar="ROOM",
        default="unknown",
        help="Room tag for --add (default: unknown)",
    )
    parser.add_argument(
        "--top", "-n",
        metavar="N",
        type=int,
        default=5,
        help="Number of results to return for --query (default: 5)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    print("\n══════════════════════════════════════════════════════")
    print("  MemoryNav  —  seed_memory.py  (ChromaDB dev tool)")
    print(f"  Store: {CHROMA_PATH}")
    print("══════════════════════════════════════════════════════\n")

    # ── --reset ──────────────────────────────────────────────────────────
    if args.reset:
        confirm = input("  ⚠️  Delete ALL memories? This cannot be undone. (yes/no): ").strip().lower()
        if confirm == "yes":
            deleted = reset_collection()
            print(f"\n  🗑️   Deleted {deleted} entries. Collection is now empty.\n")
        else:
            print("  Aborted.\n")
        return

    # ── --list ────────────────────────────────────────────────────────────
    if args.list:
        entries = list_all()
        if not entries:
            print("  Collection is empty. Use --add or --file to seed entries.\n")
            return
        print(f"  {len(entries)} entries in ChromaDB:\n")
        print_separator()
        for i, entry in enumerate(entries, 1):
            print_entry(entry, index=i)
            print()
        print_separator()
        return

    # ── --query ───────────────────────────────────────────────────────────
    if args.query:
        embedder = get_embedder()
        collection = get_collection()
        if collection.count() == 0:
            print("  Collection is empty. Seed some entries first.\n")
            return

        print(f"\n  Query: \"{args.query}\"")
        print(f"  Top {args.top} results:\n")
        print_separator()
        hits = query_context(args.query, n_results=args.top, embedder=embedder, collection=collection)
        if not hits:
            print("  No results found.\n")
        for i, hit in enumerate(hits, 1):
            print_entry(hit, index=i)
            print()
        print_separator()
        return

    # ── --file ────────────────────────────────────────────────────────────
    if args.file:
        seed_from_file(args.file)
        return

    # ── --add (single, non-interactive) ──────────────────────────────────
    if args.add:
        embedder = get_embedder()
        collection = get_collection()
        doc_id = add_context(args.add, args.room, embedder, collection)
        print(f"  ✅  Stored!  id={doc_id[:8]}…")
        print(f"      Text  : \"{args.add}\"")
        print(f"      Room  : {args.room}")
        print(f"      Total : {collection.count()} entries in collection\n")
        return

    # ── No flags → interactive mode ───────────────────────────────────────
    print("  No arguments provided → interactive mode.\n")
    print("  Commands:")
    print("    [a] Add a memory entry")
    print("    [l] List all entries")
    print("    [q] Query / test retrieval")
    print("    [x] Exit\n")

    embedder = get_embedder()
    collection = get_collection()

    while True:
        cmd = input("\n  > ").strip().lower()
        if cmd in ("x", "exit", "quit", ""):
            print("  Bye!\n")
            break
        elif cmd == "a":
            interactive_add(embedder, collection)
        elif cmd == "l":
            entries = list_all(collection)
            if not entries:
                print("\n  Collection is empty.\n")
            else:
                print(f"\n  {len(entries)} entries:\n")
                print_separator()
                for i, e in enumerate(entries, 1):
                    print_entry(e, index=i)
                    print()
                print_separator()
        elif cmd == "q":
            qtext = input("  Query text: ").strip()
            if qtext:
                hits = query_context(qtext, embedder=embedder, collection=collection)
                print(f"\n  Top {len(hits)} results for \"{qtext}\":\n")
                print_separator()
                for i, h in enumerate(hits, 1):
                    print_entry(h, index=i)
                    print()
                print_separator()
        else:
            print("  Unknown command. Use a / l / q / x.")


if __name__ == "__main__":
    main()