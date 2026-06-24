"""
MemoryNav — User Preference Memory (SQLite)
backend/app/memory_modules/preferences.py

Module 3 (Memory System): small, structured, long-lived user settings —
speech speed, language, mobility flags. Unlike spatial memory (ChromaDB,
a separate module) this is plain relational data: one row per user,
read constantly (the Risk Engine's user_context_weight wants mobility
flags on every assessment) and written rarely (only when the user
changes a setting via voice or the app).

SQLite over a flat config file because this needs concurrent-safe
access while the voice loop is running, and because it's a natural fit
for the Backend API (Module 7) to expose as a settings endpoint later
without extra plumbing.

Usage:

    from app.memory_modules.preferences import PreferencesStore

    store = PreferencesStore()
    prefs = store.load()                 # UserPreferences, defaults if none saved yet
    prefs.speech_rate_wpm = 200
    store.save(prefs)

    store.add_mobility_flag("bad_knee")  # convenience read-modify-write

Dependencies: none beyond the standard library.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List, Optional

from app.config import settings

logger = logging.getLogger(__name__)

DEFAULT_USER_ID = "default_user"


@dataclass
class UserPreferences:
    """
    One user's saved settings.

    mobility_flags is a free-form tag list rather than fixed boolean
    columns (e.g. "bad_knee", "uses_cane", "limited_mobility") — new
    flags don't need a schema migration, and the Risk Engine's
    user_context_weight lookup (not built yet) can key off whatever
    tags are present without this module needing to know what they mean.
    """

    user_id: str = DEFAULT_USER_ID
    speech_rate_wpm: int = field(default_factory=lambda: settings.TTS_RATE_WPM)
    language: str = "en"
    mobility_flags: List[str] = field(default_factory=list)
    alert_suppression_seconds: float = field(
        default_factory=lambda: settings.ALERT_SUPPRESSION_WINDOW_SECONDS
    )
    updated_at: Optional[str] = None  # ISO 8601, set by the store on save


class PreferencesStore:
    """
    SQLite-backed CRUD for UserPreferences. One row per user_id.

    Thread-safety: a fresh connection is opened per operation (cheap
    for SQLite, and avoids cross-thread connection-sharing issues since
    the voice loop, capture loop, and a future FastAPI handler may all
    call this from different threads). A lock serializes writes so two
    concurrent saves can't interleave.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = Path(db_path or settings.SQLITE_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()
        self._init_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_preferences (
                    user_id                     TEXT PRIMARY KEY,
                    speech_rate_wpm             INTEGER NOT NULL,
                    language                    TEXT NOT NULL,
                    mobility_flags              TEXT NOT NULL,
                    alert_suppression_seconds   REAL NOT NULL,
                    updated_at                  TEXT NOT NULL
                )
                """
            )
            # Migration guard: add the column to any DB created before this
            # field existed. SQLite doesn't support ALTER TABLE ADD COLUMN
            # IF NOT EXISTS, so we probe first.
            cols = {row[1] for row in conn.execute("PRAGMA table_info(user_preferences)")}
            if "alert_suppression_seconds" not in cols:
                conn.execute(
                    "ALTER TABLE user_preferences ADD COLUMN "
                    "alert_suppression_seconds REAL NOT NULL DEFAULT 4.0"
                )

    def load(self, user_id: str = DEFAULT_USER_ID) -> UserPreferences:
        """
        Returns saved preferences for `user_id`, or sane defaults (from
        config.py) if nothing's been saved yet — first run should never
        crash on a missing row.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM user_preferences WHERE user_id = ?", (user_id,)
            ).fetchone()

        if row is None:
            logger.info("No saved preferences for '%s'; returning defaults.", user_id)
            return UserPreferences(user_id=user_id)

        return UserPreferences(
            user_id=row["user_id"],
            speech_rate_wpm=row["speech_rate_wpm"],
            language=row["language"],
            mobility_flags=json.loads(row["mobility_flags"]),
            alert_suppression_seconds=row["alert_suppression_seconds"],
            updated_at=row["updated_at"],
        )

    def save(self, prefs: UserPreferences) -> UserPreferences:
        """
        Upserts `prefs` (insert if new user_id, update otherwise).
        Returns the saved copy with `updated_at` stamped, so callers
        don't need a separate load() to see it.
        """
        stamped = replace(prefs, updated_at=datetime.now(timezone.utc).isoformat())
        with self._write_lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO user_preferences
                    (user_id, speech_rate_wpm, language, mobility_flags,
                     alert_suppression_seconds, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    speech_rate_wpm             = excluded.speech_rate_wpm,
                    language                    = excluded.language,
                    mobility_flags              = excluded.mobility_flags,
                    alert_suppression_seconds   = excluded.alert_suppression_seconds,
                    updated_at                  = excluded.updated_at
                """,
                (
                    stamped.user_id,
                    stamped.speech_rate_wpm,
                    stamped.language,
                    json.dumps(stamped.mobility_flags),
                    stamped.alert_suppression_seconds,
                    stamped.updated_at,
                ),
            )
        return stamped

    def delete(self, user_id: str = DEFAULT_USER_ID) -> bool:
        """Removes a user's saved preferences. Returns True if a row was deleted."""
        with self._write_lock, self._connect() as conn:
            cursor = conn.execute("DELETE FROM user_preferences WHERE user_id = ?", (user_id,))
        return cursor.rowcount > 0

    def has_mobility_flag(self, flag: str, user_id: str = DEFAULT_USER_ID) -> bool:
        """Convenience for the Risk Engine's user_context_weight lookup."""
        return flag in self.load(user_id).mobility_flags

    def add_mobility_flag(self, flag: str, user_id: str = DEFAULT_USER_ID) -> UserPreferences:
        """Adds a flag if not already present; leaves other fields untouched."""
        prefs = self.load(user_id)
        if flag not in prefs.mobility_flags:
            prefs.mobility_flags.append(flag)
        return self.save(prefs)

    def remove_mobility_flag(self, flag: str, user_id: str = DEFAULT_USER_ID) -> UserPreferences:
        """Removes a flag if present; no-op (and no extra write) if it isn't."""
        prefs = self.load(user_id)
        if flag not in prefs.mobility_flags:
            return prefs
        prefs.mobility_flags.remove(flag)
        return self.save(prefs)


if __name__ == "__main__":
    # Quick manual check: `python -m app.memory_modules.preferences`
    import tempfile

    logging.basicConfig(level=logging.INFO)
    with tempfile.TemporaryDirectory() as tmp:
        store = PreferencesStore(db_path=f"{tmp}/test_prefs.db")

        prefs = store.load()
        print("Defaults:               ", prefs)

        prefs.speech_rate_wpm = 200
        prefs.language = "es"
        store.save(prefs)
        store.add_mobility_flag("bad_knee")

        reloaded = store.load()
        print("After save + add_flag:  ", reloaded)
        assert reloaded.speech_rate_wpm == 200
        assert reloaded.language == "es"
        assert "bad_knee" in reloaded.mobility_flags

        store.remove_mobility_flag("bad_knee")
        assert not store.has_mobility_flag("bad_knee")

        deleted = store.delete()
        assert deleted and store.load().mobility_flags == []
        print("CRUD round-trip OK.")