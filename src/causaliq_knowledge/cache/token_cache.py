"""
TokenCache: SQLite-backed cache with shared token dictionary.

Provides efficient storage for cache entries with:
- Fast indexed key lookup via SQLite
- In-memory mode via :memory:
- Concurrency support via SQLite locking
- Shared token dictionary for cross-entry compression

Note: This module is designed for future migration to causaliq-core.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


class TokenCache:
    """SQLite-backed cache with shared token dictionary.

    Attributes:
        db_path: Path to SQLite database file, or ":memory:" for in-memory.
        conn: SQLite connection (None until open() called or context entered).

    Example:
        >>> with TokenCache(":memory:") as cache:
        ...     cache.put("abc123", "test", b"hello")
        ...     data = cache.get("abc123", "test")
    """

    # SQL statements for schema creation
    _SCHEMA_SQL = """
        -- Token dictionary (grows dynamically, shared across encoders)
        CREATE TABLE IF NOT EXISTS tokens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            token TEXT UNIQUE NOT NULL,
            frequency INTEGER DEFAULT 1
        );

        -- Generic cache entries
        CREATE TABLE IF NOT EXISTS cache_entries (
            hash TEXT PRIMARY KEY,
            entry_type TEXT NOT NULL,
            data BLOB NOT NULL,
            created_at TEXT NOT NULL,
            metadata BLOB
        );

        -- Indexes for common queries
        CREATE INDEX IF NOT EXISTS idx_entry_type
            ON cache_entries(entry_type);
        CREATE INDEX IF NOT EXISTS idx_created_at
            ON cache_entries(created_at);
    """

    def __init__(self, db_path: str | Path) -> None:
        """Initialise TokenCache.

        Args:
            db_path: Path to SQLite database file. Use ":memory:" for
                in-memory database (fast, non-persistent).
        """
        self.db_path = str(db_path)
        self._conn: sqlite3.Connection | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        """Get the database connection, raising if not connected."""
        if self._conn is None:
            raise RuntimeError(
                "TokenCache not connected. Use 'with cache:' or call open()."
            )
        return self._conn

    @property
    def is_open(self) -> bool:
        """Check if the cache connection is open."""
        return self._conn is not None

    @property
    def is_memory(self) -> bool:
        """Check if this is an in-memory database."""
        return self.db_path == ":memory:"

    def open(self) -> TokenCache:
        """Open the database connection and initialise schema.

        Returns:
            self for method chaining.

        Raises:
            RuntimeError: If already connected.
        """
        if self._conn is not None:
            raise RuntimeError("TokenCache already connected.")

        self._conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,  # Allow multi-threaded access
        )
        # Enable foreign keys and WAL mode for better concurrency
        self._conn.execute("PRAGMA foreign_keys = ON")
        if not self.is_memory:
            self._conn.execute("PRAGMA journal_mode = WAL")

        self._init_schema()
        return self

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _init_schema(self) -> None:
        """Create database tables if they don't exist."""
        self.conn.executescript(self._SCHEMA_SQL)
        self.conn.commit()

    def __enter__(self) -> TokenCache:
        """Context manager entry - opens connection."""
        return self.open()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Context manager exit - closes connection."""
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Cursor]:
        """Context manager for a database transaction.

        Commits on success, rolls back on exception.

        Yields:
            SQLite cursor for executing statements.
        """
        cursor = self.conn.cursor()
        try:
            yield cursor
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        finally:
            cursor.close()

    def _utcnow_iso(self) -> str:
        """Get current UTC time as ISO 8601 string."""
        return datetime.now(timezone.utc).isoformat()

    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database.

        Args:
            table_name: Name of the table to check.

        Returns:
            True if table exists, False otherwise.
        """
        cursor = self.conn.execute(
            "SELECT name FROM sqlite_master " "WHERE type='table' AND name=?",
            (table_name,),
        )
        return cursor.fetchone() is not None

    def entry_count(self, entry_type: str | None = None) -> int:
        """Count cache entries, optionally filtered by type.

        Args:
            entry_type: If provided, count only entries of this type.

        Returns:
            Number of matching entries.
        """
        if entry_type is None:
            cursor = self.conn.execute("SELECT COUNT(*) FROM cache_entries")
        else:
            cursor = self.conn.execute(
                "SELECT COUNT(*) FROM cache_entries WHERE entry_type = ?",
                (entry_type,),
            )
        row = cursor.fetchone()
        return int(row[0]) if row else 0

    def token_count(self) -> int:
        """Count tokens in the dictionary.

        Returns:
            Number of tokens.
        """
        cursor = self.conn.execute("SELECT COUNT(*) FROM tokens")
        row = cursor.fetchone()
        return int(row[0]) if row else 0
