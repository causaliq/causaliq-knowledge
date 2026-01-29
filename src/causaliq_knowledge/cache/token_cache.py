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
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:  # pragma: no cover
    from causaliq_knowledge.cache.encoders.base import EntryEncoder


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
            hash TEXT NOT NULL,
            entry_type TEXT NOT NULL,
            data BLOB NOT NULL,
            created_at TEXT NOT NULL,
            metadata BLOB,
            hit_count INTEGER DEFAULT 0,
            last_accessed_at TEXT,
            PRIMARY KEY (hash, entry_type)
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
        # In-memory token dictionary for fast lookup
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}
        # Registered encoders for auto-encoding (entry_type -> encoder)
        self._encoders: dict[str, EntryEncoder] = {}

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
        self._load_token_dict()

    def _load_token_dict(self) -> None:
        """Load token dictionary from database into memory."""
        cursor = self.conn.execute("SELECT id, token FROM tokens")
        self._token_to_id.clear()
        self._id_to_token.clear()
        for row in cursor:
            token_id, token = row[0], row[1]
            self._token_to_id[token] = token_id
            self._id_to_token[token_id] = token

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

    def list_entry_types(self) -> list[str]:
        """List all distinct entry types in the cache.

        Returns:
            List of entry type names found in the cache.

        Example:
            >>> with TokenCache(":memory:") as cache:
            ...     cache.register_encoder("llm", LLMEntryEncoder())
            ...     cache.put_data("h1", "llm", {"data": "test"})
            ...     cache.list_entry_types()
            ['llm']
        """
        cursor = self.conn.execute(
            "SELECT DISTINCT entry_type FROM cache_entries ORDER BY entry_type"
        )
        return [row[0] for row in cursor.fetchall()]

    def token_count(self) -> int:
        """Count tokens in the dictionary.

        Returns:
            Number of tokens.
        """
        cursor = self.conn.execute("SELECT COUNT(*) FROM tokens")
        row = cursor.fetchone()
        return int(row[0]) if row else 0

    def total_hits(self, entry_type: str | None = None) -> int:
        """Get total cache hits across all entries.

        Args:
            entry_type: If provided, count only hits for this entry type.

        Returns:
            Total hit count.
        """
        if entry_type is None:
            cursor = self.conn.execute(
                "SELECT COALESCE(SUM(hit_count), 0) FROM cache_entries"
            )
        else:
            cursor = self.conn.execute(
                "SELECT COALESCE(SUM(hit_count), 0) FROM cache_entries "
                "WHERE entry_type = ?",
                (entry_type,),
            )
        row = cursor.fetchone()
        return int(row[0]) if row else 0

    def get_or_create_token(self, token: str) -> int:
        """Get token ID, creating a new entry if needed.

        This method is used by encoders to compress strings to integer IDs.
        The token dictionary grows dynamically as new tokens are encountered.

        Args:
            token: The string token to look up or create.

        Returns:
            Integer ID for the token (1-65535 range).

        Raises:
            ValueError: If token dictionary exceeds uint16 capacity.
        """
        # Fast path: check in-memory cache
        if token in self._token_to_id:
            return self._token_to_id[token]

        # Slow path: insert into database
        cursor = self.conn.execute(
            "INSERT INTO tokens (token) VALUES (?) RETURNING id",
            (token,),
        )
        token_id: int = cursor.fetchone()[0]
        self.conn.commit()

        # Check uint16 capacity (max 65,535 tokens)
        if token_id > 65535:  # pragma: no cover
            raise ValueError(
                f"Token dictionary exceeded uint16 capacity: {token_id}"
            )

        # Update in-memory cache
        self._token_to_id[token] = token_id
        self._id_to_token[token_id] = token

        return token_id

    def get_token(self, token_id: int) -> str | None:
        """Get token string by ID.

        This method is used by decoders to expand integer IDs back to strings.

        Args:
            token_id: The integer ID to look up.

        Returns:
            The token string, or None if not found.
        """
        return self._id_to_token.get(token_id)

    # ========================================================================
    # Cache entry operations
    # ========================================================================

    def put(
        self,
        hash: str,
        entry_type: str,
        data: bytes,
        metadata: bytes | None = None,
    ) -> None:
        """Store a cache entry.

        Args:
            hash: Unique identifier for the entry (e.g. SHA-256 truncated).
            entry_type: Type of entry (e.g. 'llm', 'graph', 'score').
            data: Binary data to store.
            metadata: Optional binary metadata.
        """
        self.conn.execute(
            "INSERT OR REPLACE INTO cache_entries "
            "(hash, entry_type, data, created_at, metadata) "
            "VALUES (?, ?, ?, ?, ?)",
            (hash, entry_type, data, self._utcnow_iso(), metadata),
        )
        self.conn.commit()

    def get(self, hash: str, entry_type: str) -> bytes | None:
        """Retrieve a cache entry and increment hit count.

        Args:
            hash: Unique identifier for the entry.
            entry_type: Type of entry to retrieve.

        Returns:
            Binary data if found, None otherwise.
        """
        cursor = self.conn.execute(
            "SELECT data FROM cache_entries "
            "WHERE hash = ? AND entry_type = ?",
            (hash, entry_type),
        )
        row = cursor.fetchone()
        if row:
            # Increment hit count and update last accessed time
            self.conn.execute(
                "UPDATE cache_entries SET hit_count = hit_count + 1, "
                "last_accessed_at = ? WHERE hash = ? AND entry_type = ?",
                (self._utcnow_iso(), hash, entry_type),
            )
            self.conn.commit()
            result: bytes = row[0]
            return result
        return None

    def get_with_metadata(
        self, hash: str, entry_type: str
    ) -> tuple[bytes, bytes | None] | None:
        """Retrieve a cache entry with its metadata.

        Args:
            hash: Unique identifier for the entry.
            entry_type: Type of entry to retrieve.

        Returns:
            Tuple of (data, metadata) if found, None otherwise.
        """
        cursor = self.conn.execute(
            "SELECT data, metadata FROM cache_entries "
            "WHERE hash = ? AND entry_type = ?",
            (hash, entry_type),
        )
        row = cursor.fetchone()
        return (row[0], row[1]) if row else None

    def exists(self, hash: str, entry_type: str) -> bool:
        """Check if a cache entry exists.

        Args:
            hash: Unique identifier for the entry.
            entry_type: Type of entry to check.

        Returns:
            True if entry exists, False otherwise.
        """
        cursor = self.conn.execute(
            "SELECT 1 FROM cache_entries " "WHERE hash = ? AND entry_type = ?",
            (hash, entry_type),
        )
        return cursor.fetchone() is not None

    def delete(self, hash: str, entry_type: str) -> bool:
        """Delete a cache entry.

        Args:
            hash: Unique identifier for the entry.
            entry_type: Type of entry to delete.

        Returns:
            True if entry was deleted, False if it didn't exist.
        """
        cursor = self.conn.execute(
            "DELETE FROM cache_entries WHERE hash = ? AND entry_type = ?",
            (hash, entry_type),
        )
        self.conn.commit()
        return cursor.rowcount > 0

    # ========================================================================
    # Encoder registration and auto-encoding operations
    # ========================================================================

    def register_encoder(self, entry_type: str, encoder: EntryEncoder) -> None:
        """Register an encoder for a specific entry type.

        Once registered, `put_data()` and `get_data()` will automatically
        encode/decode entries of this type using the registered encoder.

        Args:
            entry_type: Type identifier (e.g. 'llm', 'json', 'score').
            encoder: EntryEncoder instance for this type.

        Example:
            >>> from causaliq_knowledge.cache.encoders import JsonEncoder
            >>> with TokenCache(":memory:") as cache:
            ...     cache.register_encoder("json", JsonEncoder())
            ...     cache.put_data("key1", "json", {"msg": "hello"})
        """
        self._encoders[entry_type] = encoder

    def get_encoder(self, entry_type: str) -> EntryEncoder | None:
        """Get the registered encoder for an entry type.

        Args:
            entry_type: Type identifier to look up.

        Returns:
            The registered encoder, or None if not registered.
        """
        return self._encoders.get(entry_type)

    def has_encoder(self, entry_type: str) -> bool:
        """Check if an encoder is registered for an entry type.

        Args:
            entry_type: Type identifier to check.

        Returns:
            True if encoder is registered, False otherwise.
        """
        return entry_type in self._encoders

    def put_data(
        self,
        hash: str,
        entry_type: str,
        data: Any,
        metadata: Any | None = None,
    ) -> None:
        """Store data using the registered encoder for the entry type.

        This method automatically encodes the data using the encoder
        registered for the given entry_type. Use `put()` for raw bytes.

        Args:
            hash: Unique identifier for the entry.
            entry_type: Type of entry (must have registered encoder).
            data: Data to encode and store.
            metadata: Optional metadata to encode and store.

        Raises:
            KeyError: If no encoder is registered for entry_type.

        Example:
            >>> with TokenCache(":memory:") as cache:
            ...     cache.register_encoder("json", JsonEncoder())
            ...     cache.put_data("abc", "json", {"key": "value"})
        """
        encoder = self._encoders[entry_type]
        blob = encoder.encode(data, self)
        meta_blob = (
            encoder.encode(metadata, self) if metadata is not None else None
        )
        self.put(hash, entry_type, blob, meta_blob)

    def get_data(self, hash: str, entry_type: str) -> Any | None:
        """Retrieve and decode data using the registered encoder.

        This method automatically decodes the data using the encoder
        registered for the given entry_type. Use `get()` for raw bytes.

        Args:
            hash: Unique identifier for the entry.
            entry_type: Type of entry (must have registered encoder).

        Returns:
            Decoded data if found, None otherwise.

        Raises:
            KeyError: If no encoder is registered for entry_type.

        Example:
            >>> with TokenCache(":memory:") as cache:
            ...     cache.register_encoder("json", JsonEncoder())
            ...     cache.put_data("abc", "json", {"key": "value"})
            ...     data = cache.get_data("abc", "json")
        """
        blob = self.get(hash, entry_type)
        if blob is None:
            return None
        encoder = self._encoders[entry_type]
        return encoder.decode(blob, self)

    def get_data_with_metadata(
        self, hash: str, entry_type: str
    ) -> tuple[Any, Any | None] | None:
        """Retrieve and decode data with metadata using registered encoder.

        Args:
            hash: Unique identifier for the entry.
            entry_type: Type of entry (must have registered encoder).

        Returns:
            Tuple of (decoded_data, decoded_metadata) if found, None otherwise.
            metadata may be None if not stored.

        Raises:
            KeyError: If no encoder is registered for entry_type.
        """
        result = self.get_with_metadata(hash, entry_type)
        if result is None:
            return None
        data_blob, meta_blob = result
        encoder = self._encoders[entry_type]
        decoded_data = encoder.decode(data_blob, self)
        decoded_meta = encoder.decode(meta_blob, self) if meta_blob else None
        return (decoded_data, decoded_meta)

    # ========================================================================
    # Import/Export operations
    # ========================================================================

    def export_entries(
        self,
        output_dir: Path,
        entry_type: str,
        fmt: str | None = None,
    ) -> int:
        """Export cache entries to human-readable files.

        Each entry is exported to a separate file named `{hash}.{ext}` where
        ext is determined by the format or encoder's default_export_format.

        Args:
            output_dir: Directory to write exported files to. Created if
                it doesn't exist.
            entry_type: Type of entries to export (must have registered
                encoder).
            fmt: Export format (e.g. 'json', 'yaml'). If None, uses the
                encoder's default_export_format.

        Returns:
            Number of entries exported.

        Raises:
            KeyError: If no encoder is registered for entry_type.

        Example:
            >>> from pathlib import Path
            >>> from causaliq_knowledge.cache import TokenCache
            >>> from causaliq_knowledge.cache.encoders import JsonEncoder
            >>> with TokenCache(":memory:") as cache:
            ...     cache.register_encoder("json", JsonEncoder())
            ...     cache.put_data("abc123", "json", {"key": "value"})
            ...     count = cache.export_entries(Path("./export"), "json")
            ...     # Creates ./export/abc123.json
        """
        encoder = self._encoders[entry_type]
        ext = fmt or encoder.default_export_format

        # Create output directory if needed
        output_dir.mkdir(parents=True, exist_ok=True)

        # Query all entries of this type
        cursor = self.conn.execute(
            "SELECT hash, data FROM cache_entries WHERE entry_type = ?",
            (entry_type,),
        )

        count = 0
        for hash_val, blob in cursor:
            # Decode the blob to get original data
            data = encoder.decode(blob, self)
            # Export to file using encoder's export method
            file_path = output_dir / f"{hash_val}.{ext}"
            encoder.export(data, file_path)
            count += 1

        return count

    def import_entries(
        self,
        input_dir: Path,
        entry_type: str,
    ) -> int:
        """Import human-readable files into the cache.

        Each file is imported with its stem (filename without extension)
        used as the cache hash. The encoder's import_() method reads the
        file and the data is encoded before storage.

        Args:
            input_dir: Directory containing files to import.
            entry_type: Type to assign to imported entries (must have
                registered encoder).

        Returns:
            Number of entries imported.

        Raises:
            KeyError: If no encoder is registered for entry_type.
            FileNotFoundError: If input_dir doesn't exist.

        Example:
            >>> from pathlib import Path
            >>> from causaliq_knowledge.cache import TokenCache
            >>> from causaliq_knowledge.cache.encoders import JsonEncoder
            >>> with TokenCache(":memory:") as cache:
            ...     cache.register_encoder("json", JsonEncoder())
            ...     count = cache.import_entries(Path("./import"), "json")
            ...     # Imports all files from ./import as "json" entries
        """
        encoder = self._encoders[entry_type]

        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        count = 0
        for file_path in input_dir.iterdir():
            if file_path.is_file():
                # Use filename (without extension) as hash
                hash_val = file_path.stem
                # Import data using encoder
                data = encoder.import_(file_path)
                # Encode and store
                self.put_data(hash_val, entry_type, data)
                count += 1

        return count
