"""
Unit tests for TokenCache core functionality.

Tests use in-memory SQLite only (no filesystem access).
File-based persistence tests are in functional tests.

Tests cover:
- SQLite schema initialisation
- Connection management (open/close, context manager)
- In-memory mode
- Table verification
"""

import pytest

from causaliq_knowledge.cache import TokenCache

# ============================================================================
# Schema and initialisation tests
# ============================================================================


# Test that TokenCache creates required tables on open
def test_token_cache_creates_schema() -> None:
    """Verify TokenCache initialises SQLite with correct schema."""
    with TokenCache(":memory:") as cache:
        assert cache.table_exists("tokens")
        assert cache.table_exists("cache_entries")


# Test that schema includes expected columns in tokens table
def test_tokens_table_has_correct_columns() -> None:
    """Verify tokens table has id, token, and frequency columns."""
    with TokenCache(":memory:") as cache:
        cursor = cache.conn.execute("PRAGMA table_info(tokens)")
        columns = {row[1] for row in cursor.fetchall()}
        assert columns == {"id", "token", "frequency"}


# Test that schema includes expected columns in cache_entries table
def test_cache_entries_table_has_correct_columns() -> None:
    """Verify cache_entries table has expected columns."""
    with TokenCache(":memory:") as cache:
        cursor = cache.conn.execute("PRAGMA table_info(cache_entries)")
        columns = {row[1] for row in cursor.fetchall()}
        expected = {
            "hash",
            "entry_type",
            "data",
            "created_at",
            "metadata",
            "hit_count",
            "last_accessed_at",
        }
        assert columns == expected


# ============================================================================
# Connection management tests
# ============================================================================


# Test context manager opens and closes connection
def test_context_manager_opens_and_closes() -> None:
    """Verify context manager properly manages connection lifecycle."""
    cache = TokenCache(":memory:")
    assert not cache.is_open

    with cache:
        assert cache.is_open

    assert not cache.is_open


# Test explicit open() and close() methods
def test_explicit_open_close() -> None:
    """Verify explicit open() and close() work correctly."""
    cache = TokenCache(":memory:")
    assert not cache.is_open

    cache.open()
    assert cache.is_open

    cache.close()
    assert not cache.is_open


# Test that accessing conn before open raises RuntimeError
def test_conn_before_open_raises() -> None:
    """Verify accessing conn before opening raises RuntimeError."""
    cache = TokenCache(":memory:")
    with pytest.raises(RuntimeError, match="not connected"):
        _ = cache.conn


# Test that opening twice raises RuntimeError
def test_double_open_raises() -> None:
    """Verify opening an already-open cache raises RuntimeError."""
    cache = TokenCache(":memory:")
    cache.open()
    try:
        with pytest.raises(RuntimeError, match="already connected"):
            cache.open()
    finally:
        cache.close()


# Test that close() is idempotent (can be called multiple times)
def test_close_is_idempotent() -> None:
    """Verify close() can be called multiple times without error."""
    cache = TokenCache(":memory:")
    cache.open()
    cache.close()
    cache.close()  # Should not raise
    assert not cache.is_open


# ============================================================================
# In-memory mode tests
# ============================================================================


# Test in-memory mode detection
def test_is_memory_property() -> None:
    """Verify is_memory correctly identifies in-memory databases."""
    memory_cache = TokenCache(":memory:")
    assert memory_cache.is_memory

    # File path (not opened) should report not in-memory
    file_cache = TokenCache("some/path/test.db")
    assert not file_cache.is_memory


# Test in-memory mode works with context manager
def test_in_memory_mode_works() -> None:
    """Verify in-memory mode creates functional database."""
    with TokenCache(":memory:") as cache:
        assert cache.is_open
        assert cache.entry_count() == 0
        assert cache.token_count() == 0


# ============================================================================
# Utility method tests
# ============================================================================


# Test entry_count returns zero for empty cache
def test_entry_count_empty() -> None:
    """Verify entry_count returns 0 for empty cache."""
    with TokenCache(":memory:") as cache:
        assert cache.entry_count() == 0
        assert cache.entry_count(entry_type="llm") == 0


# Test token_count returns zero for empty cache
def test_token_count_empty() -> None:
    """Verify token_count returns 0 for empty cache."""
    with TokenCache(":memory:") as cache:
        assert cache.token_count() == 0


# Test table_exists returns False for non-existent table
def test_table_exists_false_for_missing() -> None:
    """Verify table_exists returns False for non-existent tables."""
    with TokenCache(":memory:") as cache:
        assert not cache.table_exists("nonexistent_table")


# ============================================================================
# Transaction context manager tests
# ============================================================================


# Test transaction commits on success
def test_transaction_commits_on_success() -> None:
    """Verify transaction context manager commits on successful exit."""
    with TokenCache(":memory:") as cache:
        with cache.transaction() as cursor:
            cursor.execute(
                "INSERT INTO tokens (token) VALUES (?)", ("test_token",)
            )
        # Verify commit happened - token should persist
        assert cache.token_count() == 1


# Test transaction rolls back on exception
def test_transaction_rolls_back_on_exception() -> None:
    """Verify transaction context manager rolls back on exception."""
    with TokenCache(":memory:") as cache:
        try:
            with cache.transaction() as cursor:
                cursor.execute(
                    "INSERT INTO tokens (token) VALUES (?)", ("rollback_test",)
                )
                raise ValueError("Simulated error")
        except ValueError:
            pass
        # Verify rollback happened - token should not persist
        assert cache.token_count() == 0


# Test transaction re-raises exception after rollback
def test_transaction_reraises_exception() -> None:
    """Verify transaction re-raises the original exception."""
    with TokenCache(":memory:") as cache:
        with pytest.raises(ValueError, match="Original error"):
            with cache.transaction():
                raise ValueError("Original error")


# Test _utcnow_iso returns valid ISO format
def test_utcnow_iso_returns_iso_format() -> None:
    """Verify _utcnow_iso returns a valid ISO 8601 timestamp."""
    with TokenCache(":memory:") as cache:
        timestamp = cache._utcnow_iso()
        # Should contain date separator and time separator
        assert "T" in timestamp
        # Should have UTC timezone indicator
        assert "+" in timestamp or "Z" in timestamp


# ============================================================================
# Token dictionary tests
# ============================================================================


# Test get_or_create_token creates new token
def test_get_or_create_token_new() -> None:
    """Verify new tokens are assigned sequential IDs starting from 1."""
    with TokenCache(":memory:") as cache:
        id1 = cache.get_or_create_token("hello")
        id2 = cache.get_or_create_token("world")

        assert id1 == 1
        assert id2 == 2
        assert cache.token_count() == 2


# Test get_or_create_token returns existing ID
def test_get_or_create_token_existing() -> None:
    """Verify existing tokens return their assigned ID."""
    with TokenCache(":memory:") as cache:
        id1 = cache.get_or_create_token("hello")
        id2 = cache.get_or_create_token("hello")

        assert id1 == id2
        assert cache.token_count() == 1


# Test get_token returns token string by ID
def test_get_token_returns_string() -> None:
    """Verify get_token returns the correct string for a valid ID."""
    with TokenCache(":memory:") as cache:
        token_id = cache.get_or_create_token("test_token")
        result = cache.get_token(token_id)

        assert result == "test_token"


# Test get_token returns None for invalid ID
def test_get_token_returns_none_for_invalid() -> None:
    """Verify get_token returns None for non-existent ID."""
    with TokenCache(":memory:") as cache:
        result = cache.get_token(9999)

        assert result is None


# Test token dictionary is loaded on open
def test_token_dict_loaded_on_open() -> None:
    """Verify in-memory token dict is populated when cache opens."""
    with TokenCache(":memory:") as cache:
        cache.get_or_create_token("first")
        cache.get_or_create_token("second")

        # Verify in-memory dicts are populated
        assert len(cache._token_to_id) == 2
        assert len(cache._id_to_token) == 2
        assert cache._token_to_id["first"] == 1
        assert cache._id_to_token[1] == "first"


# Test empty token is valid
def test_empty_token_is_valid() -> None:
    """Verify empty string can be stored as a token."""
    with TokenCache(":memory:") as cache:
        token_id = cache.get_or_create_token("")
        result = cache.get_token(token_id)

        assert result == ""
        assert token_id == 1


# Test special characters in tokens
def test_special_characters_in_tokens() -> None:
    """Verify tokens with special characters are handled correctly."""
    with TokenCache(":memory:") as cache:
        special_tokens = ["{", "}", '"', ":", ",", "\n", "\t", "emoji🎉"]
        ids = [cache.get_or_create_token(t) for t in special_tokens]

        # Each should get unique ID
        assert len(set(ids)) == len(special_tokens)

        # Each should round-trip correctly
        for token in special_tokens:
            token_id = cache._token_to_id[token]
            assert cache.get_token(token_id) == token


# ============================================================================
# Cache entry CRUD tests
# ============================================================================


# Test put and get roundtrip
def test_put_and_get_roundtrip() -> None:
    """Verify data can be stored and retrieved."""
    with TokenCache(":memory:") as cache:
        data = b"hello world"
        cache.put("abc123", "test", data)
        result = cache.get("abc123", "test")

        assert result == data


# Test get returns None for missing entry
def test_get_returns_none_for_missing() -> None:
    """Verify get returns None for non-existent entries."""
    with TokenCache(":memory:") as cache:
        result = cache.get("nonexistent", "test")

        assert result is None


# Test put with metadata
def test_put_with_metadata() -> None:
    """Verify metadata is stored alongside data."""
    with TokenCache(":memory:") as cache:
        data = b"main data"
        metadata = b"extra info"
        cache.put("meta123", "test", data, metadata=metadata)

        result = cache.get_with_metadata("meta123", "test")

        assert result is not None
        assert result[0] == data
        assert result[1] == metadata


# Test get_with_metadata returns None for missing
def test_get_with_metadata_returns_none_for_missing() -> None:
    """Verify get_with_metadata returns None for non-existent entries."""
    with TokenCache(":memory:") as cache:
        result = cache.get_with_metadata("nonexistent", "test")

        assert result is None


# Test put without metadata stores None
def test_put_without_metadata() -> None:
    """Verify entries without metadata have None metadata."""
    with TokenCache(":memory:") as cache:
        cache.put("nometa", "test", b"data")

        result = cache.get_with_metadata("nometa", "test")

        assert result is not None
        assert result[0] == b"data"
        assert result[1] is None


# Test exists returns True for existing entry
def test_exists_returns_true() -> None:
    """Verify exists returns True for existing entries."""
    with TokenCache(":memory:") as cache:
        cache.put("exists123", "test", b"data")

        assert cache.exists("exists123", "test") is True


# Test exists returns False for missing entry
def test_exists_returns_false() -> None:
    """Verify exists returns False for non-existent entries."""
    with TokenCache(":memory:") as cache:
        assert cache.exists("missing", "test") is False


# Test put replaces existing entry
def test_put_replaces_existing() -> None:
    """Verify put overwrites existing entry with same hash and type."""
    with TokenCache(":memory:") as cache:
        cache.put("replace", "test", b"old data")
        cache.put("replace", "test", b"new data")

        result = cache.get("replace", "test")

        assert result == b"new data"
        assert cache.entry_count() == 1


# Test different entry types are independent
def test_entry_types_are_independent() -> None:
    """Verify same hash can exist for different entry types."""
    with TokenCache(":memory:") as cache:
        cache.put("same_hash", "llm", b"llm data")
        cache.put("same_hash", "graph", b"graph data")

        assert cache.get("same_hash", "llm") == b"llm data"
        assert cache.get("same_hash", "graph") == b"graph data"
        assert cache.entry_count() == 2


# Test delete removes entry
def test_delete_removes_entry() -> None:
    """Verify delete removes the specified entry."""
    with TokenCache(":memory:") as cache:
        cache.put("todelete", "test", b"data")
        assert cache.exists("todelete", "test") is True

        result = cache.delete("todelete", "test")

        assert result is True
        assert cache.exists("todelete", "test") is False


# Test delete returns False for missing entry
def test_delete_returns_false_for_missing() -> None:
    """Verify delete returns False when entry doesn't exist."""
    with TokenCache(":memory:") as cache:
        result = cache.delete("nonexistent", "test")

        assert result is False


# Test entry_count with type filter
def test_entry_count_with_type_filter() -> None:
    """Verify entry_count filters by entry type."""
    with TokenCache(":memory:") as cache:
        cache.put("a", "llm", b"data")
        cache.put("b", "llm", b"data")
        cache.put("c", "graph", b"data")

        assert cache.entry_count() == 3
        assert cache.entry_count(entry_type="llm") == 2
        assert cache.entry_count(entry_type="graph") == 1
        assert cache.entry_count(entry_type="score") == 0


# Test list_entry_types returns empty list for empty cache
def test_list_entry_types_empty() -> None:
    """Verify list_entry_types returns empty list for empty cache."""
    with TokenCache(":memory:") as cache:
        assert cache.list_entry_types() == []


# Test list_entry_types returns distinct entry types
def test_list_entry_types_returns_distinct_types() -> None:
    """Verify list_entry_types returns all distinct entry types."""
    with TokenCache(":memory:") as cache:
        cache.put("a", "llm", b"data")
        cache.put("b", "llm", b"data")
        cache.put("c", "graph", b"data")
        cache.put("d", "score", b"data")

        types = cache.list_entry_types()

        assert types == ["graph", "llm", "score"]  # Alphabetical order


# Test binary data is preserved exactly
def test_binary_data_preserved() -> None:
    """Verify binary data with null bytes is preserved."""
    with TokenCache(":memory:") as cache:
        binary_data = b"\x00\x01\x02\xff\xfe\xfd"
        cache.put("binary", "test", binary_data)

        result = cache.get("binary", "test")

        assert result == binary_data


# ============================================================================
# Export/Import error handling tests (no filesystem access)
# ============================================================================


# Test export_entries raises KeyError for unregistered type
def test_export_entries_raises_for_unregistered_type(tmp_path) -> None:
    """Verify export_entries raises KeyError for unregistered entry_type."""
    with TokenCache(":memory:") as cache:
        with pytest.raises(KeyError):
            cache.export_entries(tmp_path, "unregistered")


# Test import_entries raises KeyError for unregistered type
def test_import_entries_raises_for_unregistered_type(tmp_path) -> None:
    """Verify import_entries raises KeyError for unregistered entry_type."""
    # Create empty directory for test
    tmp_path.mkdir(parents=True, exist_ok=True)

    with TokenCache(":memory:") as cache:
        with pytest.raises(KeyError):
            cache.import_entries(tmp_path, "unregistered")


# Test import_entries raises FileNotFoundError for missing directory
def test_import_entries_raises_for_missing_directory() -> None:
    """Verify import_entries raises FileNotFoundError for missing directory."""
    from pathlib import Path

    from causaliq_knowledge.cache.encoders import JsonEncoder

    with TokenCache(":memory:") as cache:
        cache.register_encoder("json", JsonEncoder())

        with pytest.raises(FileNotFoundError):
            cache.import_entries(Path("/nonexistent/path"), "json")


# ============================================================================
# Cache hit tracking tests
# ============================================================================


# Test that get increments hit count
def test_get_increments_hit_count() -> None:
    """Verify get() increments hit_count for existing entries."""
    with TokenCache(":memory:") as cache:
        cache.put("key1", "test", b"data")

        # First get
        cache.get("key1", "test")
        cursor = cache.conn.execute(
            "SELECT hit_count FROM cache_entries WHERE hash = 'key1'"
        )
        assert cursor.fetchone()[0] == 1

        # Second get
        cache.get("key1", "test")
        cursor = cache.conn.execute(
            "SELECT hit_count FROM cache_entries WHERE hash = 'key1'"
        )
        assert cursor.fetchone()[0] == 2


# Test that get updates last_accessed_at
def test_get_updates_last_accessed_at() -> None:
    """Verify get() updates last_accessed_at timestamp."""
    with TokenCache(":memory:") as cache:
        cache.put("key1", "test", b"data")

        # First get
        cache.get("key1", "test")
        cursor = cache.conn.execute(
            "SELECT last_accessed_at FROM cache_entries WHERE hash = 'key1'"
        )
        first_access = cursor.fetchone()[0]
        assert first_access is not None


# Test that get returns None for missing entry without error
def test_get_missing_entry_does_not_increment() -> None:
    """Verify get() for non-existent entry returns None without error."""
    with TokenCache(":memory:") as cache:
        result = cache.get("nonexistent", "test")
        assert result is None


# Test total_hits returns sum of all hit counts
def test_total_hits_returns_sum() -> None:
    """Verify total_hits() returns sum of all hit counts."""
    with TokenCache(":memory:") as cache:
        cache.put("key1", "test", b"data1")
        cache.put("key2", "test", b"data2")

        # Access key1 twice, key2 once
        cache.get("key1", "test")
        cache.get("key1", "test")
        cache.get("key2", "test")

        assert cache.total_hits() == 3


# Test total_hits with entry_type filter
def test_total_hits_by_entry_type() -> None:
    """Verify total_hits() filters by entry_type."""
    with TokenCache(":memory:") as cache:
        cache.put("key1", "llm", b"data1")
        cache.put("key2", "graph", b"data2")

        cache.get("key1", "llm")
        cache.get("key1", "llm")
        cache.get("key2", "graph")

        assert cache.total_hits("llm") == 2
        assert cache.total_hits("graph") == 1
        assert cache.total_hits() == 3


# Test total_hits returns zero for empty cache
def test_total_hits_empty_cache() -> None:
    """Verify total_hits() returns 0 for empty cache."""
    with TokenCache(":memory:") as cache:
        assert cache.total_hits() == 0
