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
        expected = {"hash", "entry_type", "data", "created_at", "metadata"}
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
