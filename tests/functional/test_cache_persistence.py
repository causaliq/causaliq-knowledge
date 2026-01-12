"""
Functional tests for TokenCache file-based persistence.

These tests verify that cache data persists correctly across sessions.
Test database files are created in tests/data/functional/tmp/ (gitignored).
"""

from pathlib import Path

import pytest

from causaliq_knowledge.cache import TokenCache

# Path to temporary test output directory (gitignored)
TEST_TMP_DIR = Path(__file__).parent.parent / "data" / "functional" / "tmp"


@pytest.fixture(scope="module", autouse=True)
def setup_test_dir() -> None:
    """Ensure test tmp directory exists."""
    TEST_TMP_DIR.mkdir(parents=True, exist_ok=True)


# Test that file-based cache creates database file
def test_file_cache_creates_db_file() -> None:
    """Verify file-based cache creates the database file."""
    db_path = TEST_TMP_DIR / "create_test.db"
    # Clean up from previous runs
    if db_path.exists():
        db_path.unlink()

    with TokenCache(db_path) as cache:
        assert cache.table_exists("tokens")

    assert db_path.exists()


# Test that file-based cache persists schema across sessions
def test_file_cache_persists_schema() -> None:
    """Verify schema persists when cache is reopened."""
    db_path = TEST_TMP_DIR / "persist_schema.db"
    # Clean up from previous runs
    if db_path.exists():
        db_path.unlink()

    # Create and close
    with TokenCache(db_path) as cache:
        assert cache.table_exists("tokens")
        assert cache.table_exists("cache_entries")

    # Reopen and verify
    with TokenCache(db_path) as cache:
        assert cache.table_exists("tokens")
        assert cache.table_exists("cache_entries")


# Test that tokens persist and reload across sessions
def test_tokens_persist_across_sessions() -> None:
    """Verify tokens are saved to disk and reloaded on reopen."""
    db_path = TEST_TMP_DIR / "persist_tokens.db"
    # Clean up from previous runs
    if db_path.exists():
        db_path.unlink()

    # Create tokens in first session
    with TokenCache(db_path) as cache:
        id1 = cache.get_or_create_token("hello")
        id2 = cache.get_or_create_token("world")
        assert id1 == 1
        assert id2 == 2

    # Reopen and verify tokens were loaded from disk
    with TokenCache(db_path) as cache:
        # Should return same IDs (loaded from disk)
        assert cache.get_or_create_token("hello") == 1
        assert cache.get_or_create_token("world") == 2
        # In-memory dict should be populated
        assert cache._token_to_id["hello"] == 1
        assert cache._id_to_token[1] == "hello"
        # New token should get next ID
        id3 = cache.get_or_create_token("new")
        assert id3 == 3
