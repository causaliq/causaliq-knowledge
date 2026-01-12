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
