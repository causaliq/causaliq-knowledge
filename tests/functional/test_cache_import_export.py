"""
Functional tests for TokenCache import/export operations.

These tests verify that cache entries can be exported to files and
imported from files. Test reads use tracked test data files.
Writes use tests/data/functional/tmp/ (gitignored).
"""

import json
from pathlib import Path

from causaliq_knowledge.cache import TokenCache
from causaliq_knowledge.cache.encoders import JsonEncoder

# Paths
TEST_DATA_DIR = Path(__file__).parent.parent / "data" / "functional"
IMPORT_TEST_DATA = TEST_DATA_DIR / "import_test_data"
TEST_TMP_DIR = TEST_DATA_DIR / "tmp"


# ============================================================================
# Export tests
# ============================================================================


# Test export_entries creates output directory if needed
def test_export_creates_directory() -> None:
    """Verify export_entries creates output directory if it doesn't exist."""
    output_dir = TEST_TMP_DIR / "export_test" / "nested"
    # Clean up from previous runs
    if output_dir.exists():
        for f in output_dir.iterdir():
            f.unlink()
        output_dir.rmdir()

    with TokenCache(":memory:") as cache:
        cache.register_encoder("json", JsonEncoder())
        cache.put_data("abc123", "json", {"key": "value"})

        cache.export_entries(output_dir, "json")

        assert output_dir.exists()
        assert output_dir.is_dir()
        assert (output_dir / "abc123.json").exists()


# Test export_entries exports single entry to correct file
def test_export_single_entry() -> None:
    """Verify export_entries creates file named {hash}.{ext}."""
    output_dir = TEST_TMP_DIR / "export_single"
    output_dir.mkdir(parents=True, exist_ok=True)
    # Clean up
    for f in output_dir.iterdir():
        f.unlink()

    with TokenCache(":memory:") as cache:
        cache.register_encoder("json", JsonEncoder())
        cache.put_data("myhash", "json", {"hello": "world"})

        count = cache.export_entries(output_dir, "json")

        assert count == 1
        expected_file = output_dir / "myhash.json"
        assert expected_file.exists()


# Test export_entries returns correct count
def test_export_returns_count() -> None:
    """Verify export_entries returns number of exported entries."""
    output_dir = TEST_TMP_DIR / "export_count"
    output_dir.mkdir(parents=True, exist_ok=True)
    # Clean up
    for f in output_dir.iterdir():
        f.unlink()

    with TokenCache(":memory:") as cache:
        cache.register_encoder("json", JsonEncoder())
        cache.put_data("hash1", "json", {"a": 1})
        cache.put_data("hash2", "json", {"b": 2})
        cache.put_data("hash3", "json", {"c": 3})

        count = cache.export_entries(output_dir, "json")

        assert count == 3


# Test export_entries returns zero for empty cache
def test_export_empty_cache() -> None:
    """Verify export_entries returns 0 when no entries exist."""
    output_dir = TEST_TMP_DIR / "export_empty"
    output_dir.mkdir(parents=True, exist_ok=True)

    with TokenCache(":memory:") as cache:
        cache.register_encoder("json", JsonEncoder())

        count = cache.export_entries(output_dir, "json")

        assert count == 0


# Test export_entries only exports specified entry_type
def test_export_filters_by_type() -> None:
    """Verify export_entries only exports entries of requested type."""
    output_dir = TEST_TMP_DIR / "export_filter"
    output_dir.mkdir(parents=True, exist_ok=True)
    # Clean up
    for f in output_dir.iterdir():
        f.unlink()

    with TokenCache(":memory:") as cache:
        cache.register_encoder("json", JsonEncoder())
        cache.register_encoder("other", JsonEncoder())
        cache.put_data("hash1", "json", {"type": "json"})
        cache.put_data("hash2", "other", {"type": "other"})

        count = cache.export_entries(output_dir, "json")

        assert count == 1
        assert (output_dir / "hash1.json").exists()
        assert not (output_dir / "hash2.json").exists()


# Test export_entries uses encoder's default format
def test_export_uses_default_format() -> None:
    """Verify export_entries uses encoder.default_export_format."""
    output_dir = TEST_TMP_DIR / "export_default_fmt"
    output_dir.mkdir(parents=True, exist_ok=True)
    # Clean up
    for f in output_dir.iterdir():
        f.unlink()

    with TokenCache(":memory:") as cache:
        encoder = JsonEncoder()
        assert encoder.default_export_format == "json"
        cache.register_encoder("mytype", encoder)
        cache.put_data("hashval", "mytype", {"data": 123})

        cache.export_entries(output_dir, "mytype")

        assert (output_dir / "hashval.json").exists()


# Test export_entries accepts custom format
def test_export_custom_format() -> None:
    """Verify export_entries accepts fmt parameter for extension."""
    output_dir = TEST_TMP_DIR / "export_custom_fmt"
    output_dir.mkdir(parents=True, exist_ok=True)
    # Clean up
    for f in output_dir.iterdir():
        f.unlink()

    with TokenCache(":memory:") as cache:
        cache.register_encoder("json", JsonEncoder())
        cache.put_data("myhash", "json", {"key": "value"})

        cache.export_entries(output_dir, "json", fmt="txt")

        assert (output_dir / "myhash.txt").exists()
        assert not (output_dir / "myhash.json").exists()


# Test export_entries file content is valid JSON
def test_export_content_is_valid_json() -> None:
    """Verify exported file contains valid JSON matching original data."""
    output_dir = TEST_TMP_DIR / "export_content"
    output_dir.mkdir(parents=True, exist_ok=True)
    # Clean up
    for f in output_dir.iterdir():
        f.unlink()

    original_data = {"name": "test", "count": 42, "nested": {"a": [1, 2, 3]}}

    with TokenCache(":memory:") as cache:
        cache.register_encoder("json", JsonEncoder())
        cache.put_data("datahash", "json", original_data)

        cache.export_entries(output_dir, "json")

        exported_file = output_dir / "datahash.json"
        with open(exported_file, encoding="utf-8") as f:
            loaded = json.load(f)

        assert loaded == original_data


# Test export_entries handles multiple entries with same type
def test_export_multiple_entries() -> None:
    """Verify export_entries exports all entries of a type."""
    output_dir = TEST_TMP_DIR / "export_multi"
    output_dir.mkdir(parents=True, exist_ok=True)
    # Clean up
    for f in output_dir.iterdir():
        f.unlink()

    with TokenCache(":memory:") as cache:
        cache.register_encoder("json", JsonEncoder())
        cache.put_data("first", "json", {"order": 1})
        cache.put_data("second", "json", {"order": 2})
        cache.put_data("third", "json", {"order": 3})

        count = cache.export_entries(output_dir, "json")

        assert count == 3
        for name, expected in [
            ("first", {"order": 1}),
            ("second", {"order": 2}),
            ("third", {"order": 3}),
        ]:
            path = output_dir / f"{name}.json"
            assert path.exists()
            with open(path, encoding="utf-8") as f:
                assert json.load(f) == expected


# ============================================================================
# Import tests (read from tracked test data)
# ============================================================================


# Test import_entries reads from tracked test data
def test_import_from_test_data() -> None:
    """Verify import_entries reads files from test data directory."""
    with TokenCache(":memory:") as cache:
        cache.register_encoder("json", JsonEncoder())

        count = cache.import_entries(IMPORT_TEST_DATA, "json")

        assert count == 3


# Test import_entries uses filename stem as hash
def test_import_uses_filename_as_hash() -> None:
    """Verify import_entries uses filename (without extension) as hash."""
    with TokenCache(":memory:") as cache:
        cache.register_encoder("json", JsonEncoder())

        cache.import_entries(IMPORT_TEST_DATA, "json")

        # Files are entry1.json, entry2.json, entry3.json
        assert cache.exists("entry1", "json")
        assert cache.exists("entry2", "json")
        assert cache.exists("entry3", "json")


# Test import_entries correctly parses JSON content
def test_import_parses_json_content() -> None:
    """Verify imported data matches original file content."""
    with TokenCache(":memory:") as cache:
        cache.register_encoder("json", JsonEncoder())

        cache.import_entries(IMPORT_TEST_DATA, "json")

        # Verify entry1
        data1 = cache.get_data("entry1", "json")
        assert data1 == {"key": "value", "count": 42}

        # Verify entry2 with nested object
        data2 = cache.get_data("entry2", "json")
        assert data2 == {"name": "test entry", "nested": {"a": 1, "b": 2}}

        # Verify entry3 with list and mixed types
        data3 = cache.get_data("entry3", "json")
        assert data3["items"] == [1, 2, 3, 4, 5]
        assert data3["active"] is True
        assert data3["score"] == 3.14


# Test import_entries returns zero for empty directory
def test_import_empty_directory() -> None:
    """Verify import_entries returns 0 for directory with no files."""
    empty_dir = TEST_TMP_DIR / "import_empty"
    empty_dir.mkdir(parents=True, exist_ok=True)
    # Clean up any files
    for f in empty_dir.iterdir():
        if f.is_file():
            f.unlink()

    with TokenCache(":memory:") as cache:
        cache.register_encoder("json", JsonEncoder())

        count = cache.import_entries(empty_dir, "json")

        assert count == 0


# Test import_entries skips subdirectories
def test_import_skips_subdirectories() -> None:
    """Verify import_entries only imports files, not subdirectories."""
    test_dir = TEST_TMP_DIR / "import_subdir"
    test_dir.mkdir(parents=True, exist_ok=True)
    # Create a subdirectory
    (test_dir / "subdir").mkdir(exist_ok=True)
    # Create a file
    (test_dir / "valid.json").write_text('{"key": "value"}')

    with TokenCache(":memory:") as cache:
        cache.register_encoder("json", JsonEncoder())

        count = cache.import_entries(test_dir, "json")

        # Should only count the file, not the subdirectory
        assert count == 1
        assert cache.exists("valid", "json")


# ============================================================================
# Round-trip tests
# ============================================================================


# Test export then import round-trip preserves data
def test_export_import_round_trip() -> None:
    """Verify data survives export then import round-trip."""
    export_dir = TEST_TMP_DIR / "round_trip"
    export_dir.mkdir(parents=True, exist_ok=True)
    # Clean up
    for f in export_dir.iterdir():
        if f.is_file():
            f.unlink()

    original_data = {
        "complex": {
            "nested": {"deep": {"value": 42}},
            "list": [1, 2, 3],
        },
        "string": "hello world",
        "number": 3.14159,
        "bool": True,
        "null_val": None,
    }

    # Export from first cache
    with TokenCache(":memory:") as cache1:
        cache1.register_encoder("json", JsonEncoder())
        cache1.put_data("testkey", "json", original_data)

        cache1.export_entries(export_dir, "json")

    # Import into second cache
    with TokenCache(":memory:") as cache2:
        cache2.register_encoder("json", JsonEncoder())
        cache2.import_entries(export_dir, "json")

        restored_data = cache2.get_data("testkey", "json")

        assert restored_data == original_data


# Test round-trip with multiple entries
def test_round_trip_multiple_entries() -> None:
    """Verify multiple entries survive round-trip."""
    export_dir = TEST_TMP_DIR / "round_trip_multi"
    export_dir.mkdir(parents=True, exist_ok=True)
    # Clean up
    for f in export_dir.iterdir():
        if f.is_file():
            f.unlink()

    entries = {
        "alpha": {"name": "first", "value": 1},
        "beta": {"name": "second", "value": 2},
        "gamma": {"name": "third", "value": 3},
    }

    # Export from first cache
    with TokenCache(":memory:") as cache1:
        cache1.register_encoder("json", JsonEncoder())
        for key, data in entries.items():
            cache1.put_data(key, "json", data)

        count = cache1.export_entries(export_dir, "json")
        assert count == 3

    # Import into second cache
    with TokenCache(":memory:") as cache2:
        cache2.register_encoder("json", JsonEncoder())
        count = cache2.import_entries(export_dir, "json")
        assert count == 3

        for key, expected in entries.items():
            assert cache2.get_data(key, "json") == expected


# Test round-trip with Unicode and special characters
def test_round_trip_unicode() -> None:
    """Verify Unicode content survives round-trip."""
    export_dir = TEST_TMP_DIR / "round_trip_unicode"
    export_dir.mkdir(parents=True, exist_ok=True)
    # Clean up
    for f in export_dir.iterdir():
        if f.is_file():
            f.unlink()

    original_data = {
        "chinese": "你好世界",
        "japanese": "こんにちは",
        "russian": "Привет мир",
        "arabic": "مرحبا بالعالم",
        "emoji": "🌍🎉🚀",
        "special": "© ® ™ € £ ¥ • → ←",
    }

    with TokenCache(":memory:") as cache1:
        cache1.register_encoder("json", JsonEncoder())
        cache1.put_data("unicode_test", "json", original_data)
        cache1.export_entries(export_dir, "json")

    with TokenCache(":memory:") as cache2:
        cache2.register_encoder("json", JsonEncoder())
        cache2.import_entries(export_dir, "json")
        restored = cache2.get_data("unicode_test", "json")
        assert restored == original_data


# Test round-trip with LLMCacheEntry data
def test_round_trip_llm_cache_entry() -> None:
    """Verify LLMCacheEntry survives round-trip via export/import."""
    from causaliq_knowledge.llm.cache import LLMCacheEntry, LLMEntryEncoder

    export_dir = TEST_TMP_DIR / "round_trip_llm"
    export_dir.mkdir(parents=True, exist_ok=True)
    # Clean up
    for f in export_dir.iterdir():
        if f.is_file():
            f.unlink()

    entry = LLMCacheEntry.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is Python?"},
        ],
        content="Python is a programming language.",
        temperature=0.7,
        max_tokens=1000,
        provider="openai",
        latency_ms=850,
        input_tokens=25,
        output_tokens=10,
        cost_usd=0.002,
    )

    # Export using LLMEntryEncoder
    with TokenCache(":memory:") as cache1:
        cache1.register_encoder("llm", LLMEntryEncoder())
        cache1.put_data("llm_request_hash", "llm", entry.to_dict())
        cache1.export_entries(export_dir, "llm")

    # Import into fresh cache
    with TokenCache(":memory:") as cache2:
        cache2.register_encoder("llm", LLMEntryEncoder())
        cache2.import_entries(export_dir, "llm")

        restored_dict = cache2.get_data("llm_request_hash", "llm")
        restored = LLMCacheEntry.from_dict(restored_dict)

        assert restored.model == entry.model
        assert restored.messages == entry.messages
        assert restored.temperature == entry.temperature
        assert restored.max_tokens == entry.max_tokens
        assert restored.response.content == entry.response.content
        assert restored.metadata.provider == entry.metadata.provider
        assert restored.metadata.latency_ms == entry.metadata.latency_ms
        assert restored.metadata.tokens.input == entry.metadata.tokens.input
