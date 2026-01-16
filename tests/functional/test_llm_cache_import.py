"""Functional tests for LLM caching with imported test data.

These tests demonstrate loading pre-cached LLM responses from files,
then using cached_completion to serve responses without API calls.
"""

from pathlib import Path
from typing import Any, Dict, List

from causaliq_knowledge.cache import TokenCache
from causaliq_knowledge.llm.base_client import (
    BaseLLMClient,
    LLMConfig,
    LLMResponse,
)
from causaliq_knowledge.llm.cache import LLMCacheEntry, LLMEntryEncoder

# Paths
TEST_DATA_DIR = Path(__file__).parent.parent / "data" / "functional"
LLM_CACHE_DATA = TEST_DATA_DIR / "llm_cache_data"
TEST_TMP_DIR = TEST_DATA_DIR / "tmp"


# --- Mock Client for Testing ---


class MockLLMClient(BaseLLMClient):
    """Mock LLM client that tracks API calls."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._call_count = 0

    @property
    def provider_name(self) -> str:
        return "openai"

    def completion(
        self, messages: List[Dict[str, str]], **kwargs: Any
    ) -> LLMResponse:
        self._call_count += 1
        return LLMResponse(
            content="API response - should not see this if cached",
            model=self.config.model,
            input_tokens=10,
            output_tokens=10,
            cost=0.001,
        )

    @property
    def call_count(self) -> int:
        return self._call_count

    def is_available(self) -> bool:
        return True

    def list_models(self) -> List[str]:
        return ["gpt-4"]


# --- Fixture-Style Import Tests ---


def test_import_llm_cache_data_from_files() -> None:
    """Verify LLM cache data can be imported from test files."""
    with TokenCache(":memory:") as cache:
        cache.register_encoder("llm", LLMEntryEncoder())

        count = cache.import_entries(LLM_CACHE_DATA, "llm")

        assert count == 2
        assert cache.exists("python_question", "llm")
        assert cache.exists("ml_question", "llm")


def test_imported_data_contains_valid_llm_entries() -> None:
    """Verify imported data deserializes to valid LLMCacheEntry."""
    with TokenCache(":memory:") as cache:
        cache.register_encoder("llm", LLMEntryEncoder())
        cache.import_entries(LLM_CACHE_DATA, "llm")

        data = cache.get_data("python_question", "llm")
        entry = LLMCacheEntry.from_dict(data)

        assert entry.model == "gpt-4"
        assert entry.temperature == 0.1
        assert "Python" in entry.response.content
        assert entry.metadata.provider == "openai"
        assert entry.metadata.latency_ms == 850


# --- Cache Hit from Imported Data ---


def test_cached_completion_uses_imported_data() -> None:
    """Verify cached_completion returns imported responses without API calls.

    This test demonstrates the key workflow:
    1. Import pre-cached LLM responses from files
    2. Create a client with the cache
    3. Call cached_completion - should return cached response
    4. Verify no actual API call was made
    """
    # Create cache and import test data
    with TokenCache(":memory:") as cache:
        cache.register_encoder("llm", LLMEntryEncoder())
        cache.import_entries(LLM_CACHE_DATA, "llm")

        # The imported data uses hash "python_question"
        # We need to query with matching parameters to get a cache hit

        # Read the imported entry to get matching query params
        data = cache.get_data("python_question", "llm")
        entry = LLMCacheEntry.from_dict(data)

        # Create client with cache
        config = LLMConfig(
            model=entry.model,
            temperature=entry.temperature,
            max_tokens=entry.max_tokens,
        )
        client = MockLLMClient(config)
        client.set_cache(cache)

        # Build the cache key for the same query
        cache_key = client._build_cache_key(entry.messages)

        # Store with correct key (rename from filename to computed hash)
        cache.put_data(cache_key, "llm", entry.to_dict())

        # Now cached_completion should hit cache
        response = client.cached_completion(entry.messages)

        assert "Python" in response.content
        assert client.call_count == 0  # No API call made


def test_cache_miss_falls_through_to_api() -> None:
    """Verify cache miss makes API call for unknown queries."""
    with TokenCache(":memory:") as cache:
        cache.register_encoder("llm", LLMEntryEncoder())

        config = LLMConfig(model="gpt-4", temperature=0.1, max_tokens=500)
        client = MockLLMClient(config)
        client.set_cache(cache)

        # Query not in cache
        response = client.cached_completion(
            [{"role": "user", "content": "Unknown question not in cache"}]
        )

        assert (
            response.content == "API response - should not see this if cached"
        )
        assert client.call_count == 1


# --- Persistence Round-Trip Tests ---


def test_export_import_round_trip_preserves_llm_data(tmp_path: Path) -> None:
    """Verify LLM cache entries survive export/import round-trip."""
    export_dir = tmp_path / "llm_export"
    export_dir.mkdir()

    entry = LLMCacheEntry.create(
        model="claude-3",
        messages=[{"role": "user", "content": "Test question"}],
        content="Test answer",
        temperature=0.5,
        max_tokens=1000,
        provider="anthropic",
        latency_ms=500,
        input_tokens=5,
        output_tokens=3,
        cost_usd=0.001,
    )

    # Export from first cache
    with TokenCache(":memory:") as cache1:
        cache1.register_encoder("llm", LLMEntryEncoder())
        cache1.put_data("test_hash", "llm", entry.to_dict())
        cache1.export_entries(export_dir, "llm")

    # Import to second cache
    with TokenCache(":memory:") as cache2:
        cache2.register_encoder("llm", LLMEntryEncoder())
        cache2.import_entries(export_dir, "llm")

        data = cache2.get_data("test_hash", "llm")
        restored = LLMCacheEntry.from_dict(data)

        assert restored.model == entry.model
        assert restored.messages == entry.messages
        assert restored.response.content == entry.response.content
        assert restored.metadata.provider == entry.metadata.provider
        assert restored.metadata.tokens.input == entry.metadata.tokens.input
