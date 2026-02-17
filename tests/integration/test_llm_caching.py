"""Integration tests for LLM caching.

These tests verify the caching integration with mock LLM clients,
testing cache hit/miss scenarios without making real API calls.
"""

from typing import Any, Dict, List

from causaliq_core.cache import TokenCache

from causaliq_knowledge.llm.base_client import (
    BaseLLMClient,
    LLMConfig,
    LLMResponse,
)
from causaliq_knowledge.llm.cache import LLMCacheEntry

# --- Mock Client for Testing ---


class MockLLMClient(BaseLLMClient):
    """Mock LLM client for testing cache integration."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._call_count = 0
        self._responses: Dict[str, str] = {}
        self._default_response = "Default mock response"

    @property
    def provider_name(self) -> str:
        return "mock"

    def completion(
        self, messages: List[Dict[str, str]], **kwargs: Any
    ) -> LLMResponse:
        self._call_count += 1
        # Use last user message as key for custom responses
        user_msg = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"),
            "",
        )
        content = self._responses.get(user_msg, self._default_response)
        return LLMResponse(
            content=content,
            model=self.config.model,
            input_tokens=len(user_msg.split()),
            output_tokens=len(content.split()),
            cost=0.001,
        )

    @property
    def call_count(self) -> int:
        return self._call_count

    def is_available(self) -> bool:
        return True

    def list_models(self) -> List[str]:
        return ["mock-model"]

    def set_response(self, user_message: str, response: str) -> None:
        """Set a custom response for a specific user message."""
        self._responses[user_message] = response


# --- Cache Hit/Miss Tests ---


# Test cache miss - first call makes API request
def test_cache_miss_makes_api_call():
    """First call should make API request and store in cache."""
    config = LLMConfig(model="mock-model")
    client = MockLLMClient(config)

    with TokenCache(":memory:") as cache:
        client.set_cache(cache)

        response = client.cached_completion(
            [{"role": "user", "content": "What is Python?"}]
        )

        assert response.content == "Default mock response"
        assert client.call_count == 1
        assert cache.entry_count() == 1


# Test cache hit - second call returns cached response
def test_cache_hit_returns_cached():
    """Second identical call should return cached response."""
    config = LLMConfig(model="mock-model")
    client = MockLLMClient(config)
    client.set_response("What is Python?", "Python is a language")

    with TokenCache(":memory:") as cache:
        client.set_cache(cache)

        # First call - cache miss
        response1 = client.cached_completion(
            [{"role": "user", "content": "What is Python?"}]
        )
        assert response1.content == "Python is a language"
        assert client.call_count == 1

        # Change the response (simulating API change)
        client.set_response("What is Python?", "Python is different now")

        # Second call - cache hit
        response2 = client.cached_completion(
            [{"role": "user", "content": "What is Python?"}]
        )
        assert response2.content == "Python is a language"  # Original cached
        assert client.call_count == 1  # No new API call


# Test different messages have different cache entries
def test_different_messages_different_cache():
    """Different messages should have separate cache entries."""
    config = LLMConfig(model="mock-model")
    client = MockLLMClient(config)
    client.set_response("Hello", "Hi there!")
    client.set_response("Goodbye", "See you!")

    with TokenCache(":memory:") as cache:
        client.set_cache(cache)

        r1 = client.cached_completion([{"role": "user", "content": "Hello"}])
        r2 = client.cached_completion([{"role": "user", "content": "Goodbye"}])

        assert r1.content == "Hi there!"
        assert r2.content == "See you!"
        assert client.call_count == 2
        assert cache.entry_count() == 2


# --- Cache Key Sensitivity Tests ---


# Test different models produce different cache entries
def test_different_models_different_cache():
    """Same message with different models should cache separately."""
    client1 = MockLLMClient(LLMConfig(model="model-a"))
    client2 = MockLLMClient(LLMConfig(model="model-b"))

    with TokenCache(":memory:") as cache:
        client1.set_cache(cache)
        client2.set_cache(cache)

        client1.cached_completion([{"role": "user", "content": "Hello"}])
        client2.cached_completion([{"role": "user", "content": "Hello"}])

        assert cache.entry_count() == 2


# Test different temperatures produce different cache entries
def test_different_temperatures_different_cache():
    """Same message with different temperatures should cache separately."""
    config = LLMConfig(model="mock-model", temperature=0.0)
    client = MockLLMClient(config)

    with TokenCache(":memory:") as cache:
        client.set_cache(cache)

        client.cached_completion(
            [{"role": "user", "content": "Hello"}],
            temperature=0.0,
        )
        client.cached_completion(
            [{"role": "user", "content": "Hello"}],
            temperature=0.7,
        )

        assert cache.entry_count() == 2


# --- Metadata Capture Tests ---


# Test latency is captured in cached entry
def test_latency_captured():
    """Latency should be captured when caching response."""
    config = LLMConfig(model="mock-model")
    client = MockLLMClient(config)

    with TokenCache(":memory:") as cache:
        client.set_cache(cache)

        client.cached_completion([{"role": "user", "content": "Hello"}])

        cache_key = client._build_cache_key(
            [{"role": "user", "content": "Hello"}]
        )
        cached_data = cache.get_data(cache_key)
        entry = LLMCacheEntry.from_dict(cached_data)

        assert entry.metadata.latency_ms >= 0


# Test token counts are captured
def test_token_counts_captured():
    """Token counts should be captured in cached entry."""
    config = LLMConfig(model="mock-model")
    client = MockLLMClient(config)
    client.set_response("Count tokens", "One two three four five")

    with TokenCache(":memory:") as cache:
        client.set_cache(cache)

        client.cached_completion([{"role": "user", "content": "Count tokens"}])

        cache_key = client._build_cache_key(
            [{"role": "user", "content": "Count tokens"}]
        )
        cached_data = cache.get_data(cache_key)
        entry = LLMCacheEntry.from_dict(cached_data)

        assert entry.metadata.tokens.input == 2  # "Count tokens"
        assert entry.metadata.tokens.output == 5  # "One two three four five"


# Test provider is captured
def test_provider_captured():
    """Provider name should be captured in cached entry."""
    config = LLMConfig(model="mock-model")
    client = MockLLMClient(config)

    with TokenCache(":memory:") as cache:
        client.set_cache(cache)

        client.cached_completion([{"role": "user", "content": "Hello"}])

        cache_key = client._build_cache_key(
            [{"role": "user", "content": "Hello"}]
        )
        cached_data = cache.get_data(cache_key)
        entry = LLMCacheEntry.from_dict(cached_data)

        assert entry.metadata.provider == "mock"


# --- Cache Disable Tests ---


# Test use_cache=False bypasses cache read
def test_use_cache_false_bypasses_read():
    """With use_cache=False, cache should not be read."""
    config = LLMConfig(model="mock-model")
    client = MockLLMClient(config)

    with TokenCache(":memory:") as cache:
        client.set_cache(cache, use_cache=True)

        # First call - populates cache
        client.set_response("Hello", "First response")
        client.cached_completion([{"role": "user", "content": "Hello"}])

        # Disable cache
        client.set_cache(cache, use_cache=False)

        # Change response
        client.set_response("Hello", "Second response")

        # Second call - should bypass cache
        response = client.cached_completion(
            [{"role": "user", "content": "Hello"}]
        )

        assert response.content == "Second response"
        assert client.call_count == 2


# Test use_cache=False bypasses cache write
def test_use_cache_false_bypasses_write():
    """With use_cache=False, responses should not be cached."""
    config = LLMConfig(model="mock-model")
    client = MockLLMClient(config)

    with TokenCache(":memory:") as cache:
        client.set_cache(cache, use_cache=False)

        client.cached_completion([{"role": "user", "content": "Hello"}])

        assert cache.entry_count() == 0


# --- Persistence Tests ---


# Test cache persists across sessions (file-based)
def test_cache_persists_across_sessions(tmp_path):
    """Cache entries should persist in file-based cache."""
    cache_path = tmp_path / "test_cache.db"

    # Session 1: Populate cache
    config = LLMConfig(model="mock-model")
    client1 = MockLLMClient(config)
    client1.set_response("Persist test", "Persisted response")

    with TokenCache(str(cache_path)) as cache:
        client1.set_cache(cache)
        client1.cached_completion(
            [{"role": "user", "content": "Persist test"}]
        )
        assert cache.entry_count() == 1

    # Session 2: Read from cache
    client2 = MockLLMClient(config)

    with TokenCache(str(cache_path)) as cache:
        client2.set_cache(cache)
        response = client2.cached_completion(
            [{"role": "user", "content": "Persist test"}]
        )

        assert response.content == "Persisted response"
        assert client2.call_count == 0  # No API call needed


# --- Multi-Turn Conversation Tests ---


# Test multi-turn conversation caching
def test_multi_turn_conversation_caching():
    """Multi-turn conversations should be cached correctly."""
    config = LLMConfig(model="mock-model")
    client = MockLLMClient(config)

    with TokenCache(":memory:") as cache:
        client.set_cache(cache)

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]

        # First call
        response1 = client.cached_completion(messages)
        assert response1.content == "Default mock response"
        assert client.call_count == 1

        # Second call with same messages
        response2 = client.cached_completion(messages)
        assert response2.content == response1.content
        assert client.call_count == 1  # Cache hit

        # Different conversation order - should be different cache entry
        different_messages = [
            {"role": "user", "content": "How are you?"},
            {"role": "system", "content": "You are helpful."},
        ]
        client.cached_completion(different_messages)
        assert client.call_count == 2
