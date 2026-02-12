"""Unit tests for BaseLLMClient abstract interface."""

from typing import Any, Dict, List

import pytest

from causaliq_knowledge.llm.base_client import (
    BaseLLMClient,
    LLMConfig,
    LLMResponse,
)

# --- LLMConfig Tests ---


# Test that LLMConfig has sensible default values
def test_llm_config_default_values():
    config = LLMConfig(model="test-model")
    assert config.model == "test-model"
    assert config.temperature == 0.1
    assert config.max_tokens == 500
    assert config.timeout == 30.0
    assert config.api_key is None


# Test that LLMConfig accepts custom values
def test_llm_config_custom_values():
    config = LLMConfig(
        model="custom-model",
        temperature=0.5,
        max_tokens=1000,
        timeout=60.0,
        api_key="test-key",
    )
    assert config.model == "custom-model"
    assert config.temperature == 0.5
    assert config.max_tokens == 1000
    assert config.timeout == 60.0
    assert config.api_key == "test-key"


# --- LLMResponse Tests ---


# Test basic LLMResponse creation with minimal fields
def test_llm_response_basic():
    response = LLMResponse(
        content="Hello, world!",
        model="test-model",
    )
    assert response.content == "Hello, world!"
    assert response.model == "test-model"
    assert response.input_tokens == 0
    assert response.output_tokens == 0
    assert response.cost == 0.0
    assert response.raw_response is None


# Test LLMResponse with all fields populated
def test_llm_response_full():
    raw = {"id": "test-id", "choices": []}
    response = LLMResponse(
        content="Test content",
        model="test-model",
        input_tokens=10,
        output_tokens=20,
        cost=0.001,
        raw_response=raw,
    )
    assert response.input_tokens == 10
    assert response.output_tokens == 20
    assert response.cost == 0.001
    assert response.raw_response == raw


# Test parsing valid JSON content
def test_llm_response_parse_json_valid():
    response = LLMResponse(
        content='{"key": "value", "number": 42}',
        model="test-model",
    )
    result = response.parse_json()
    assert result == {"key": "value", "number": 42}


# Test parsing JSON wrapped in markdown json code block
def test_llm_response_parse_json_with_markdown_code_block():
    response = LLMResponse(
        content='```json\n{"key": "value"}\n```',
        model="test-model",
    )
    result = response.parse_json()
    assert result == {"key": "value"}


# Test parsing JSON wrapped in generic code block
def test_llm_response_parse_json_with_generic_code_block():
    response = LLMResponse(
        content='```\n{"key": "value"}\n```',
        model="test-model",
    )
    result = response.parse_json()
    assert result == {"key": "value"}


# Test that invalid JSON returns None
def test_llm_response_parse_json_invalid():
    response = LLMResponse(
        content="This is not JSON",
        model="test-model",
    )
    result = response.parse_json()
    assert result is None


# Test that empty content returns None when parsing
def test_llm_response_parse_json_empty():
    response = LLMResponse(
        content="",
        model="test-model",
    )
    result = response.parse_json()
    assert result is None


# --- BaseLLMClient Tests ---


# Helper: Create a concrete mock implementation of BaseLLMClient
def _create_mock_client_class():
    class MockClient(BaseLLMClient):
        def __init__(self, config: LLMConfig) -> None:
            self.config = config
            self._calls = 0
            self._response_content = "Mock response"
            self._available = True
            self._models = ["model-a", "model-b"]

        @property
        def provider_name(self) -> str:
            return "mock"

        def completion(
            self, messages: List[Dict[str, str]], **kwargs: Any
        ) -> LLMResponse:
            self._calls += 1
            return LLMResponse(
                content=self._response_content,
                model=self.config.model,
            )

        @property
        def call_count(self) -> int:
            return self._calls

        def is_available(self) -> bool:
            return self._available

        def list_models(self) -> List[str]:
            return self._models

    return MockClient


# Test that BaseLLMClient cannot be instantiated directly
def test_base_llm_client_cannot_instantiate():
    with pytest.raises(TypeError, match="abstract"):
        BaseLLMClient(LLMConfig(model="test"))  # type: ignore


# Test that a concrete implementation works correctly
def test_base_llm_client_concrete_implementation():
    MockClient = _create_mock_client_class()
    config = LLMConfig(model="mock-model")
    client = MockClient(config)

    assert client.provider_name == "mock"
    assert client.call_count == 0
    assert client.model_name == "mock-model"

    response = client.completion([{"role": "user", "content": "Hello"}])
    assert response.content == "Mock response"
    assert client.call_count == 1


# Test that complete_json calls completion and parses the response
def test_base_llm_client_complete_json():
    MockClient = _create_mock_client_class()
    config = LLMConfig(model="mock-model")
    client = MockClient(config)
    client._response_content = '{"exists": true, "confidence": 0.9}'

    parsed, response = client.complete_json(
        [{"role": "user", "content": "Test"}]
    )

    assert parsed == {"exists": True, "confidence": 0.9}
    assert response.content == '{"exists": true, "confidence": 0.9}'
    assert client.call_count == 1


# Test that is_available method works on mock implementation
def test_base_llm_client_is_available():
    MockClient = _create_mock_client_class()
    config = LLMConfig(model="mock-model")
    client = MockClient(config)

    assert client.is_available() is True
    client._available = False
    assert client.is_available() is False


# Test that list_models method works on mock implementation
def test_base_llm_client_list_models():
    MockClient = _create_mock_client_class()
    config = LLMConfig(model="mock-model")
    client = MockClient(config)

    models = client.list_models()
    assert models == ["model-a", "model-b"]

    client._models = ["new-model"]
    assert client.list_models() == ["new-model"]


# --- Inheritance Verification Tests ---


# Test that GroqClient properly inherits from BaseLLMClient
def test_groq_client_inherits_from_base():
    from causaliq_knowledge.llm.groq_client import GroqClient

    assert issubclass(GroqClient, BaseLLMClient)


# Test that GeminiClient properly inherits from BaseLLMClient
def test_gemini_client_inherits_from_base():
    from causaliq_knowledge.llm.gemini_client import GeminiClient

    assert issubclass(GeminiClient, BaseLLMClient)


# Test that GroqConfig properly inherits from LLMConfig
def test_groq_config_inherits_from_base():
    from causaliq_knowledge.llm.groq_client import GroqConfig

    assert issubclass(GroqConfig, LLMConfig)


# Test that GeminiConfig properly inherits from LLMConfig
def test_gemini_config_inherits_from_base():
    from causaliq_knowledge.llm.gemini_client import GeminiConfig

    assert issubclass(GeminiConfig, LLMConfig)


# --- Cache Key Building Tests ---


# Test that _build_cache_key returns a 16-character hex string
def test_build_cache_key_returns_hex_string():
    MockClient = _create_mock_client_class()
    config = LLMConfig(model="test-model")
    client = MockClient(config)

    key = client._build_cache_key([{"role": "user", "content": "Hello"}])

    assert len(key) == 16
    assert all(c in "0123456789abcdef" for c in key)


# Test that _build_cache_key is deterministic (same inputs = same key)
def test_build_cache_key_is_deterministic():
    MockClient = _create_mock_client_class()
    config = LLMConfig(model="test-model")
    client = MockClient(config)

    messages = [{"role": "user", "content": "Hello"}]
    key1 = client._build_cache_key(messages)
    key2 = client._build_cache_key(messages)

    assert key1 == key2


# Test that different messages produce different keys
def test_build_cache_key_different_messages():
    MockClient = _create_mock_client_class()
    config = LLMConfig(model="test-model")
    client = MockClient(config)

    key1 = client._build_cache_key([{"role": "user", "content": "Hello"}])
    key2 = client._build_cache_key([{"role": "user", "content": "Goodbye"}])

    assert key1 != key2


# Test that different models produce different keys
def test_build_cache_key_different_models():
    MockClient = _create_mock_client_class()
    client1 = MockClient(LLMConfig(model="model-a"))
    client2 = MockClient(LLMConfig(model="model-b"))

    messages = [{"role": "user", "content": "Hello"}]
    key1 = client1._build_cache_key(messages)
    key2 = client2._build_cache_key(messages)

    assert key1 != key2


# Test that different temperatures produce different keys
def test_build_cache_key_different_temperatures():
    MockClient = _create_mock_client_class()
    config = LLMConfig(model="test-model")
    client = MockClient(config)

    messages = [{"role": "user", "content": "Hello"}]
    key1 = client._build_cache_key(messages, temperature=0.0)
    key2 = client._build_cache_key(messages, temperature=0.5)

    assert key1 != key2


# Test that different max_tokens produce different keys
def test_build_cache_key_different_max_tokens():
    MockClient = _create_mock_client_class()
    config = LLMConfig(model="test-model")
    client = MockClient(config)

    messages = [{"role": "user", "content": "Hello"}]
    key1 = client._build_cache_key(messages, max_tokens=100)
    key2 = client._build_cache_key(messages, max_tokens=500)

    assert key1 != key2


# Test that _build_cache_key uses config defaults when kwargs not provided
def test_build_cache_key_uses_config_defaults():
    MockClient = _create_mock_client_class()
    config = LLMConfig(model="test-model", temperature=0.5, max_tokens=1000)
    client = MockClient(config)

    messages = [{"role": "user", "content": "Hello"}]
    key_default = client._build_cache_key(messages)
    key_explicit = client._build_cache_key(
        messages, temperature=0.5, max_tokens=1000
    )

    assert key_default == key_explicit


# Test that message order affects the cache key
def test_build_cache_key_message_order_matters():
    MockClient = _create_mock_client_class()
    config = LLMConfig(model="test-model")
    client = MockClient(config)

    key1 = client._build_cache_key(
        [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
    )
    key2 = client._build_cache_key(
        [
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Hello"},
        ]
    )

    assert key1 != key2


# Test that _build_cache_key handles multi-turn conversations
def test_build_cache_key_multi_turn():
    MockClient = _create_mock_client_class()
    config = LLMConfig(model="test-model")
    client = MockClient(config)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a programming language."},
        {"role": "user", "content": "Tell me more."},
    ]
    key = client._build_cache_key(messages)

    assert len(key) == 16
    assert all(c in "0123456789abcdef" for c in key)


# Test that _build_cache_key handles empty messages list
def test_build_cache_key_empty_messages():
    MockClient = _create_mock_client_class()
    config = LLMConfig(model="test-model")
    client = MockClient(config)

    key = client._build_cache_key([])

    assert len(key) == 16
    assert all(c in "0123456789abcdef" for c in key)


# --- Cache Integration Tests ---


# Test that set_cache configures cache on client
def test_set_cache_configures_client():
    from causaliq_core.cache import TokenCache

    MockClient = _create_mock_client_class()
    client = MockClient(LLMConfig(model="test-model"))

    with TokenCache(":memory:") as cache:
        client.set_cache(cache, use_cache=True)

        assert client.cache is cache
        assert client.use_cache is True


# Test that set_cache with None disables caching
def test_set_cache_with_none():
    MockClient = _create_mock_client_class()
    client = MockClient(LLMConfig(model="test-model"))

    client.set_cache(None, use_cache=False)

    assert client.cache is None
    assert client.use_cache is False


# Test that cache/use_cache defaults work without set_cache
def test_cache_defaults_without_set_cache():
    MockClient = _create_mock_client_class()
    client = MockClient(LLMConfig(model="test-model"))

    assert client.cache is None
    assert client.use_cache is True


# Test that cached_completion calls completion without cache
def test_cached_completion_without_cache():
    MockClient = _create_mock_client_class()
    client = MockClient(LLMConfig(model="test-model"))
    client._response_content = "API response"

    response = client.cached_completion([{"role": "user", "content": "Hello"}])

    assert response.content == "API response"
    assert client.call_count == 1


# Test that cached_completion stores result in cache
def test_cached_completion_stores_in_cache():
    from causaliq_core.cache import TokenCache

    MockClient = _create_mock_client_class()
    client = MockClient(LLMConfig(model="test-model"))
    client._response_content = "API response"

    with TokenCache(":memory:") as cache:
        client.set_cache(cache)

        client.cached_completion([{"role": "user", "content": "Hello"}])

        # Verify entry was cached
        assert cache.entry_count() == 1


# Test that cached_completion returns cached result on second call
def test_cached_completion_cache_hit():
    from causaliq_core.cache import TokenCache

    MockClient = _create_mock_client_class()
    client = MockClient(LLMConfig(model="test-model"))
    client._response_content = "First response"

    with TokenCache(":memory:") as cache:
        client.set_cache(cache)

        # First call - should hit API
        response1 = client.cached_completion(
            [{"role": "user", "content": "Hello"}]
        )
        assert response1.content == "First response"
        assert client.call_count == 1

        # Change the response the mock would return
        client._response_content = "Second response"

        # Second call - should hit cache
        response2 = client.cached_completion(
            [{"role": "user", "content": "Hello"}]
        )
        assert response2.content == "First response"  # Cached value
        assert client.call_count == 1  # No additional API call


# Test that cached_completion respects use_cache=False
def test_cached_completion_respects_use_cache_false():
    from causaliq_core.cache import TokenCache

    MockClient = _create_mock_client_class()
    client = MockClient(LLMConfig(model="test-model"))
    client._response_content = "API response"

    with TokenCache(":memory:") as cache:
        client.set_cache(cache, use_cache=False)

        client.cached_completion([{"role": "user", "content": "Hello"}])

        # Should not cache when use_cache=False
        assert cache.entry_count() == 0


# Test that cached_completion uses correct cache key
def test_cached_completion_different_messages_different_cache():
    from causaliq_core.cache import TokenCache

    MockClient = _create_mock_client_class()
    client = MockClient(LLMConfig(model="test-model"))

    with TokenCache(":memory:") as cache:
        client.set_cache(cache)

        client._response_content = "Response A"
        client.cached_completion([{"role": "user", "content": "Message A"}])

        client._response_content = "Response B"
        client.cached_completion([{"role": "user", "content": "Message B"}])

        # Should have 2 cached entries
        assert cache.entry_count() == 2
        assert client.call_count == 2


# Test that cached_completion registers LLMEntryEncoder automatically
def test_cached_completion_registers_encoder():
    from causaliq_core.cache import TokenCache

    MockClient = _create_mock_client_class()
    client = MockClient(LLMConfig(model="test-model"))
    client._response_content = "Test"

    with TokenCache(":memory:") as cache:
        client.set_cache(cache)

        # No encoder registered initially
        assert not cache.has_encoder("llm")

        client.cached_completion([{"role": "user", "content": "Hello"}])

        # Should auto-register encoder
        assert cache.has_encoder("llm")


# Test that cached_completion captures latency
def test_cached_completion_captures_latency():
    from causaliq_core.cache import TokenCache

    from causaliq_knowledge.llm.cache import LLMCacheEntry

    MockClient = _create_mock_client_class()
    client = MockClient(LLMConfig(model="test-model"))
    client._response_content = "Test response"

    with TokenCache(":memory:") as cache:
        client.set_cache(cache)

        client.cached_completion([{"role": "user", "content": "Hello"}])

        # Retrieve the cached entry and check latency
        cache_key = client._build_cache_key(
            [{"role": "user", "content": "Hello"}]
        )
        cached_data = cache.get_data(cache_key, "llm")
        entry = LLMCacheEntry.from_dict(cached_data)

        # Latency should be captured (>=0, mock is fast so usually 0-1ms)
        assert entry.metadata.latency_ms >= 0


# Test that cached_completion captures timestamp
def test_cached_completion_captures_timestamp():
    from causaliq_core.cache import TokenCache

    from causaliq_knowledge.llm.cache import LLMCacheEntry

    MockClient = _create_mock_client_class()
    client = MockClient(LLMConfig(model="test-model"))
    client._response_content = "Test response"

    with TokenCache(":memory:") as cache:
        client.set_cache(cache)

        client.cached_completion([{"role": "user", "content": "Hello"}])

        cache_key = client._build_cache_key(
            [{"role": "user", "content": "Hello"}]
        )
        cached_data = cache.get_data(cache_key, "llm")
        entry = LLMCacheEntry.from_dict(cached_data)

        # Timestamp should be set
        assert entry.metadata.timestamp is not None
        assert len(entry.metadata.timestamp) > 0


# Test that cached_completion handles invalid timestamp in cache gracefully.
def test_cached_completion_handles_invalid_timestamp():
    from causaliq_core.cache import TokenCache

    from causaliq_knowledge.llm.cache import (
        LLMCacheEntry,
        LLMEntryEncoder,
        LLMMetadata,
    )
    from causaliq_knowledge.llm.cache import LLMResponse as CachedLLMResponse
    from causaliq_knowledge.llm.cache import (
        LLMTokenUsage,
    )

    MockClient = _create_mock_client_class()
    client = MockClient(LLMConfig(model="test-model"))

    with TokenCache(":memory:") as cache:
        client.set_cache(cache)

        # Manually insert a cache entry with an invalid timestamp
        cache_key = client._build_cache_key(
            [{"role": "user", "content": "Hello"}]
        )

        entry = LLMCacheEntry(
            model="test-model",
            response=CachedLLMResponse(content="Cached response"),
            metadata=LLMMetadata(
                timestamp="invalid-timestamp-format",  # Invalid format
                tokens=LLMTokenUsage(input=10, output=20, total=30),
            ),
        )

        encoder = LLMEntryEncoder()
        cache.register_encoder("llm", encoder)
        cache.put_data(cache_key, "llm", entry.to_dict())

        # Call cached_completion - should retrieve from cache without error
        response = client.cached_completion(
            [{"role": "user", "content": "Hello"}]
        )

        # Should return cached content despite invalid timestamp
        assert response.content == "Cached response"
        # llm_timestamp should be None due to parse failure
        assert response.llm_timestamp is None
