"""Unit tests for LLM cache encoder and data structures."""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest
from causaliq_core.cache import TokenCache

from causaliq_knowledge.llm.cache import (
    LLMCacheEntry,
    LLMEntryEncoder,
    LLMMetadata,
    LLMResponse,
    LLMTokenUsage,
)

# =============================================================================
# LLMTokenUsage Tests
# =============================================================================


# Test LLMTokenUsage default values.
def test_llm_token_usage_defaults():
    usage = LLMTokenUsage()
    assert usage.input == 0
    assert usage.output == 0
    assert usage.total == 0


# Test LLMTokenUsage with custom values.
def test_llm_token_usage_custom_values():
    usage = LLMTokenUsage(input=100, output=50, total=150)
    assert usage.input == 100
    assert usage.output == 50
    assert usage.total == 150


# Test LLMTokenUsage equality.
def test_llm_token_usage_equality():
    usage1 = LLMTokenUsage(input=10, output=20, total=30)
    usage2 = LLMTokenUsage(input=10, output=20, total=30)
    assert usage1 == usage2


# =============================================================================
# LLMMetadata Tests
# =============================================================================


# Test LLMMetadata default values.
def test_llm_metadata_defaults():
    meta = LLMMetadata()
    assert meta.provider == ""
    assert meta.timestamp == ""
    assert meta.latency_ms == 0
    assert meta.tokens == LLMTokenUsage()
    assert meta.cost_usd == 0.0
    assert meta.cache_hit is False
    assert meta.request_id == ""


# Test LLMMetadata with custom values.
def test_llm_metadata_custom_values():
    tokens = LLMTokenUsage(input=500, output=100, total=600)
    meta = LLMMetadata(
        provider="anthropic",
        timestamp="2024-01-15T10:30:00+00:00",
        latency_ms=1234,
        tokens=tokens,
        cost_usd=0.025,
        cache_hit=True,
        request_id="expt42",
    )
    assert meta.provider == "anthropic"
    assert meta.timestamp == "2024-01-15T10:30:00+00:00"
    assert meta.latency_ms == 1234
    assert meta.tokens == tokens
    assert meta.cost_usd == 0.025
    assert meta.cache_hit is True
    assert meta.request_id == "expt42"


# Test LLMMetadata to_dict conversion.
def test_llm_metadata_to_dict():
    tokens = LLMTokenUsage(input=100, output=50, total=150)
    meta = LLMMetadata(
        provider="openai",
        timestamp="2024-01-01T00:00:00Z",
        latency_ms=500,
        tokens=tokens,
        cost_usd=0.01,
        cache_hit=False,
        request_id="test123",
    )
    result = meta.to_dict()
    expected = {
        "provider": "openai",
        "timestamp": "2024-01-01T00:00:00Z",
        "latency_ms": 500,
        "tokens": {"input": 100, "output": 50, "total": 150},
        "cost_usd": 0.01,
        "cache_hit": False,
        "request_id": "test123",
    }
    assert result == expected


# Test LLMMetadata from_dict conversion.
def test_llm_metadata_from_dict():
    data = {
        "provider": "gemini",
        "timestamp": "2024-06-01T12:00:00Z",
        "latency_ms": 750,
        "tokens": {"input": 200, "output": 100, "total": 300},
        "cost_usd": 0.005,
        "cache_hit": True,
        "request_id": "run01",
    }
    meta = LLMMetadata.from_dict(data)
    assert meta.provider == "gemini"
    assert meta.timestamp == "2024-06-01T12:00:00Z"
    assert meta.latency_ms == 750
    assert meta.tokens.input == 200
    assert meta.tokens.output == 100
    assert meta.tokens.total == 300
    assert meta.cost_usd == 0.005
    assert meta.cache_hit is True
    assert meta.request_id == "run01"


# Test LLMMetadata from_dict with missing fields uses defaults.
def test_llm_metadata_from_dict_missing_fields():
    data = {"provider": "ollama"}
    meta = LLMMetadata.from_dict(data)
    assert meta.provider == "ollama"
    assert meta.timestamp == ""
    assert meta.latency_ms == 0
    assert meta.tokens.input == 0
    assert meta.tokens.output == 0
    assert meta.tokens.total == 0
    assert meta.cost_usd == 0.0
    assert meta.cache_hit is False
    assert meta.request_id == ""


# Test LLMMetadata round-trip via dict.
def test_llm_metadata_round_trip():
    tokens = LLMTokenUsage(input=1000, output=500, total=1500)
    original = LLMMetadata(
        provider="anthropic",
        timestamp="2024-03-15T08:30:00+00:00",
        latency_ms=2500,
        tokens=tokens,
        cost_usd=0.075,
        cache_hit=False,
        request_id="batch_run_42",
    )
    restored = LLMMetadata.from_dict(original.to_dict())
    assert restored.provider == original.provider
    assert restored.timestamp == original.timestamp
    assert restored.latency_ms == original.latency_ms
    assert restored.tokens.input == original.tokens.input
    assert restored.tokens.output == original.tokens.output
    assert restored.tokens.total == original.tokens.total
    assert restored.cost_usd == original.cost_usd
    assert restored.cache_hit == original.cache_hit
    assert restored.request_id == original.request_id


# =============================================================================
# LLMResponse Tests
# =============================================================================


# Test LLMResponse default values.
def test_llm_response_defaults():
    resp = LLMResponse()
    assert resp.content == ""
    assert resp.finish_reason == "stop"
    assert resp.model_version == ""


# Test LLMResponse with custom values.
def test_llm_response_custom_values():
    resp = LLMResponse(
        content="Hello, world!",
        finish_reason="length",
        model_version="gpt-4-0125-preview",
    )
    assert resp.content == "Hello, world!"
    assert resp.finish_reason == "length"
    assert resp.model_version == "gpt-4-0125-preview"


# Test LLMResponse to_dict conversion.
def test_llm_response_to_dict():
    resp = LLMResponse(
        content="Test response",
        finish_reason="stop",
        model_version="claude-3-opus-20240229",
    )
    result = resp.to_dict()
    expected = {
        "content": "Test response",
        "finish_reason": "stop",
        "model_version": "claude-3-opus-20240229",
    }
    assert result == expected


# Test LLMResponse from_dict conversion.
def test_llm_response_from_dict():
    data = {
        "content": "Restored response",
        "finish_reason": "length",
        "model_version": "gemini-pro",
    }
    resp = LLMResponse.from_dict(data)
    assert resp.content == "Restored response"
    assert resp.finish_reason == "length"
    assert resp.model_version == "gemini-pro"


# Test LLMResponse from_dict with missing fields uses defaults.
def test_llm_response_from_dict_missing_fields():
    data = {"content": "Just content"}
    resp = LLMResponse.from_dict(data)
    assert resp.content == "Just content"
    assert resp.finish_reason == "stop"
    assert resp.model_version == ""


# Test LLMResponse round-trip via dict.
def test_llm_response_round_trip():
    original = LLMResponse(
        content="A long response with multiple sentences.",
        finish_reason="stop",
        model_version="gpt-4-turbo",
    )
    restored = LLMResponse.from_dict(original.to_dict())
    assert restored == original


# Test LLMResponse to_export_dict parses JSON content.
def test_llm_response_to_export_dict_parses_json():
    json_content = '{"edges": [{"source": "A", "target": "B"}]}'
    resp = LLMResponse(content=json_content, finish_reason="stop")
    export = resp.to_export_dict()
    assert export["content"] == {"edges": [{"source": "A", "target": "B"}]}
    assert export["finish_reason"] == "stop"


# Test LLMResponse to_export_dict keeps non-JSON as string.
def test_llm_response_to_export_dict_keeps_string_for_non_json():
    resp = LLMResponse(content="Plain text response", finish_reason="stop")
    export = resp.to_export_dict()
    assert export["content"] == "Plain text response"


# Test LLMResponse from_dict handles parsed JSON content from export files.
def test_llm_response_from_dict_handles_parsed_json():
    data = {
        "content": {"edges": [{"source": "A", "target": "B"}]},
        "finish_reason": "stop",
        "model_version": "test-model",
    }
    resp = LLMResponse.from_dict(data)
    # Content should be serialised back to string
    assert resp.content == '{"edges": [{"source": "A", "target": "B"}]}'
    assert resp.finish_reason == "stop"


# =============================================================================
# LLMCacheEntry Tests
# =============================================================================


# Test LLMCacheEntry default values.
def test_llm_cache_entry_defaults():
    entry = LLMCacheEntry()
    assert entry.model == ""
    assert entry.messages == []
    assert entry.temperature == 0.0
    assert entry.max_tokens is None
    assert entry.response == LLMResponse()
    assert entry.metadata == LLMMetadata()


# Test LLMCacheEntry with custom values.
def test_llm_cache_entry_custom_values():
    messages = [{"role": "user", "content": "Hello"}]
    response = LLMResponse(content="Hi!", finish_reason="stop")
    metadata = LLMMetadata(provider="openai", latency_ms=200)
    entry = LLMCacheEntry(
        model="gpt-4",
        messages=messages,
        temperature=0.7,
        max_tokens=1000,
        response=response,
        metadata=metadata,
    )
    assert entry.model == "gpt-4"
    assert entry.messages == messages
    assert entry.temperature == 0.7
    assert entry.max_tokens == 1000
    assert entry.response == response
    assert entry.metadata == metadata


# Test LLMCacheEntry to_dict structure.
def test_llm_cache_entry_to_dict():
    messages = [{"role": "user", "content": "What is 2+2?"}]
    response = LLMResponse(
        content="4", finish_reason="stop", model_version="gpt-4"
    )
    metadata = LLMMetadata(
        provider="openai",
        timestamp="2024-01-01T00:00:00Z",
        latency_ms=100,
        tokens=LLMTokenUsage(input=10, output=1, total=11),
        cost_usd=0.001,
    )
    entry = LLMCacheEntry(
        model="gpt-4",
        messages=messages,
        temperature=0.0,
        max_tokens=100,
        response=response,
        metadata=metadata,
    )
    result = entry.to_dict()
    assert "cache_key" in result
    assert result["cache_key"]["model"] == "gpt-4"
    assert result["cache_key"]["messages"] == messages
    assert result["cache_key"]["temperature"] == 0.0
    assert result["cache_key"]["max_tokens"] == 100
    assert "response" in result
    assert result["response"]["content"] == "4"
    assert "metadata" in result
    assert result["metadata"]["provider"] == "openai"


# Test LLMCacheEntry to_export_dict parses JSON response content.
def test_llm_cache_entry_to_export_dict():
    json_content = '{"edges": [{"source": "A", "target": "B"}]}'
    entry = LLMCacheEntry(
        model="test-model",
        messages=[{"role": "user", "content": "Test"}],
        response=LLMResponse(content=json_content, finish_reason="stop"),
    )
    result = entry.to_export_dict()
    # Response content should be parsed JSON
    assert result["response"]["content"] == {
        "edges": [{"source": "A", "target": "B"}]
    }
    assert result["cache_key"]["model"] == "test-model"


# Test LLMCacheEntry to_export_dict splits multiline message content.
def test_llm_cache_entry_to_export_dict_splits_message_lines():
    multiline_content = "Line 1\nLine 2\nLine 3"
    entry = LLMCacheEntry(
        model="test-model",
        messages=[
            {"role": "system", "content": multiline_content},
            {"role": "user", "content": "Single line"},
        ],
    )
    result = entry.to_export_dict()
    # Multiline content should be split into array
    assert result["cache_key"]["messages"][0]["content"] == [
        "Line 1",
        "Line 2",
        "Line 3",
    ]
    # Single line content stays as string
    assert result["cache_key"]["messages"][1]["content"] == "Single line"


# Test LLMCacheEntry from_dict joins array message content.
def test_llm_cache_entry_from_dict_joins_array_content():
    data = {
        "cache_key": {
            "model": "test-model",
            "messages": [
                {"role": "system", "content": ["Line 1", "Line 2", "Line 3"]},
                {"role": "user", "content": "Single line"},
            ],
        },
    }
    entry = LLMCacheEntry.from_dict(data)
    # Array content should be joined with newlines
    assert entry.messages[0]["content"] == "Line 1\nLine 2\nLine 3"
    # String content stays as string
    assert entry.messages[1]["content"] == "Single line"


# Test LLMCacheEntry from_dict conversion.
def test_llm_cache_entry_from_dict():
    data = {
        "cache_key": {
            "model": "claude-3",
            "messages": [{"role": "user", "content": "Hi"}],
            "temperature": 0.5,
            "max_tokens": 500,
        },
        "response": {
            "content": "Hello!",
            "finish_reason": "stop",
            "model_version": "claude-3-sonnet-20240229",
        },
        "metadata": {
            "provider": "anthropic",
            "timestamp": "2024-02-01T12:00:00Z",
            "latency_ms": 300,
            "tokens": {"input": 5, "output": 1, "total": 6},
            "cost_usd": 0.0001,
            "cache_hit": False,
        },
    }
    entry = LLMCacheEntry.from_dict(data)
    assert entry.model == "claude-3"
    assert entry.messages == [{"role": "user", "content": "Hi"}]
    assert entry.temperature == 0.5
    assert entry.max_tokens == 500
    assert entry.response.content == "Hello!"
    assert entry.metadata.provider == "anthropic"


# Test LLMCacheEntry from_dict with missing fields.
def test_llm_cache_entry_from_dict_missing_fields():
    data = {"cache_key": {"model": "gpt-4"}}
    entry = LLMCacheEntry.from_dict(data)
    assert entry.model == "gpt-4"
    assert entry.messages == []
    assert entry.temperature == 0.0
    assert entry.max_tokens is None
    assert entry.response.content == ""
    assert entry.metadata.provider == ""


# Test LLMCacheEntry round-trip via dict.
def test_llm_cache_entry_round_trip():
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello!"},
    ]
    original = LLMCacheEntry(
        model="gpt-4-turbo",
        messages=messages,
        temperature=0.8,
        max_tokens=2048,
        response=LLMResponse(
            content="Hi there!",
            finish_reason="stop",
            model_version="gpt-4-turbo-2024-04-09",
        ),
        metadata=LLMMetadata(
            provider="openai",
            timestamp="2024-04-10T09:00:00Z",
            latency_ms=1500,
            tokens=LLMTokenUsage(input=50, output=10, total=60),
            cost_usd=0.005,
            cache_hit=False,
        ),
    )
    restored = LLMCacheEntry.from_dict(original.to_dict())
    assert restored.model == original.model
    assert restored.messages == original.messages
    assert restored.temperature == original.temperature
    assert restored.max_tokens == original.max_tokens
    assert restored.response.content == original.response.content
    assert restored.metadata.provider == original.metadata.provider


# Test LLMCacheEntry.create factory method.
def test_llm_cache_entry_create_basic():
    entry = LLMCacheEntry.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
        content="Hi!",
    )
    assert entry.model == "gpt-4"
    assert entry.messages == [{"role": "user", "content": "Hello"}]
    assert entry.response.content == "Hi!"
    assert entry.response.finish_reason == "stop"
    assert entry.response.model_version == "gpt-4"
    assert entry.temperature == 0.0
    assert entry.max_tokens is None
    assert entry.metadata.cache_hit is False


# Test LLMCacheEntry.create with all optional parameters.
def test_llm_cache_entry_create_full():
    entry = LLMCacheEntry.create(
        model="claude-3-opus",
        messages=[{"role": "user", "content": "Explain caching"}],
        content="Caching is a technique...",
        temperature=0.7,
        max_tokens=4096,
        finish_reason="stop",
        model_version="claude-3-opus-20240229",
        provider="anthropic",
        latency_ms=2500,
        input_tokens=20,
        output_tokens=100,
        cost_usd=0.05,
        request_id="expt_batch_01",
    )
    assert entry.model == "claude-3-opus"
    assert entry.temperature == 0.7
    assert entry.max_tokens == 4096
    assert entry.response.content == "Caching is a technique..."
    assert entry.response.finish_reason == "stop"
    assert entry.response.model_version == "claude-3-opus-20240229"
    assert entry.metadata.provider == "anthropic"
    assert entry.metadata.latency_ms == 2500
    assert entry.metadata.tokens.input == 20
    assert entry.metadata.tokens.output == 100
    assert entry.metadata.tokens.total == 120
    assert entry.metadata.cost_usd == 0.05
    assert entry.metadata.cache_hit is False
    assert entry.metadata.request_id == "expt_batch_01"


# Test LLMCacheEntry.create generates valid timestamp.
def test_llm_cache_entry_create_timestamp():
    before = datetime.now(timezone.utc)
    entry = LLMCacheEntry.create(
        model="gpt-4",
        messages=[],
        content="Test",
    )
    after = datetime.now(timezone.utc)
    timestamp = datetime.fromisoformat(entry.metadata.timestamp)
    assert before <= timestamp <= after


# Test LLMCacheEntry.create uses model as model_version when not specified.
def test_llm_cache_entry_create_model_version_fallback():
    entry = LLMCacheEntry.create(
        model="gemini-pro",
        messages=[],
        content="Response",
    )
    assert entry.response.model_version == "gemini-pro"


# Test LLMCacheEntry.create with explicit model_version.
def test_llm_cache_entry_create_model_version_explicit():
    entry = LLMCacheEntry.create(
        model="gpt-4",
        messages=[],
        content="Response",
        model_version="gpt-4-0125-preview",
    )
    assert entry.response.model_version == "gpt-4-0125-preview"


# =============================================================================
# LLMEntryEncoder Tests
# =============================================================================


# Test LLMEntryEncoder inherits from JsonCompressor.
def test_llm_entry_encoder_is_json_compressor():
    from causaliq_core.cache.compressors import JsonCompressor

    encoder = LLMEntryEncoder()
    assert isinstance(encoder, JsonCompressor)


# Test LLMEntryEncoder.encode_entry encodes to bytes.
def test_llm_entry_encoder_encode_entry():
    with TokenCache(":memory:") as cache:
        encoder = LLMEntryEncoder()
        entry = LLMCacheEntry.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            content="Hi!",
            provider="openai",
        )
        blob = encoder.encode_entry(entry, cache)
        assert isinstance(blob, bytes)
        assert len(blob) > 0


# Test LLMEntryEncoder.decode_entry decodes to LLMCacheEntry.
def test_llm_entry_encoder_decode_entry():
    with TokenCache(":memory:") as cache:
        encoder = LLMEntryEncoder()
        entry = LLMCacheEntry.create(
            model="claude-3",
            messages=[{"role": "user", "content": "Test"}],
            content="Response",
            provider="anthropic",
        )
        blob = encoder.encode_entry(entry, cache)
        restored = encoder.decode_entry(blob, cache)
        assert isinstance(restored, LLMCacheEntry)
        assert restored.model == entry.model
        assert restored.response.content == entry.response.content


# Test LLMEntryEncoder encode/decode round-trip.
def test_llm_entry_encoder_round_trip():
    with TokenCache(":memory:") as cache:
        encoder = LLMEntryEncoder()
        original = LLMCacheEntry.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "What is Python?"},
            ],
            content="Python is a programming language.",
            temperature=0.5,
            max_tokens=1000,
            finish_reason="stop",
            model_version="gpt-4-turbo-2024-04-09",
            provider="openai",
            latency_ms=800,
            input_tokens=25,
            output_tokens=10,
            cost_usd=0.002,
        )
        blob = encoder.encode_entry(original, cache)
        restored = encoder.decode_entry(blob, cache)

        assert restored.model == original.model
        assert restored.messages == original.messages
        assert restored.temperature == original.temperature
        assert restored.max_tokens == original.max_tokens
        assert restored.response.content == original.response.content
        assert (
            restored.response.finish_reason == original.response.finish_reason
        )
        assert (
            restored.response.model_version == original.response.model_version
        )
        assert restored.metadata.provider == original.metadata.provider
        assert restored.metadata.latency_ms == original.metadata.latency_ms
        assert restored.metadata.tokens.input == original.metadata.tokens.input
        assert (
            restored.metadata.tokens.output == original.metadata.tokens.output
        )
        assert restored.metadata.tokens.total == original.metadata.tokens.total
        assert restored.metadata.cost_usd == pytest.approx(
            original.metadata.cost_usd
        )
        assert restored.metadata.cache_hit == original.metadata.cache_hit


# Test LLMEntryEncoder with complex conversation.
def test_llm_entry_encoder_complex_conversation():
    with TokenCache(":memory:") as cache:
        encoder = LLMEntryEncoder()
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Write a haiku about coding."},
            {
                "role": "assistant",
                "content": (
                    "Lines of code cascade\n"
                    "Bugs hide in the logic flow\n"
                    "Debug, fix, repeat"
                ),
            },
            {"role": "user", "content": "Now one about Python."},
        ]
        haiku = (
            "Indentation rules\n"
            "Whitespace has meaning here\n"
            "Zen of Python guides"
        )
        original = LLMCacheEntry.create(
            model="gpt-4",
            messages=messages,
            content=haiku,
            provider="openai",
        )
        blob = encoder.encode_entry(original, cache)
        restored = encoder.decode_entry(blob, cache)
        assert restored.messages == messages
        assert restored.response.content == haiku


# Test LLMEntryEncoder.export_entry writes JSON file.
def test_llm_entry_encoder_export_entry():
    encoder = LLMEntryEncoder()
    entry = LLMCacheEntry.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Test"}],
        content="Response",
        provider="openai",
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "entry.json"
        encoder.export_entry(entry, path)
        assert path.exists()
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["cache_key"]["model"] == "gpt-4"
        assert data["response"]["content"] == "Response"


# Test LLMEntryEncoder.import_entry reads JSON file.
def test_llm_entry_encoder_import_entry():
    encoder = LLMEntryEncoder()
    data = {
        "cache_key": {
            "model": "claude-3",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.5,
            "max_tokens": 100,
        },
        "response": {
            "content": "Hi there!",
            "finish_reason": "stop",
            "model_version": "claude-3-sonnet-20240229",
        },
        "metadata": {
            "provider": "anthropic",
            "timestamp": "2024-01-01T00:00:00Z",
            "latency_ms": 200,
            "tokens": {"input": 5, "output": 3, "total": 8},
            "cost_usd": 0.0001,
            "cache_hit": False,
        },
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "entry.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        entry = encoder.import_entry(path)
        assert isinstance(entry, LLMCacheEntry)
        assert entry.model == "claude-3"
        assert entry.response.content == "Hi there!"
        assert entry.metadata.provider == "anthropic"


# Test LLMEntryEncoder export/import round-trip.
def test_llm_entry_encoder_export_import_round_trip():
    encoder = LLMEntryEncoder()
    original = LLMCacheEntry.create(
        model="gemini-pro",
        messages=[{"role": "user", "content": "What is the meaning of life?"}],
        content="42, according to Douglas Adams.",
        temperature=0.7,
        max_tokens=500,
        provider="google",
        latency_ms=400,
        input_tokens=10,
        output_tokens=8,
        cost_usd=0.0005,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "round_trip.json"
        encoder.export_entry(original, path)
        restored = encoder.import_entry(path)
        assert restored.model == original.model
        assert restored.messages == original.messages
        assert restored.temperature == original.temperature
        assert restored.max_tokens == original.max_tokens
        assert restored.response.content == original.response.content
        assert restored.metadata.provider == original.metadata.provider


# Test LLMEntryEncoder base encode/decode still works.
def test_llm_entry_encoder_base_encode_decode():
    with TokenCache(":memory:") as cache:
        encoder = LLMEntryEncoder()
        data = {"key": "value", "number": 42}
        blob = encoder.compress(data, cache)
        restored = encoder.decompress(blob, cache)
        assert restored == data


# Test LLMEntryEncoder can encode entry to_dict directly.
def test_llm_entry_encoder_encode_entry_dict():
    with TokenCache(":memory:") as cache:
        encoder = LLMEntryEncoder()
        entry = LLMCacheEntry.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            content="Hi!",
        )
        blob = encoder.compress(entry.to_dict(), cache)
        data = encoder.decompress(blob, cache)
        restored = LLMCacheEntry.from_dict(data)
        assert restored.model == entry.model


# Test LLMEntryEncoder with empty messages.
def test_llm_entry_encoder_empty_messages():
    with TokenCache(":memory:") as cache:
        encoder = LLMEntryEncoder()
        entry = LLMCacheEntry.create(
            model="gpt-4",
            messages=[],
            content="Default response",
        )
        blob = encoder.encode_entry(entry, cache)
        restored = encoder.decode_entry(blob, cache)
        assert restored.messages == []
        assert restored.response.content == "Default response"


# Test LLMEntryEncoder with None max_tokens.
def test_llm_entry_encoder_none_max_tokens():
    with TokenCache(":memory:") as cache:
        encoder = LLMEntryEncoder()
        entry = LLMCacheEntry.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Test"}],
            content="Response",
            max_tokens=None,
        )
        blob = encoder.encode_entry(entry, cache)
        restored = encoder.decode_entry(blob, cache)
        assert restored.max_tokens is None


# Test LLMEntryEncoder with zero temperature.
def test_llm_entry_encoder_zero_temperature():
    with TokenCache(":memory:") as cache:
        encoder = LLMEntryEncoder()
        entry = LLMCacheEntry.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Test"}],
            content="Deterministic response",
            temperature=0.0,
        )
        blob = encoder.encode_entry(entry, cache)
        restored = encoder.decode_entry(blob, cache)
        assert restored.temperature == 0.0


# Test LLMEntryEncoder with high temperature.
def test_llm_entry_encoder_high_temperature():
    with TokenCache(":memory:") as cache:
        encoder = LLMEntryEncoder()
        entry = LLMCacheEntry.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Be creative"}],
            content="Creative response",
            temperature=1.5,
        )
        blob = encoder.encode_entry(entry, cache)
        restored = encoder.decode_entry(blob, cache)
        assert restored.temperature == 1.5


# Test LLMEntryEncoder with special characters in content.
def test_llm_entry_encoder_special_characters():
    with TokenCache(":memory:") as cache:
        encoder = LLMEntryEncoder()
        content = "Special chars: © ® ™ € £ ¥ • → ← ↑ ↓ with newlines\n\ttab"
        entry = LLMCacheEntry.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Test special chars"}],
            content=content,
        )
        blob = encoder.encode_entry(entry, cache)
        restored = encoder.decode_entry(blob, cache)
        assert restored.response.content == content


# Test LLMEntryEncoder with Unicode content.
def test_llm_entry_encoder_unicode():
    with TokenCache(":memory:") as cache:
        encoder = LLMEntryEncoder()
        content = "Unicode: 你好世界 こんにちは мир العالم 🌍🎉🚀"
        entry = LLMCacheEntry.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": "Say hello in many languages"}
            ],
            content=content,
        )
        blob = encoder.encode_entry(entry, cache)
        restored = encoder.decode_entry(blob, cache)
        assert restored.response.content == content


# Test LLMEntryEncoder with long content.
def test_llm_entry_encoder_long_content():
    with TokenCache(":memory:") as cache:
        encoder = LLMEntryEncoder()
        content = "Lorem ipsum " * 1000
        entry = LLMCacheEntry.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Write a long response"}],
            content=content,
        )
        blob = encoder.encode_entry(entry, cache)
        restored = encoder.decode_entry(blob, cache)
        assert restored.response.content == content


# Test multiple entries can be encoded in same cache.
def test_llm_entry_encoder_multiple_entries():
    with TokenCache(":memory:") as cache:
        encoder = LLMEntryEncoder()
        entries = []
        for i in range(5):
            entry = LLMCacheEntry.create(
                model=f"model-{i}",
                messages=[{"role": "user", "content": f"Message {i}"}],
                content=f"Response {i}",
            )
            entries.append(entry)

        blobs = [encoder.encode_entry(e, cache) for e in entries]
        restored = [encoder.decode_entry(b, cache) for b in blobs]

        for i, entry in enumerate(restored):
            assert entry.model == f"model-{i}"
            assert entry.response.content == f"Response {i}"


# Test token dictionary is shared across entries.
def test_llm_entry_encoder_shared_tokens():
    with TokenCache(":memory:") as cache:
        encoder = LLMEntryEncoder()
        # First entry with common strings
        entry1 = LLMCacheEntry.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello world"}],
            content="Hello back!",
            provider="openai",
        )
        blob1 = encoder.encode_entry(entry1, cache)

        # Second entry with same common strings
        entry2 = LLMCacheEntry.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello again"}],
            content="Hello once more!",
            provider="openai",
        )
        blob2 = encoder.encode_entry(entry2, cache)

        # Both should decode correctly
        restored1 = encoder.decode_entry(blob1, cache)
        restored2 = encoder.decode_entry(blob2, cache)
        assert restored1.model == "gpt-4"
        assert restored2.model == "gpt-4"
        assert restored1.metadata.provider == "openai"
        assert restored2.metadata.provider == "openai"


# =============================================================================
# LLMEntryEncoder Filename Generation Tests
# =============================================================================


# Test generate_export_filename with request_id and provider.
def test_generate_export_filename_with_request_id():
    encoder = LLMEntryEncoder()
    entry = LLMCacheEntry.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "test"}],
        content="Response",
        provider="openai",
        request_id="expt23",
    )
    filename = encoder.generate_export_filename(entry, "abc12345")

    assert filename.endswith(".json")
    assert filename.startswith("expt23_")
    assert "_openai.json" in filename
    # Format: expt23_yyyy-mm-dd-hhmmss_openai.json
    parts = filename.replace(".json", "").split("_")
    assert len(parts) == 3
    assert parts[0] == "expt23"
    assert parts[2] == "openai"


# Test generate_export_filename falls back to hash when no request_id.
def test_generate_export_filename_no_request_id():
    encoder = LLMEntryEncoder()
    entry = LLMCacheEntry.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "test"}],
        content="Response",
        provider="openai",
    )
    filename = encoder.generate_export_filename(entry, "abc12345def")

    assert filename.endswith(".json")
    # Should use first 8 chars of hash as fallback
    assert filename.startswith("abc12345_")
    assert "_openai.json" in filename


# Test generate_export_filename sanitises request_id.
def test_generate_export_filename_sanitises_request_id():
    encoder = LLMEntryEncoder()
    entry = LLMCacheEntry.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "test"}],
        content="Response",
        provider="openai",
        request_id="test/with:special@chars!",
    )
    filename = encoder.generate_export_filename(entry, "abc12345")

    assert filename.endswith(".json")
    # Special chars should be removed
    assert filename.startswith("testwithspecialchars_")


# Test generate_export_filename with different providers.
def test_generate_export_filename_various_providers():
    encoder = LLMEntryEncoder()

    for provider in ["groq", "anthropic", "gemini", "deepseek"]:
        entry = LLMCacheEntry.create(
            model="test-model",
            messages=[{"role": "user", "content": "test"}],
            content="Response",
            provider=provider,
            request_id="test",
        )
        filename = encoder.generate_export_filename(entry, "hash123")

        assert filename.endswith(f"_{provider}.json")


# Test generate_export_filename handles missing provider.
def test_generate_export_filename_missing_provider():
    encoder = LLMEntryEncoder()
    entry = LLMCacheEntry.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "test"}],
        content="Response",
        request_id="test",
    )
    filename = encoder.generate_export_filename(entry, "abc123")

    assert filename.endswith("_unknown.json")


# Test generate_export_filename handles missing timestamp.
def test_generate_export_filename_missing_timestamp():
    encoder = LLMEntryEncoder()
    entry = LLMCacheEntry.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "test"}],
        content="Response",
        provider="openai",
        request_id="test",
    )
    # Clear the timestamp
    entry.metadata.timestamp = ""
    filename = encoder.generate_export_filename(entry, "abc123")

    assert filename.endswith(".json")
    assert "_unknown_" in filename


# Test generate_export_filename timestamp format.
def test_generate_export_filename_timestamp_format():
    encoder = LLMEntryEncoder()
    entry = LLMCacheEntry.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "test"}],
        content="Response",
        provider="openai",
        request_id="test",
    )
    # Set a known timestamp
    entry.metadata.timestamp = "2026-01-29T14:30:52+00:00"
    filename = encoder.generate_export_filename(entry, "abc123")

    # Should format as yyyy-mm-dd-hhmmss
    assert "2026-01-29-143052" in filename
    assert filename == "test_2026-01-29-143052_openai.json"


# Test generate_export_filename with request_id that sanitises to empty.
def test_generate_export_filename_request_id_sanitises_to_empty():
    encoder = LLMEntryEncoder()
    entry = LLMCacheEntry.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "test"}],
        content="Response",
        provider="openai",
        request_id="@#$%^&*()",  # All special chars, sanitises to empty
    )
    filename = encoder.generate_export_filename(entry, "fallback123")

    # Should use first 8 chars of cache_key as fallback
    assert filename.startswith("fallback_")
    assert "_openai.json" in filename


# Test generate_export_filename with invalid timestamp format.
def test_generate_export_filename_invalid_timestamp():
    encoder = LLMEntryEncoder()
    entry = LLMCacheEntry.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "test"}],
        content="Response",
        provider="openai",
        request_id="test",
    )
    # Set an invalid timestamp that will fail to parse
    entry.metadata.timestamp = "not-a-valid-timestamp"
    filename = encoder.generate_export_filename(entry, "abc123")

    # Should use "unknown" for timestamp
    assert filename == "test_unknown_openai.json"


# Test generate_export_filename with provider that sanitises to empty.
def test_generate_export_filename_provider_sanitises_to_empty():
    encoder = LLMEntryEncoder()
    entry = LLMCacheEntry.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "test"}],
        content="Response",
        provider="@#$%",  # All special chars, sanitises to empty
        request_id="test",
    )
    filename = encoder.generate_export_filename(entry, "abc123")

    # Should use "unknown" for provider
    assert filename.endswith("_unknown.json")
