"""Unit tests for TokenCache encoder integration."""

import pytest

from causaliq_knowledge.cache import TokenCache
from causaliq_knowledge.cache.encoders import JsonEncoder

# --- register_encoder tests ---


# Test registering an encoder.
def test_register_encoder():
    with TokenCache(":memory:") as cache:
        encoder = JsonEncoder()
        cache.register_encoder("json", encoder)
        assert cache.has_encoder("json")


# Test registering multiple encoders.
def test_register_multiple_encoders():
    with TokenCache(":memory:") as cache:
        cache.register_encoder("json", JsonEncoder())
        cache.register_encoder("data", JsonEncoder())
        assert cache.has_encoder("json")
        assert cache.has_encoder("data")


# Test has_encoder returns False for unregistered type.
def test_has_encoder_unregistered():
    with TokenCache(":memory:") as cache:
        assert not cache.has_encoder("unknown")


# Test get_encoder returns registered encoder.
def test_get_encoder():
    with TokenCache(":memory:") as cache:
        encoder = JsonEncoder()
        cache.register_encoder("json", encoder)
        assert cache.get_encoder("json") is encoder


# Test get_encoder returns None for unregistered type.
def test_get_encoder_unregistered():
    with TokenCache(":memory:") as cache:
        assert cache.get_encoder("unknown") is None


# Test replacing an encoder.
def test_replace_encoder():
    with TokenCache(":memory:") as cache:
        encoder1 = JsonEncoder()
        encoder2 = JsonEncoder()
        cache.register_encoder("json", encoder1)
        cache.register_encoder("json", encoder2)
        assert cache.get_encoder("json") is encoder2


# --- put_data/get_data basic tests ---


# Test put_data and get_data with simple dict.
def test_put_get_data_simple_dict():
    with TokenCache(":memory:") as cache:
        cache.register_encoder("json", JsonEncoder())
        data = {"key": "value"}
        cache.put_data("hash1", "json", data)
        result = cache.get_data("hash1", "json")
        assert result == data


# Test put_data and get_data with nested structure.
def test_put_get_data_nested():
    with TokenCache(":memory:") as cache:
        cache.register_encoder("json", JsonEncoder())
        data = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
            "temperature": 0.7,
        }
        cache.put_data("hash1", "json", data)
        result = cache.get_data("hash1", "json")
        assert result["messages"] == data["messages"]
        assert result["temperature"] == pytest.approx(0.7)


# Test get_data returns None for missing entry.
def test_get_data_missing():
    with TokenCache(":memory:") as cache:
        cache.register_encoder("json", JsonEncoder())
        assert cache.get_data("nonexistent", "json") is None


# Test put_data with list.
def test_put_get_data_list():
    with TokenCache(":memory:") as cache:
        cache.register_encoder("json", JsonEncoder())
        data = [1, 2, 3, "four", 5.0]
        cache.put_data("hash1", "json", data)
        result = cache.get_data("hash1", "json")
        assert result[0] == 1
        assert result[3] == "four"
        assert result[4] == pytest.approx(5.0)


# Test put_data with string.
def test_put_get_data_string():
    with TokenCache(":memory:") as cache:
        cache.register_encoder("json", JsonEncoder())
        cache.put_data("hash1", "json", "hello world")
        assert cache.get_data("hash1", "json") == "hello world"


# Test put_data with integer.
def test_put_get_data_integer():
    with TokenCache(":memory:") as cache:
        cache.register_encoder("json", JsonEncoder())
        cache.put_data("hash1", "json", 42)
        assert cache.get_data("hash1", "json") == 42


# Test put_data with float.
def test_put_get_data_float():
    with TokenCache(":memory:") as cache:
        cache.register_encoder("json", JsonEncoder())
        cache.put_data("hash1", "json", 3.14159)
        assert cache.get_data("hash1", "json") == pytest.approx(3.14159)


# Test put_data with boolean.
def test_put_get_data_boolean():
    with TokenCache(":memory:") as cache:
        cache.register_encoder("json", JsonEncoder())
        cache.put_data("hash1", "json", True)
        cache.put_data("hash2", "json", False)
        assert cache.get_data("hash1", "json") is True
        assert cache.get_data("hash2", "json") is False


# Test put_data with None.
def test_put_get_data_none():
    with TokenCache(":memory:") as cache:
        cache.register_encoder("json", JsonEncoder())
        cache.put_data("hash1", "json", None)
        assert cache.get_data("hash1", "json") is None


# --- put_data/get_data with metadata ---


# Test put_data with metadata.
def test_put_get_data_with_metadata():
    with TokenCache(":memory:") as cache:
        cache.register_encoder("json", JsonEncoder())
        data = {"response": "Hello"}
        metadata = {"latency_ms": 150, "model": "gpt-4"}
        cache.put_data("hash1", "json", data, metadata=metadata)
        result = cache.get_data_with_metadata("hash1", "json")
        assert result is not None
        decoded_data, decoded_meta = result
        assert decoded_data == data
        assert decoded_meta["latency_ms"] == 150
        assert decoded_meta["model"] == "gpt-4"


# Test get_data_with_metadata when no metadata stored.
def test_get_data_with_metadata_no_metadata():
    with TokenCache(":memory:") as cache:
        cache.register_encoder("json", JsonEncoder())
        cache.put_data("hash1", "json", {"data": "value"})
        result = cache.get_data_with_metadata("hash1", "json")
        assert result is not None
        decoded_data, decoded_meta = result
        assert decoded_data == {"data": "value"}
        assert decoded_meta is None


# Test get_data_with_metadata for missing entry.
def test_get_data_with_metadata_missing():
    with TokenCache(":memory:") as cache:
        cache.register_encoder("json", JsonEncoder())
        assert cache.get_data_with_metadata("nonexistent", "json") is None


# --- Error handling ---


# Test put_data raises KeyError for unregistered encoder.
def test_put_data_unregistered_encoder():
    with TokenCache(":memory:") as cache:
        with pytest.raises(KeyError):
            cache.put_data("hash1", "unknown", {"data": "value"})


# Test get_data raises KeyError for unregistered encoder.
def test_get_data_unregistered_encoder():
    with TokenCache(":memory:") as cache:
        # First store raw data
        cache.put("hash1", "unknown", b"raw data")
        with pytest.raises(KeyError):
            cache.get_data("hash1", "unknown")


# Test get_data_with_metadata raises KeyError for unregistered encoder.
def test_get_data_with_metadata_unregistered_encoder():
    with TokenCache(":memory:") as cache:
        cache.put("hash1", "unknown", b"raw data")
        with pytest.raises(KeyError):
            cache.get_data_with_metadata("hash1", "unknown")


# --- Token reuse across entries ---


# Test that tokens are shared across multiple entries.
def test_token_reuse_across_entries():
    with TokenCache(":memory:") as cache:
        cache.register_encoder("json", JsonEncoder())
        # Store multiple entries with common terms
        cache.put_data("hash1", "json", {"role": "user", "content": "Hello"})
        cache.put_data("hash2", "json", {"role": "assistant", "content": "Hi"})
        cache.put_data("hash3", "json", {"role": "user", "content": "Bye"})
        # "role", "user", "content" should be shared
        # Token count should be less than if each was unique
        token_count = cache.token_count()
        # With sharing: ", {, }, :, ,, role, user, assistant, content,
        #               Hello, Hi, Bye, [ and ] if any
        # Should be < 20 tokens for this data
        assert token_count < 20


# --- Overwrite behavior ---


# Test that put_data overwrites existing entry.
def test_put_data_overwrites():
    with TokenCache(":memory:") as cache:
        cache.register_encoder("json", JsonEncoder())
        cache.put_data("hash1", "json", {"version": 1})
        cache.put_data("hash1", "json", {"version": 2})
        result = cache.get_data("hash1", "json")
        assert result == {"version": 2}
        assert cache.entry_count() == 1


# --- Different entry types ---


# Test multiple entry types with same hash.
def test_different_entry_types_same_hash():
    with TokenCache(":memory:") as cache:
        cache.register_encoder("json", JsonEncoder())
        cache.register_encoder("data", JsonEncoder())
        cache.put_data("hash1", "json", {"type": "json"})
        cache.put_data("hash1", "data", {"type": "data"})
        assert cache.get_data("hash1", "json") == {"type": "json"}
        assert cache.get_data("hash1", "data") == {"type": "data"}
        assert cache.entry_count() == 2


# --- Integration with raw put/get ---


# Test that put_data entries can be read with raw get.
def test_put_data_read_raw():
    with TokenCache(":memory:") as cache:
        cache.register_encoder("json", JsonEncoder())
        cache.put_data("hash1", "json", {"key": "value"})
        raw = cache.get("hash1", "json")
        # Raw should be bytes (encoded)
        assert isinstance(raw, bytes)
        assert len(raw) > 0


# Test that raw put can be decoded with get_data.
def test_raw_put_read_with_get_data():
    with TokenCache(":memory:") as cache:
        encoder = JsonEncoder()
        cache.register_encoder("json", encoder)
        # Manually encode and store
        data = {"key": "value"}
        blob = encoder.encode(data, cache)
        cache.put("hash1", "json", blob)
        # Should be able to read with get_data
        result = cache.get_data("hash1", "json")
        assert result == data
