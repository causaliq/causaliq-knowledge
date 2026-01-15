"""Unit tests for JsonEncoder."""

import struct

import pytest

from causaliq_knowledge.cache import TokenCache
from causaliq_knowledge.cache.encoders import JsonEncoder

# --- Default export format ---


# Test that default_export_format returns 'json'.
def test_default_export_format_is_json():
    encoder = JsonEncoder()
    assert encoder.default_export_format == "json"


# --- Encoding/decoding primitive types ---


# Test encoding and decoding None value.
def test_encode_decode_none():
    with TokenCache(":memory:") as cache:
        encoder = JsonEncoder()
        blob = encoder.encode(None, cache)
        assert encoder.decode(blob, cache) is None


# Test encoding and decoding boolean True.
def test_encode_decode_true():
    with TokenCache(":memory:") as cache:
        encoder = JsonEncoder()
        blob = encoder.encode(True, cache)
        assert encoder.decode(blob, cache) is True


# Test encoding and decoding boolean False.
def test_encode_decode_false():
    with TokenCache(":memory:") as cache:
        encoder = JsonEncoder()
        blob = encoder.encode(False, cache)
        assert encoder.decode(blob, cache) is False


# Test encoding and decoding positive integer.
def test_encode_decode_positive_int():
    with TokenCache(":memory:") as cache:
        encoder = JsonEncoder()
        blob = encoder.encode(42, cache)
        assert encoder.decode(blob, cache) == 42


# Test encoding and decoding negative integer.
def test_encode_decode_negative_int():
    with TokenCache(":memory:") as cache:
        encoder = JsonEncoder()
        blob = encoder.encode(-12345, cache)
        assert encoder.decode(blob, cache) == -12345


# Test encoding and decoding zero.
def test_encode_decode_zero():
    with TokenCache(":memory:") as cache:
        encoder = JsonEncoder()
        blob = encoder.encode(0, cache)
        assert encoder.decode(blob, cache) == 0


# Test encoding and decoding large integer.
def test_encode_decode_large_int():
    with TokenCache(":memory:") as cache:
        encoder = JsonEncoder()
        large = 2**62
        blob = encoder.encode(large, cache)
        assert encoder.decode(blob, cache) == large


# Test encoding and decoding float.
def test_encode_decode_float():
    with TokenCache(":memory:") as cache:
        encoder = JsonEncoder()
        blob = encoder.encode(3.14159, cache)
        assert encoder.decode(blob, cache) == pytest.approx(3.14159)


# Test encoding and decoding negative float.
def test_encode_decode_negative_float():
    with TokenCache(":memory:") as cache:
        encoder = JsonEncoder()
        blob = encoder.encode(-273.15, cache)
        assert encoder.decode(blob, cache) == pytest.approx(-273.15)


# Test encoding and decoding float zero.
def test_encode_decode_float_zero():
    with TokenCache(":memory:") as cache:
        encoder = JsonEncoder()
        blob = encoder.encode(0.0, cache)
        assert encoder.decode(blob, cache) == 0.0


# --- Encoding/decoding strings ---


# Test encoding and decoding simple string.
def test_encode_decode_simple_string():
    with TokenCache(":memory:") as cache:
        encoder = JsonEncoder()
        blob = encoder.encode("hello", cache)
        assert encoder.decode(blob, cache) == "hello"


# Test encoding and decoding empty string.
def test_encode_decode_empty_string():
    with TokenCache(":memory:") as cache:
        encoder = JsonEncoder()
        blob = encoder.encode("", cache)
        assert encoder.decode(blob, cache) == ""


# Test encoding and decoding string with spaces.
def test_encode_decode_string_with_spaces():
    with TokenCache(":memory:") as cache:
        encoder = JsonEncoder()
        blob = encoder.encode("hello world", cache)
        assert encoder.decode(blob, cache) == "hello world"


# Test encoding and decoding string with punctuation.
def test_encode_decode_string_with_punctuation():
    with TokenCache(":memory:") as cache:
        encoder = JsonEncoder()
        blob = encoder.encode("Hello, World!", cache)
        assert encoder.decode(blob, cache) == "Hello, World!"


# Test encoding and decoding string with numbers.
def test_encode_decode_string_with_numbers():
    with TokenCache(":memory:") as cache:
        encoder = JsonEncoder()
        blob = encoder.encode("test123", cache)
        assert encoder.decode(blob, cache) == "test123"


# Test encoding and decoding multi-word sentence.
def test_encode_decode_sentence():
    with TokenCache(":memory:") as cache:
        encoder = JsonEncoder()
        text = "BMI represents Body Mass Index"
        blob = encoder.encode(text, cache)
        assert encoder.decode(blob, cache) == text


# --- Encoding/decoding lists ---


# Test encoding and decoding empty list.
def test_encode_decode_empty_list():
    with TokenCache(":memory:") as cache:
        encoder = JsonEncoder()
        blob = encoder.encode([], cache)
        assert encoder.decode(blob, cache) == []


# Test encoding and decoding list of integers.
def test_encode_decode_list_of_ints():
    with TokenCache(":memory:") as cache:
        encoder = JsonEncoder()
        blob = encoder.encode([1, 2, 3], cache)
        assert encoder.decode(blob, cache) == [1, 2, 3]


# Test encoding and decoding list of strings.
def test_encode_decode_list_of_strings():
    with TokenCache(":memory:") as cache:
        encoder = JsonEncoder()
        blob = encoder.encode(["a", "b", "c"], cache)
        assert encoder.decode(blob, cache) == ["a", "b", "c"]


# Test encoding and decoding mixed list.
def test_encode_decode_mixed_list():
    with TokenCache(":memory:") as cache:
        encoder = JsonEncoder()
        data = [1, "two", 3.0, True, None]
        blob = encoder.encode(data, cache)
        decoded = encoder.decode(blob, cache)
        assert decoded[0] == 1
        assert decoded[1] == "two"
        assert decoded[2] == pytest.approx(3.0)
        assert decoded[3] is True
        assert decoded[4] is None


# Test encoding and decoding nested list.
def test_encode_decode_nested_list():
    with TokenCache(":memory:") as cache:
        encoder = JsonEncoder()
        data = [[1, 2], [3, 4]]
        blob = encoder.encode(data, cache)
        assert encoder.decode(blob, cache) == [[1, 2], [3, 4]]


# --- Encoding/decoding dicts ---


# Test encoding and decoding empty dict.
def test_encode_decode_empty_dict():
    with TokenCache(":memory:") as cache:
        encoder = JsonEncoder()
        blob = encoder.encode({}, cache)
        assert encoder.decode(blob, cache) == {}


# Test encoding and decoding simple dict.
def test_encode_decode_simple_dict():
    with TokenCache(":memory:") as cache:
        encoder = JsonEncoder()
        data = {"key": "value"}
        blob = encoder.encode(data, cache)
        assert encoder.decode(blob, cache) == data


# Test encoding and decoding dict with int values.
def test_encode_decode_dict_with_int():
    with TokenCache(":memory:") as cache:
        encoder = JsonEncoder()
        data = {"count": 42}
        blob = encoder.encode(data, cache)
        assert encoder.decode(blob, cache) == data


# Test encoding and decoding dict with float values.
def test_encode_decode_dict_with_float():
    with TokenCache(":memory:") as cache:
        encoder = JsonEncoder()
        data = {"score": 3.14159}
        blob = encoder.encode(data, cache)
        decoded = encoder.decode(blob, cache)
        assert decoded["score"] == pytest.approx(3.14159)


# Test encoding and decoding dict with multiple keys.
def test_encode_decode_dict_multiple_keys():
    with TokenCache(":memory:") as cache:
        encoder = JsonEncoder()
        data = {"a": 1, "b": 2, "c": 3}
        blob = encoder.encode(data, cache)
        assert encoder.decode(blob, cache) == data


# Test encoding and decoding nested dict.
def test_encode_decode_nested_dict():
    with TokenCache(":memory:") as cache:
        encoder = JsonEncoder()
        data = {"outer": {"inner": "value"}}
        blob = encoder.encode(data, cache)
        assert encoder.decode(blob, cache) == data


# Test encoding and decoding dict with list values.
def test_encode_decode_dict_with_list():
    with TokenCache(":memory:") as cache:
        encoder = JsonEncoder()
        data = {"items": [1, 2, 3]}
        blob = encoder.encode(data, cache)
        assert encoder.decode(blob, cache) == data


# Test encoding and decoding list of dicts.
def test_encode_decode_list_of_dicts():
    with TokenCache(":memory:") as cache:
        encoder = JsonEncoder()
        data = [{"a": 1}, {"b": 2}]
        blob = encoder.encode(data, cache)
        assert encoder.decode(blob, cache) == data


# --- Complex structures ---


# Test encoding and decoding LLM-like message structure.
def test_encode_decode_llm_messages():
    with TokenCache(":memory:") as cache:
        encoder = JsonEncoder()
        data = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is BMI?"},
            ],
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 100,
        }
        blob = encoder.encode(data, cache)
        decoded = encoder.decode(blob, cache)
        assert decoded["messages"] == data["messages"]
        assert decoded["model"] == data["model"]
        assert decoded["temperature"] == pytest.approx(data["temperature"])
        assert decoded["max_tokens"] == data["max_tokens"]


# Test encoding and decoding deeply nested structure.
def test_encode_decode_deeply_nested():
    with TokenCache(":memory:") as cache:
        encoder = JsonEncoder()
        data = {"l1": {"l2": {"l3": {"l4": {"value": "deep"}}}}}
        blob = encoder.encode(data, cache)
        assert encoder.decode(blob, cache) == data


# --- Token reuse ---


# Test that repeated strings reuse tokens.
def test_token_reuse():
    with TokenCache(":memory:") as cache:
        encoder = JsonEncoder()
        data = ["hello", "hello", "hello"]
        blob = encoder.encode(data, cache)
        encoder.decode(blob, cache)
        # "hello" should only create one token (plus structural tokens)
        # Check token count is reasonable (not 3 separate "hello" tokens)
        assert cache.token_count() < 10


# Test that encoding same data twice produces same result.
def test_deterministic_encoding():
    with TokenCache(":memory:") as cache:
        encoder = JsonEncoder()
        data = {"key": "value", "count": 42}
        blob1 = encoder.encode(data, cache)
        blob2 = encoder.encode(data, cache)
        assert blob1 == blob2


# --- Tokenisation helper ---


# Test tokenise_string with simple word.
def test_tokenise_string_simple():
    encoder = JsonEncoder()
    tokens = encoder._tokenise_string("hello")
    assert tokens == ["hello"]


# Test tokenise_string with multiple words.
def test_tokenise_string_multiple_words():
    encoder = JsonEncoder()
    tokens = encoder._tokenise_string("hello world")
    assert tokens == ["hello", " ", "world"]


# Test tokenise_string with punctuation.
def test_tokenise_string_punctuation():
    encoder = JsonEncoder()
    tokens = encoder._tokenise_string("Hello, World!")
    assert tokens == ["Hello", ",", " ", "World", "!"]


# Test tokenise_string empty string.
def test_tokenise_string_empty():
    encoder = JsonEncoder()
    tokens = encoder._tokenise_string("")
    assert tokens == []


# --- Export/Import ---


# Test export writes valid JSON file.
def test_export_writes_json(tmp_path):
    encoder = JsonEncoder()
    data = {"key": "value", "count": 42}
    path = tmp_path / "test.json"
    encoder.export(data, path)
    assert path.exists()
    content = path.read_text()
    assert '"key"' in content
    assert '"value"' in content


# Test import reads JSON file correctly.
def test_import_reads_json(tmp_path):
    encoder = JsonEncoder()
    path = tmp_path / "test.json"
    path.write_text('{"key": "value", "count": 42}')
    data = encoder.import_(path)
    assert data == {"key": "value", "count": 42}


# Test export/import round-trip.
def test_export_import_roundtrip(tmp_path):
    encoder = JsonEncoder()
    original = {
        "messages": [{"role": "user", "content": "Hello"}],
        "temp": 0.7,
    }
    path = tmp_path / "roundtrip.json"
    encoder.export(original, path)
    imported = encoder.import_(path)
    assert imported["messages"] == original["messages"]
    assert imported["temp"] == pytest.approx(original["temp"])


# --- Error handling ---


# Test decode with truncated data raises error.
def test_decode_truncated_data():
    with TokenCache(":memory:") as cache:
        encoder = JsonEncoder()
        # Encode a value then truncate
        blob = encoder.encode(42, cache)
        with pytest.raises((ValueError, struct.error)):
            encoder.decode(blob[:2], cache)


# Test decode with invalid type marker raises error.
def test_decode_invalid_type_marker():
    with TokenCache(":memory:") as cache:
        encoder = JsonEncoder()
        # Invalid type marker 0xFF
        blob = bytes([0xFF, 0x00, 0x00])
        with pytest.raises(ValueError, match="Unknown type marker"):
            encoder.decode(blob, cache)


# Test decode with unknown token ID raises error.
def test_decode_unknown_token_id():
    with TokenCache(":memory:") as cache:
        encoder = JsonEncoder()
        # Create blob with TOKEN_REF pointing to non-existent token ID 999
        blob = bytes([0x00]) + struct.pack("<H", 999)
        with pytest.raises(ValueError, match="Unknown token ID"):
            encoder.decode(blob, cache)


# Test decode with empty blob raises error.
def test_decode_empty_blob():
    with TokenCache(":memory:") as cache:
        encoder = JsonEncoder()
        with pytest.raises(ValueError, match="Unexpected end of data"):
            encoder.decode(b"", cache)


# Test decode with unexpected token at value position.
def test_decode_unexpected_token_at_value():
    with TokenCache(":memory:") as cache:
        encoder = JsonEncoder()
        # Create a token that's not a valid value starter (like ":")
        colon_id = cache.get_or_create_token(":")
        blob = bytes([0x00]) + struct.pack("<H", colon_id)
        with pytest.raises(ValueError, match="Unexpected token at value"):
            encoder.decode(blob, cache)


# Test decode with non-token in string raises error.
def test_decode_non_token_in_string():
    with TokenCache(":memory:") as cache:
        encoder = JsonEncoder()
        # Start a string then put a literal int marker inside
        quote_id = cache.get_or_create_token('"')
        blob = bytes([0x00]) + struct.pack("<H", quote_id)
        blob += bytes([0x01])  # LITERAL_INT inside string
        with pytest.raises(ValueError, match="Expected token in string"):
            encoder.decode(blob, cache)


# Test decode with unterminated string raises error.
def test_decode_unterminated_string():
    with TokenCache(":memory:") as cache:
        encoder = JsonEncoder()
        # Start a string but don't close it
        quote_id = cache.get_or_create_token('"')
        word_id = cache.get_or_create_token("hello")
        blob = bytes([0x00]) + struct.pack("<H", quote_id)
        blob += bytes([0x00]) + struct.pack("<H", word_id)
        # No closing quote
        with pytest.raises(ValueError, match="Unterminated string"):
            encoder.decode(blob, cache)


# Test decode with unterminated list raises error.
def test_decode_unterminated_list():
    with TokenCache(":memory:") as cache:
        encoder = JsonEncoder()
        # Start a list, add an item, but don't close it
        bracket_id = cache.get_or_create_token("[")
        blob = bytes([0x00]) + struct.pack("<H", bracket_id)
        blob += bytes([0x01]) + struct.pack("<q", 42)  # integer item
        # No closing bracket - runs out of data
        with pytest.raises(ValueError, match="Unterminated list"):
            encoder.decode(blob, cache)


# Test decode with non-token after list item raises error.
def test_decode_non_token_after_list_item():
    with TokenCache(":memory:") as cache:
        encoder = JsonEncoder()
        bracket_id = cache.get_or_create_token("[")
        blob = bytes([0x00]) + struct.pack("<H", bracket_id)
        blob += bytes([0x01]) + struct.pack("<q", 42)  # integer item
        blob += bytes([0x01])  # Another LITERAL_INT instead of comma or ]
        with pytest.raises(ValueError, match="Expected token after list"):
            encoder.decode(blob, cache)


# Test decode with invalid token after list item raises error.
def test_decode_invalid_token_in_list():
    with TokenCache(":memory:") as cache:
        encoder = JsonEncoder()
        bracket_id = cache.get_or_create_token("[")
        colon_id = cache.get_or_create_token(":")
        blob = bytes([0x00]) + struct.pack("<H", bracket_id)
        blob += bytes([0x01]) + struct.pack("<q", 42)  # integer item
        blob += bytes([0x00]) + struct.pack(
            "<H", colon_id
        )  # : instead of , or ]
        with pytest.raises(ValueError, match="Expected ',' or ']'"):
            encoder.decode(blob, cache)


# Test decode with unterminated dict raises error.
def test_decode_unterminated_dict():
    with TokenCache(":memory:") as cache:
        encoder = JsonEncoder()
        brace_id = cache.get_or_create_token("{")
        quote_id = cache.get_or_create_token('"')
        key_id = cache.get_or_create_token("key")
        colon_id = cache.get_or_create_token(":")
        blob = bytes([0x00]) + struct.pack("<H", brace_id)
        # Key: "key"
        blob += bytes([0x00]) + struct.pack("<H", quote_id)
        blob += bytes([0x00]) + struct.pack("<H", key_id)
        blob += bytes([0x00]) + struct.pack("<H", quote_id)
        # Colon
        blob += bytes([0x00]) + struct.pack("<H", colon_id)
        # Value
        blob += bytes([0x01]) + struct.pack("<q", 42)
        # No closing brace - runs out of data
        with pytest.raises(ValueError, match="Unterminated dict"):
            encoder.decode(blob, cache)


# Test decode with non-token after dict value raises error.
def test_decode_non_token_after_dict_value():
    with TokenCache(":memory:") as cache:
        encoder = JsonEncoder()
        brace_id = cache.get_or_create_token("{")
        quote_id = cache.get_or_create_token('"')
        key_id = cache.get_or_create_token("key")
        colon_id = cache.get_or_create_token(":")
        blob = bytes([0x00]) + struct.pack("<H", brace_id)
        blob += bytes([0x00]) + struct.pack("<H", quote_id)
        blob += bytes([0x00]) + struct.pack("<H", key_id)
        blob += bytes([0x00]) + struct.pack("<H", quote_id)
        blob += bytes([0x00]) + struct.pack("<H", colon_id)
        blob += bytes([0x01]) + struct.pack("<q", 42)
        blob += bytes([0x01])  # LITERAL_INT instead of , or }
        with pytest.raises(ValueError, match="Expected token after dict"):
            encoder.decode(blob, cache)


# Test decode with invalid token after dict value raises error.
def test_decode_invalid_token_in_dict():
    with TokenCache(":memory:") as cache:
        encoder = JsonEncoder()
        brace_id = cache.get_or_create_token("{")
        quote_id = cache.get_or_create_token('"')
        key_id = cache.get_or_create_token("key")
        colon_id = cache.get_or_create_token(":")
        bracket_id = cache.get_or_create_token("[")
        blob = bytes([0x00]) + struct.pack("<H", brace_id)
        blob += bytes([0x00]) + struct.pack("<H", quote_id)
        blob += bytes([0x00]) + struct.pack("<H", key_id)
        blob += bytes([0x00]) + struct.pack("<H", quote_id)
        blob += bytes([0x00]) + struct.pack("<H", colon_id)
        blob += bytes([0x01]) + struct.pack("<q", 42)
        blob += bytes([0x00]) + struct.pack(
            "<H", bracket_id
        )  # [ instead of , or }
        with pytest.raises(ValueError, match="Expected ',' or '}'"):
            encoder.decode(blob, cache)


# Test decode dict with non-string key raises error.
def test_decode_dict_non_string_key():
    with TokenCache(":memory:") as cache:
        encoder = JsonEncoder()
        brace_id = cache.get_or_create_token("{")
        blob = bytes([0x00]) + struct.pack("<H", brace_id)
        # Integer key instead of string
        blob += bytes([0x01]) + struct.pack("<q", 42)
        with pytest.raises(ValueError, match="Dict key must be string"):
            encoder.decode(blob, cache)


# Test decode dict missing colon raises error.
def test_decode_dict_missing_colon():
    with TokenCache(":memory:") as cache:
        encoder = JsonEncoder()
        brace_id = cache.get_or_create_token("{")
        quote_id = cache.get_or_create_token('"')
        key_id = cache.get_or_create_token("key")
        blob = bytes([0x00]) + struct.pack("<H", brace_id)
        blob += bytes([0x00]) + struct.pack("<H", quote_id)
        blob += bytes([0x00]) + struct.pack("<H", key_id)
        blob += bytes([0x00]) + struct.pack("<H", quote_id)
        # No colon - blob ends
        with pytest.raises(ValueError, match="Expected ':'"):
            encoder.decode(blob, cache)


# Test decode dict wrong token after key raises error.
def test_decode_dict_wrong_token_after_key():
    with TokenCache(":memory:") as cache:
        encoder = JsonEncoder()
        brace_id = cache.get_or_create_token("{")
        quote_id = cache.get_or_create_token('"')
        key_id = cache.get_or_create_token("key")
        comma_id = cache.get_or_create_token(",")
        blob = bytes([0x00]) + struct.pack("<H", brace_id)
        blob += bytes([0x00]) + struct.pack("<H", quote_id)
        blob += bytes([0x00]) + struct.pack("<H", key_id)
        blob += bytes([0x00]) + struct.pack("<H", quote_id)
        blob += bytes([0x00]) + struct.pack("<H", comma_id)  # , instead of :
        with pytest.raises(ValueError, match="Expected ':'"):
            encoder.decode(blob, cache)


# --- Edge cases ---


# Test encoding integer key (converted to string).
def test_dict_with_int_key():
    with TokenCache(":memory:") as cache:
        encoder = JsonEncoder()
        data = {1: "one", 2: "two"}
        blob = encoder.encode(data, cache)
        decoded = encoder.decode(blob, cache)
        # JSON keys are always strings
        assert decoded == {"1": "one", "2": "two"}


# Test encoding non-JSON type falls back to string.
def test_encode_non_json_type():
    with TokenCache(":memory:") as cache:
        encoder = JsonEncoder()

        class Custom:
            def __str__(self):
                return "custom_value"

        blob = encoder.encode(Custom(), cache)
        assert encoder.decode(blob, cache) == "custom_value"


# Test encoding string with only whitespace.
def test_encode_whitespace_string():
    with TokenCache(":memory:") as cache:
        encoder = JsonEncoder()
        blob = encoder.encode("   ", cache)
        assert encoder.decode(blob, cache) == "   "


# Test encoding string with newlines.
def test_encode_string_with_newlines():
    with TokenCache(":memory:") as cache:
        encoder = JsonEncoder()
        text = "line1\nline2\nline3"
        blob = encoder.encode(text, cache)
        assert encoder.decode(blob, cache) == text


# Test single item list.
def test_encode_single_item_list():
    with TokenCache(":memory:") as cache:
        encoder = JsonEncoder()
        blob = encoder.encode([42], cache)
        assert encoder.decode(blob, cache) == [42]
