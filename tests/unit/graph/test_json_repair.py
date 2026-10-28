"""Tests for the JSON repair helpers used by response parsing."""

from __future__ import annotations

import pytest

from causaliq_knowledge.graph.response import (
    _sanitise_json_text,
    _try_parse_json,
    parse_graph_response,
    parse_pdg_response,
)

# --- _sanitise_json_text tests ---


# Test sanitise with escaped quotes preserves backslash counting.
def test_sanitise_json_text_escaped_quotes() -> None:
    """Test that escaped quotes are preserved (backslash counting)."""
    # A string with escaped quote inside: {"key": "value \"quoted\" end"}
    text = '{"key": "value \\"quoted\\" end"}'
    result = _sanitise_json_text(text)
    # Escaped quotes should remain unchanged
    assert result == text


# Test sanitise with multiple backslashes before quote.
def test_sanitise_json_text_multiple_backslashes() -> None:
    """Test backslash counting with multiple backslashes."""
    # Two backslashes followed by quote = unescaped quote (even count)
    # The string: {"key": "path\\"}
    text = '{"key": "path\\\\"}'
    result = _sanitise_json_text(text)
    assert result == text


# Test sanitise with carriage return control character.
def test_sanitise_json_text_carriage_return() -> None:
    """Test that carriage return inside string is escaped."""
    # JSON with actual \r (carriage return) inside string
    text = '{"key": "line1\rline2"}'
    result = _sanitise_json_text(text)
    # \r should be escaped to \\r
    assert result == '{"key": "line1\\rline2"}'


# Test sanitise with tab control character.
def test_sanitise_json_text_tab() -> None:
    """Test that tab inside string is escaped."""
    # JSON with actual \t (tab) inside string
    text = '{"key": "col1\tcol2"}'
    result = _sanitise_json_text(text)
    # \t should be escaped to \\t
    assert result == '{"key": "col1\\tcol2"}'


# Test sanitise with other control characters uses unicode escape.
def test_sanitise_json_text_unicode_escape() -> None:
    """Test that other control chars use unicode escape."""
    # JSON with form feed (0x0C) inside string
    text = '{"key": "before\x0cafter"}'
    result = _sanitise_json_text(text)
    # Form feed should be escaped as \u000c
    assert result == '{"key": "before\\u000cafter"}'


# Test sanitise preserves newlines outside strings.
def test_sanitise_json_text_newline_outside_string() -> None:
    """Test that newlines outside strings are preserved."""
    text = '{\n    "key": "value"\n}'
    result = _sanitise_json_text(text)
    # Newlines outside strings should remain unchanged
    assert result == text


# Test parse_graph_response with truncated JSON reports EOF.
def test_parse_graph_response_truncated_json() -> None:
    """Test that truncated JSON triggers parse error."""
    # Truncated JSON - ends expecting more content
    response_text = '{"edges": ['
    with pytest.raises(ValueError, match="Failed to parse JSON"):
        parse_graph_response(response_text, ["a", "b"])


# Test parse_graph_response with minimal truncation triggers EOF diagnostic.
def test_parse_graph_response_eof_position() -> None:
    """Test truncated JSON where error position is at EOF."""
    # Single opening brace - error position will be at len(text)
    response_text = "{"
    with pytest.raises(ValueError, match="Failed to parse JSON"):
        parse_graph_response(response_text, ["a", "b"])


# Test parse_pdg_response with truncated JSON reports EOF.
def test_parse_pdg_response_truncated_json() -> None:
    """Test that truncated JSON triggers parse error."""
    # Truncated JSON - ends expecting more content
    response_text = '{"edges": ['
    with pytest.raises(ValueError, match="Failed to parse JSON"):
        parse_pdg_response(response_text, ["a", "b"])


# Test parse_pdg_response with minimal truncation triggers EOF diagnostic.
def test_parse_pdg_response_eof_position() -> None:
    """Test truncated JSON where error position is at EOF."""
    # Single opening brace - error position will be at len(text)
    response_text = "{"
    with pytest.raises(ValueError, match="Failed to parse JSON"):
        parse_pdg_response(response_text, ["a", "b"])


# --- _try_parse_json sanitise fallback tests ---


# Test _try_parse_json falls through when raw_decode also fails.
def test_try_parse_json_raw_decode_fallthrough(monkeypatch) -> None:
    """Test Extra data path falls through to sanitise when raw_decode fails."""
    import json as _json

    original_raw_decode = _json.JSONDecoder.raw_decode
    call_count = 0

    def _failing_on_second(self, s, idx=0):
        nonlocal call_count
        call_count += 1
        # Let the first call (inside json.loads) succeed normally
        # so json.loads raises "Extra data". Fail on explicit call.
        if call_count <= 1:
            return original_raw_decode(self, s, idx)
        raise _json.JSONDecodeError("Unexpected", s, 0)

    monkeypatch.setattr(_json.JSONDecoder, "raw_decode", _failing_on_second)

    # Text with trailing garbage triggers "Extra data"
    text = '{"a": 1}  extra'
    obj, err = _try_parse_json(text)
    # The explicit raw_decode call should fail, falling through
    # to the sanitise path. Sanitise won't help either, so we
    # expect the error diagnostic.
    assert obj is None
    assert err is not None


# Test _try_parse_json recovers via sanitise for control chars.
def test_try_parse_json_sanitise_fallback() -> None:
    """Test sanitise fallback when control chars break parsing."""
    # JSON with a raw tab inside a string value — invalid JSON
    text = '{"key": "val\tue"}'
    obj, err = _try_parse_json(text)
    assert err is None
    assert obj == {"key": "val\tue"}


# Test _try_parse_json recovers via sanitise + raw_decode.
def test_try_parse_json_sanitise_raw_decode_fallback() -> None:
    """Test sanitise then raw_decode for control chars + trailing garbage."""
    # JSON with raw tab inside string AND trailing extra brace
    text = '{"key": "val\tue"}}'
    obj, err = _try_parse_json(text)
    assert err is None
    assert obj == {"key": "val\tue"}


# --- _try_parse_json trailing garbage recovery tests ---


# Test parse_pdg_response recovers from trailing extra braces.
def test_parse_pdg_response_trailing_brace() -> None:
    """Test that trailing garbage (extra braces) is handled gracefully."""
    # JSON with extra closing brace - simulates LLM output error
    response_text = (
        '{"edges":[{"source":"a","target":"b",'
        '"existence":0.9,"orientation":0.8}]}}'
    )
    pdg = parse_pdg_response(response_text, ["a", "b"])
    assert ("a", "b") in pdg.edges


# Test parse_graph_response recovers from trailing extra braces.
def test_parse_graph_response_trailing_brace() -> None:
    """Test that trailing garbage (extra braces) is handled gracefully."""
    # JSON with extra closing brace - simulates LLM output error
    response_text = (
        '{"edges":[{"source":"a","target":"b","confidence":0.9}],'
        '"reasoning":"test"}}'
    )
    result = parse_graph_response(response_text, ["a", "b"])
    assert len(result.edges) == 1
    assert result.edges[0].source == "a"
