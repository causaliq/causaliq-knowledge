"""JSON repair helpers for LLM graph generation responses.

LLMs occasionally append extra characters after a valid JSON document, or
emit unescaped control characters inside string values. This module
extracts the valid JSON portion and builds a diagnostic message when every
recovery attempt fails.
"""

from __future__ import annotations

import json
import logging
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Sentinel returned by the recovery helpers when text cannot be recovered
_UNRECOVERED = object()

# Control characters that have a short JSON escape sequence
_CONTROL_ESCAPES = {"\n": "\\n", "\r": "\\r", "\t": "\\t"}


def _is_escaped(text: str, pos: int) -> bool:
    """Check if char at pos is escaped by counting backslashes."""
    num_backslashes = 0
    j = pos - 1
    while j >= 0 and text[j] == "\\":
        num_backslashes += 1
        j -= 1
    return num_backslashes % 2 == 1


def _escape_control_char(char: str) -> str:
    """Return the JSON escape sequence for a control character."""
    if char in _CONTROL_ESCAPES:
        return _CONTROL_ESCAPES[char]
    return f"\\u{ord(char):04x}"


def _append_sanitised_char(
    result: List[str],
    text: str,
    index: int,
    char: str,
    in_string: bool,
) -> bool:
    """Append one sanitised character and return the new string state."""
    if char == '"' and not _is_escaped(text, index):
        # Toggle string state on unescaped quote
        result.append(char)
        return not in_string
    if in_string and ord(char) < 0x20:
        # Control character inside string - must escape
        result.append(_escape_control_char(char))
        return in_string
    result.append(char)
    return in_string


def _sanitise_json_text(text: str) -> str:
    """Sanitise JSON text by escaping control characters inside strings.

    LLMs sometimes include unescaped control characters (especially
    newlines) inside JSON string values. This function escapes them
    while preserving valid JSON structure.

    Args:
        text: Raw JSON text potentially containing invalid characters.

    Returns:
        Sanitised JSON text with control characters inside strings escaped.
    """
    result: List[str] = []
    in_string = False
    i = 0
    while i < len(text):
        in_string = _append_sanitised_char(result, text, i, text[i], in_string)
        i += 1

    return "".join(result)


def _recover_extra_data(text: str) -> Any:
    """Recover the valid JSON prefix of text with trailing garbage."""
    try:
        obj, _ = json.JSONDecoder().raw_decode(text)
    except json.JSONDecodeError:
        return _UNRECOVERED
    logger.warning("Recovered from trailing garbage in JSON response")
    return obj


def _recover_sanitised(text: str) -> Any:
    """Parse text after escaping control characters inside strings."""
    try:
        return json.loads(_sanitise_json_text(text))
    except json.JSONDecodeError:
        return _UNRECOVERED


def _recover_sanitised_extra_data(text: str) -> Any:
    """Recover the valid sanitised JSON prefix of text."""
    try:
        obj, _ = json.JSONDecoder().raw_decode(_sanitise_json_text(text))
    except json.JSONDecodeError:
        return _UNRECOVERED
    logger.warning(
        "Recovered from trailing garbage in sanitised JSON response"
    )
    return obj


def _recover_json(error: json.JSONDecodeError, text: str) -> Any:
    """Try every recovery strategy and return the parsed value."""
    # Check if it's an "Extra data" error - valid JSON followed by junk
    if "Extra data" in str(error):
        recovered = _recover_extra_data(text)
        if recovered is not _UNRECOVERED:
            return recovered

    # Try sanitising control characters
    recovered = _recover_sanitised(text)
    if recovered is not _UNRECOVERED:
        return recovered

    # Try raw_decode on sanitised text for Extra data errors
    recovered = _recover_sanitised_extra_data(text)
    if recovered is not _UNRECOVERED:
        return recovered

    return _UNRECOVERED


def _describe_char_at(text: str, pos: int) -> str:
    """Describe the character at a position, or report end of file."""
    if pos < len(text):
        char_at_pos = text[pos]
        return f"Char: {repr(char_at_pos)} (0x{ord(char_at_pos):02x})"
    return "Char: EOF"


def _format_parse_error(error: json.JSONDecodeError, text: str) -> str:
    """Build a diagnostic message for an unrecoverable JSON error."""
    pos = error.pos if hasattr(error, "pos") else 0
    context_start = max(0, pos - 50)
    context_end = min(len(text), pos + 50)
    context = text[context_start:context_end]
    char_info = _describe_char_at(text, pos)
    error_msg = (
        f"JSON parse error at position {pos}. "
        f"{char_info}. "
        f"Context: {repr(context)}"
    )
    return f"{error_msg}\nOriginal error: {error}"


def _recover_from_error(
    error: json.JSONDecodeError, text: str
) -> Tuple[Any, Optional[str]]:
    """Return a recovered value or a diagnostic error message."""
    recovered = _recover_json(error, text)
    if recovered is _UNRECOVERED:
        return None, _format_parse_error(error, text)
    return recovered, None


def _try_parse_json(text: str) -> Tuple[Any, Optional[str]]:
    """Attempt to parse JSON with fallback for trailing garbage.

    LLMs sometimes append extra characters after valid JSON (e.g., extra
    closing braces). This function uses raw_decode to extract just the
    valid JSON portion.

    Args:
        text: JSON text to parse.

    Returns:
        Tuple of (parsed_object, error_message). error_message is None
        on success.
    """
    # First try normal parsing
    try:
        return json.loads(text), None
    except json.JSONDecodeError as error:
        return _recover_from_error(error, text)
