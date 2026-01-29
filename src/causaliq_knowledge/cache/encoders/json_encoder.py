"""
Generic JSON encoder with tokenisation and literal handling.

Tokenises JSON structure (keys, structural chars, string values) while
storing numbers as compact binary literals. Achieves 50-70% compression
on typical JSON data.

Note: This module is designed for future migration to causaliq-core.
"""

from __future__ import annotations

import json
import re
import struct
from pathlib import Path
from typing import TYPE_CHECKING, Any

from causaliq_knowledge.cache.encoders.base import EntryEncoder

if TYPE_CHECKING:  # pragma: no cover
    from causaliq_knowledge.cache.token_cache import TokenCache


# Type markers for encoded values
TOKEN_REF = 0x00
LITERAL_INT = 0x01
LITERAL_FLOAT = 0x02


class JsonEncoder(EntryEncoder):
    """Tokenised encoding for JSON-serialisable data.

    Uses shared token dictionary for JSON structure and text content.
    Numbers are stored as binary literals. Typical compression is 50-70%.

    Encoding format:
        - Token reference: 0x00 + uint16 (token ID)
        - Integer literal: 0x01 + int64 (8 bytes, signed)
        - Float literal: 0x02 + float64 (8 bytes, double)

    Example:
        >>> from causaliq_knowledge.cache import TokenCache
        >>> with TokenCache(":memory:") as cache:
        ...     encoder = JsonEncoder()
        ...     data = {"key": "value", "count": 42}
        ...     blob = encoder.encode(data, cache)
        ...     decoded = encoder.decode(blob, cache)
        ...     assert decoded == data
    """

    def _get_token(self, token_id: int, token_cache: TokenCache) -> str:
        """Get token by ID, raising error if not found.

        Args:
            token_id: The token ID to look up.
            token_cache: Cache instance for token dictionary.

        Returns:
            The token string.

        Raises:
            ValueError: If token ID not found (corrupted cache).
        """
        token = token_cache.get_token(token_id)
        if token is None:
            raise ValueError(f"Unknown token ID: {token_id}")
        return token

    @property
    def default_export_format(self) -> str:
        """Default file extension for exports."""
        return "json"

    def encode(self, data: Any, token_cache: TokenCache) -> bytes:
        """Encode JSON-serialisable data to tokenised binary format.

        Args:
            data: Any JSON-serialisable data (dict, list, str, int, etc.).
            token_cache: Cache instance for shared token dictionary.

        Returns:
            Compact binary representation using token IDs and literals.
        """
        result = bytearray()
        self._encode_value(data, token_cache, result)
        return bytes(result)

    def decode(self, blob: bytes, token_cache: TokenCache) -> Any:
        """Decode tokenised binary data back to JSON structure.

        Args:
            blob: Binary data from cache.
            token_cache: Cache instance for shared token dictionary.

        Returns:
            Decoded JSON-compatible data structure.
        """
        offset = 0
        value, _ = self._decode_value(blob, offset, token_cache)
        return value

    def export(self, data: Any, path: Path) -> None:
        """Export data to JSON file.

        Args:
            data: The decoded data to export.
            path: Destination file path.
        """
        path.write_text(json.dumps(data, indent=2))

    def import_(self, path: Path) -> Any:
        """Import data from JSON file.

        Args:
            path: Source file path.

        Returns:
            Imported JSON data ready for encoding.
        """
        return json.loads(path.read_text())

    def _encode_value(
        self, value: Any, token_cache: TokenCache, result: bytearray
    ) -> None:
        """Recursively encode a JSON value.

        Args:
            value: Value to encode.
            token_cache: Cache for token dictionary.
            result: Bytearray to append encoded data to.
        """
        if value is None:
            self._encode_token("null", token_cache, result)
        elif isinstance(value, bool):
            # Must check bool before int (bool is subclass of int)
            self._encode_token(
                "true" if value else "false", token_cache, result
            )
        elif isinstance(value, int):
            result.append(LITERAL_INT)
            result.extend(struct.pack("<q", value))
        elif isinstance(value, float):
            result.append(LITERAL_FLOAT)
            result.extend(struct.pack("<d", value))
        elif isinstance(value, str):
            self._encode_string(value, token_cache, result)
        elif isinstance(value, list):
            self._encode_list(value, token_cache, result)
        elif isinstance(value, dict):
            self._encode_dict(value, token_cache, result)
        else:
            # Fallback: convert to string
            self._encode_string(str(value), token_cache, result)

    def _encode_token(
        self, token: str, token_cache: TokenCache, result: bytearray
    ) -> None:
        """Encode a single token reference.

        Args:
            token: Token string to encode.
            token_cache: Cache for token dictionary.
            result: Bytearray to append encoded data to.
        """
        token_id = token_cache.get_or_create_token(token)
        result.append(TOKEN_REF)
        result.extend(struct.pack("<H", token_id))

    def _encode_string(
        self, value: str, token_cache: TokenCache, result: bytearray
    ) -> None:
        """Encode a string value with tokenisation.

        Strings are split into tokens (words/punctuation) with special
        markers for string start/end. Double quotes within the string
        are encoded as '\\"' token to distinguish from string delimiters.

        Args:
            value: String to encode.
            token_cache: Cache for token dictionary.
            result: Bytearray to append encoded data to.
        """
        self._encode_token('"', token_cache, result)
        # Split on whitespace and punctuation, keeping delimiters
        tokens = self._tokenise_string(value)
        for token in tokens:
            # Escape embedded quotes to distinguish from string delimiter
            if token == '"':
                self._encode_token('\\"', token_cache, result)
            else:
                self._encode_token(token, token_cache, result)
        self._encode_token('"', token_cache, result)

    def _encode_list(
        self, value: list, token_cache: TokenCache, result: bytearray
    ) -> None:
        """Encode a list value.

        Args:
            value: List to encode.
            token_cache: Cache for token dictionary.
            result: Bytearray to append encoded data to.
        """
        self._encode_token("[", token_cache, result)
        for i, item in enumerate(value):
            if i > 0:
                self._encode_token(",", token_cache, result)
            self._encode_value(item, token_cache, result)
        self._encode_token("]", token_cache, result)

    def _encode_dict(
        self, value: dict, token_cache: TokenCache, result: bytearray
    ) -> None:
        """Encode a dict value.

        Args:
            value: Dict to encode.
            token_cache: Cache for token dictionary.
            result: Bytearray to append encoded data to.
        """
        self._encode_token("{", token_cache, result)
        for i, (key, val) in enumerate(value.items()):
            if i > 0:
                self._encode_token(",", token_cache, result)
            self._encode_string(str(key), token_cache, result)
            self._encode_token(":", token_cache, result)
            self._encode_value(val, token_cache, result)
        self._encode_token("}", token_cache, result)

    def _tokenise_string(self, value: str) -> list[str]:
        """Split string into tokens for encoding.

        Splits on whitespace and punctuation boundaries, preserving
        all characters. Empty string returns empty list.

        Args:
            value: String to tokenise.

        Returns:
            List of token strings.
        """
        if not value:
            return []
        # Split on word boundaries, keeping all parts
        # Matches: word chars, whitespace runs, or single punctuation
        tokens = re.findall(r"\w+|\s+|[^\w\s]", value)
        return tokens

    def _decode_value(
        self, blob: bytes, offset: int, token_cache: TokenCache
    ) -> tuple[Any, int]:
        """Decode a single value from blob at offset.

        Args:
            blob: Binary data to decode.
            offset: Current position in blob.
            token_cache: Cache for token dictionary.

        Returns:
            Tuple of (decoded value, new offset).
        """
        if offset >= len(blob):
            raise ValueError("Unexpected end of data")

        type_marker = blob[offset]
        offset += 1

        if type_marker == LITERAL_INT:
            value = struct.unpack("<q", blob[offset : offset + 8])[0]
            return value, offset + 8
        elif type_marker == LITERAL_FLOAT:
            value = struct.unpack("<d", blob[offset : offset + 8])[0]
            return value, offset + 8
        elif type_marker == TOKEN_REF:
            token_id = struct.unpack("<H", blob[offset : offset + 2])[0]
            offset += 2
            token = self._get_token(token_id, token_cache)

            if token == "null":
                return None, offset
            elif token == "true":
                return True, offset
            elif token == "false":
                return False, offset
            elif token == '"':
                return self._decode_string(blob, offset, token_cache)
            elif token == "[":
                return self._decode_list(blob, offset, token_cache)
            elif token == "{":
                return self._decode_dict(blob, offset, token_cache)
            else:
                raise ValueError(
                    f"Unexpected token at value position: {token}"
                )
        else:
            raise ValueError(f"Unknown type marker: {type_marker}")

    def _decode_string(
        self, blob: bytes, offset: int, token_cache: TokenCache
    ) -> tuple[str, int]:
        """Decode a string value (after opening quote consumed).

        Handles escaped quotes ('\\"' token) which represent literal
        double quotes within the string content.

        Args:
            blob: Binary data to decode.
            offset: Current position (after opening quote).
            token_cache: Cache for token dictionary.

        Returns:
            Tuple of (decoded string, new offset).
        """
        parts: list[str] = []
        while offset < len(blob):
            type_marker = blob[offset]
            if type_marker != TOKEN_REF:
                raise ValueError(
                    f"Expected token in string, got {type_marker}"
                )
            token_id = struct.unpack("<H", blob[offset + 1 : offset + 3])[0]
            offset += 3
            token = self._get_token(token_id, token_cache)
            if token == '"':
                # End of string
                return "".join(parts), offset
            elif token == '\\"':
                # Escaped quote - append literal quote character
                parts.append('"')
            else:
                parts.append(token)
        raise ValueError("Unterminated string")

    def _decode_list(
        self, blob: bytes, offset: int, token_cache: TokenCache
    ) -> tuple[list, int]:
        """Decode a list value (after opening bracket consumed).

        Args:
            blob: Binary data to decode.
            offset: Current position (after opening bracket).
            token_cache: Cache for token dictionary.

        Returns:
            Tuple of (decoded list, new offset).
        """
        items = []
        # Check for empty list
        if offset < len(blob) and blob[offset] == TOKEN_REF:
            token_id = struct.unpack("<H", blob[offset + 1 : offset + 3])[0]
            token = self._get_token(token_id, token_cache)
            if token == "]":
                return [], offset + 3

        while offset < len(blob):
            value, offset = self._decode_value(blob, offset, token_cache)
            items.append(value)

            # Check for comma or closing bracket
            if offset >= len(blob):
                raise ValueError("Unterminated list")
            if blob[offset] != TOKEN_REF:
                raise ValueError("Expected token after list item")
            token_id = struct.unpack("<H", blob[offset + 1 : offset + 3])[0]
            offset += 3
            token = self._get_token(token_id, token_cache)
            if token == "]":
                return items, offset
            elif token != ",":
                raise ValueError(f"Expected ',' or ']' in list, got '{token}'")

        raise ValueError("Unterminated list")  # pragma: no cover

    def _decode_dict(
        self, blob: bytes, offset: int, token_cache: TokenCache
    ) -> tuple[dict, int]:
        """Decode a dict value (after opening brace consumed).

        Args:
            blob: Binary data to decode.
            offset: Current position (after opening brace).
            token_cache: Cache for token dictionary.

        Returns:
            Tuple of (decoded dict, new offset).
        """
        result = {}
        # Check for empty dict
        if offset < len(blob) and blob[offset] == TOKEN_REF:
            token_id = struct.unpack("<H", blob[offset + 1 : offset + 3])[0]
            token = self._get_token(token_id, token_cache)
            if token == "}":
                return {}, offset + 3

        while offset < len(blob):
            # Decode key (must be string)
            key, offset = self._decode_value(blob, offset, token_cache)
            if not isinstance(key, str):
                raise ValueError(f"Dict key must be string, got {type(key)}")

            # Expect colon
            if offset >= len(blob) or blob[offset] != TOKEN_REF:
                raise ValueError("Expected ':' after dict key")
            token_id = struct.unpack("<H", blob[offset + 1 : offset + 3])[0]
            offset += 3
            token = self._get_token(token_id, token_cache)
            if token != ":":
                raise ValueError(f"Expected ':', got '{token}'")

            # Decode value
            value, offset = self._decode_value(blob, offset, token_cache)
            result[key] = value

            # Check for comma or closing brace
            if offset >= len(blob):
                raise ValueError("Unterminated dict")
            if blob[offset] != TOKEN_REF:
                raise ValueError("Expected token after dict value")
            token_id = struct.unpack("<H", blob[offset + 1 : offset + 3])[0]
            offset += 3
            token = self._get_token(token_id, token_cache)
            if token == "}":
                return result, offset
            elif token != ",":
                raise ValueError(
                    f"Expected ',' or '}}' in dict, got '{token}'"
                )

        raise ValueError("Unterminated dict")  # pragma: no cover
