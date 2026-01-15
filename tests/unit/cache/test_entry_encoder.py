"""
Unit tests for EntryEncoder abstract base class.

Tests verify:
- ABC cannot be instantiated directly
- Concrete implementations must implement all abstract methods
- default_export_format property works
"""

from pathlib import Path
from typing import Any

import pytest

from causaliq_knowledge.cache import TokenCache
from causaliq_knowledge.cache.encoders import EntryEncoder

# ============================================================================
# ABC instantiation tests
# ============================================================================


# Test that EntryEncoder cannot be instantiated directly
def test_entry_encoder_cannot_instantiate() -> None:
    """Verify EntryEncoder ABC cannot be instantiated."""
    with pytest.raises(TypeError, match="abstract"):
        EntryEncoder()  # type: ignore[abstract]


# ============================================================================
# Concrete implementation tests
# ============================================================================


# Helper: minimal concrete encoder for testing
class MinimalEncoder(EntryEncoder):
    """Minimal concrete encoder for testing."""

    def encode(self, data: Any, token_cache: TokenCache) -> bytes:
        """Encode data as UTF-8 string."""
        return str(data).encode("utf-8")

    def decode(self, blob: bytes, token_cache: TokenCache) -> Any:
        """Decode UTF-8 string."""
        return blob.decode("utf-8")

    def export(self, data: Any, path: Path) -> None:
        """Export to text file."""
        path.write_text(str(data))

    def import_(self, path: Path) -> Any:
        """Import from text file."""
        return path.read_text()


# Test that concrete implementation can be instantiated
def test_concrete_encoder_can_instantiate() -> None:
    """Verify concrete encoder can be instantiated."""
    encoder = MinimalEncoder()
    assert encoder is not None


# Test default_export_format property
def test_default_export_format() -> None:
    """Verify default_export_format returns 'json'."""
    encoder = MinimalEncoder()
    assert encoder.default_export_format == "json"


# Test encode method works
def test_encode_method() -> None:
    """Verify encode method works on concrete encoder."""
    encoder = MinimalEncoder()
    with TokenCache(":memory:") as cache:
        result = encoder.encode("hello", cache)
        assert result == b"hello"


# Test decode method works
def test_decode_method() -> None:
    """Verify decode method works on concrete encoder."""
    encoder = MinimalEncoder()
    with TokenCache(":memory:") as cache:
        result = encoder.decode(b"world", cache)
        assert result == "world"


# Test encode/decode roundtrip
def test_encode_decode_roundtrip() -> None:
    """Verify encode/decode roundtrip preserves data."""
    encoder = MinimalEncoder()
    with TokenCache(":memory:") as cache:
        original = "test data 123"
        encoded = encoder.encode(original, cache)
        decoded = encoder.decode(encoded, cache)
        assert decoded == original
