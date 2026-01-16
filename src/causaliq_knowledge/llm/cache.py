"""
LLM-specific cache encoder and data structures.

This module provides the LLMEntryEncoder for caching LLM requests and
responses with rich metadata for analysis.

Note: This module stays in causaliq-knowledge (LLM-specific).
The base cache infrastructure will migrate to causaliq-core.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from causaliq_knowledge.cache.encoders import JsonEncoder

if TYPE_CHECKING:  # pragma: no cover
    from causaliq_knowledge.cache.token_cache import TokenCache


@dataclass
class LLMTokenUsage:
    """Token usage statistics for an LLM request.

    Attributes:
        input: Number of tokens in the prompt.
        output: Number of tokens in the completion.
        total: Total tokens (input + output).
    """

    input: int = 0
    output: int = 0
    total: int = 0


@dataclass
class LLMMetadata:
    """Metadata for a cached LLM response.

    Attributes:
        provider: LLM provider name (openai, anthropic, etc.).
        timestamp: When the original request was made (ISO format).
        latency_ms: Response time in milliseconds.
        tokens: Token usage statistics.
        cost_usd: Estimated cost of the request in USD.
        cache_hit: Whether this was served from cache.
    """

    provider: str = ""
    timestamp: str = ""
    latency_ms: int = 0
    tokens: LLMTokenUsage = field(default_factory=LLMTokenUsage)
    cost_usd: float = 0.0
    cache_hit: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialisation."""
        return {
            "provider": self.provider,
            "timestamp": self.timestamp,
            "latency_ms": self.latency_ms,
            "tokens": asdict(self.tokens),
            "cost_usd": self.cost_usd,
            "cache_hit": self.cache_hit,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LLMMetadata:
        """Create from dictionary."""
        tokens_data = data.get("tokens", {})
        return cls(
            provider=data.get("provider", ""),
            timestamp=data.get("timestamp", ""),
            latency_ms=data.get("latency_ms", 0),
            tokens=LLMTokenUsage(
                input=tokens_data.get("input", 0),
                output=tokens_data.get("output", 0),
                total=tokens_data.get("total", 0),
            ),
            cost_usd=data.get("cost_usd", 0.0),
            cache_hit=data.get("cache_hit", False),
        )


@dataclass
class LLMResponse:
    """LLM response data for caching.

    Attributes:
        content: The full text response from the LLM.
        finish_reason: Why generation stopped (stop, length, etc.).
        model_version: Actual model version used.
    """

    content: str = ""
    finish_reason: str = "stop"
    model_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialisation."""
        return {
            "content": self.content,
            "finish_reason": self.finish_reason,
            "model_version": self.model_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LLMResponse:
        """Create from dictionary."""
        return cls(
            content=data.get("content", ""),
            finish_reason=data.get("finish_reason", "stop"),
            model_version=data.get("model_version", ""),
        )


@dataclass
class LLMCacheEntry:
    """Complete LLM cache entry with request, response, and metadata.

    Attributes:
        model: The model name requested.
        messages: The conversation messages.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in response.
        response: The LLM response data.
        metadata: Rich metadata for analysis.
    """

    model: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)
    temperature: float = 0.0
    max_tokens: int | None = None
    response: LLMResponse = field(default_factory=LLMResponse)
    metadata: LLMMetadata = field(default_factory=LLMMetadata)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialisation."""
        return {
            "cache_key": {
                "model": self.model,
                "messages": self.messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
            "response": self.response.to_dict(),
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LLMCacheEntry:
        """Create from dictionary."""
        cache_key = data.get("cache_key", {})
        return cls(
            model=cache_key.get("model", ""),
            messages=cache_key.get("messages", []),
            temperature=cache_key.get("temperature", 0.0),
            max_tokens=cache_key.get("max_tokens"),
            response=LLMResponse.from_dict(data.get("response", {})),
            metadata=LLMMetadata.from_dict(data.get("metadata", {})),
        )

    @classmethod
    def create(
        cls,
        model: str,
        messages: list[dict[str, Any]],
        content: str,
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        finish_reason: str = "stop",
        model_version: str = "",
        provider: str = "",
        latency_ms: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> LLMCacheEntry:
        """Create a cache entry with common parameters.

        Args:
            model: The model name requested.
            messages: The conversation messages.
            content: The response content.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
            finish_reason: Why generation stopped.
            model_version: Actual model version.
            provider: LLM provider name.
            latency_ms: Response time in milliseconds.
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            cost_usd: Estimated cost in USD.

        Returns:
            Configured LLMCacheEntry.
        """
        return cls(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response=LLMResponse(
                content=content,
                finish_reason=finish_reason,
                model_version=model_version or model,
            ),
            metadata=LLMMetadata(
                provider=provider,
                timestamp=datetime.now(timezone.utc).isoformat(),
                latency_ms=latency_ms,
                tokens=LLMTokenUsage(
                    input=input_tokens,
                    output=output_tokens,
                    total=input_tokens + output_tokens,
                ),
                cost_usd=cost_usd,
                cache_hit=False,
            ),
        )


class LLMEntryEncoder(JsonEncoder):
    """Encoder for LLM cache entries.

    Extends JsonEncoder with LLM-specific convenience methods for
    encoding/decoding LLMCacheEntry objects.

    The encoder stores data in the standard JSON tokenised format,
    achieving 50-70% compression through the shared token dictionary.

    Example:
        >>> from causaliq_knowledge.cache import TokenCache
        >>> from causaliq_knowledge.llm.cache import (
        ...     LLMEntryEncoder, LLMCacheEntry,
        ... )
        >>> with TokenCache(":memory:") as cache:
        ...     encoder = LLMEntryEncoder()
        ...     entry = LLMCacheEntry.create(
        ...         model="gpt-4",
        ...         messages=[{"role": "user", "content": "Hello"}],
        ...         content="Hi there!",
        ...         provider="openai",
        ...     )
        ...     blob = encoder.encode(entry.to_dict(), cache)
        ...     data = encoder.decode(blob, cache)
        ...     restored = LLMCacheEntry.from_dict(data)
    """

    def encode_entry(self, entry: LLMCacheEntry, cache: TokenCache) -> bytes:
        """Encode an LLMCacheEntry to bytes.

        Convenience method that handles to_dict conversion.

        Args:
            entry: The cache entry to encode.
            cache: TokenCache for token dictionary.

        Returns:
            Encoded bytes.
        """
        return self.encode(entry.to_dict(), cache)

    def decode_entry(self, blob: bytes, cache: TokenCache) -> LLMCacheEntry:
        """Decode bytes to an LLMCacheEntry.

        Convenience method that handles from_dict conversion.

        Args:
            blob: Encoded bytes.
            cache: TokenCache for token dictionary.

        Returns:
            Decoded LLMCacheEntry.
        """
        data = self.decode(blob, cache)
        return LLMCacheEntry.from_dict(data)

    def generate_export_filename(
        self, entry: LLMCacheEntry, cache_key: str
    ) -> str:
        """Generate a human-readable filename for export.

        Creates a filename from model name and query details, with a
        short hash suffix for uniqueness.

        For edge queries, extracts node names for format:
            {model}_{node_a}_{node_b}_edge_{hash}.json

        For other queries, uses prompt excerpt:
            {model}_{prompt_excerpt}_{hash}.json

        Args:
            entry: The cache entry to generate filename for.
            cache_key: The cache key (hash) for uniqueness suffix.

        Returns:
            Human-readable filename with .json extension.

        Example:
            >>> encoder = LLMEntryEncoder()
            >>> entry = LLMCacheEntry.create(
            ...     model="gpt-4",
            ...     messages=[{"role": "user", "content": "smoking and lung"}],
            ...     content="Yes...",
            ... )
            >>> encoder.generate_export_filename(entry, "a1b2c3d4e5f6")
            'gpt4_smoking_lung_edge_a1b2.json'
        """
        import re

        # Sanitize model name (alphanumeric only, lowercase)
        model = re.sub(r"[^a-z0-9]", "", entry.model.lower())
        if len(model) > 15:
            model = model[:15]

        # Extract user message content
        prompt = ""
        for msg in entry.messages:
            if msg.get("role") == "user":
                prompt = msg.get("content", "")
                break

        # Try to extract node names for edge queries
        # Look for patterns like "X and Y", "X cause Y", "between X and Y"
        prompt_lower = prompt.lower()
        slug = ""

        # Pattern: "between X and Y" or "X and Y"
        match = re.search(r"(?:between\s+)?(\w+)\s+and\s+(\w+)", prompt_lower)
        if match:
            node_a = match.group(1)[:15]
            node_b = match.group(2)[:15]
            slug = f"{node_a}_{node_b}_edge"

        # Fallback: extract first significant words from prompt
        if not slug:
            # Remove common words, keep alphanumeric
            cleaned = re.sub(r"[^a-z0-9\s]", "", prompt_lower)
            words = [
                w
                for w in cleaned.split()
                if w
                not in ("the", "a", "an", "is", "are", "does", "do", "can")
            ]
            slug = "_".join(words[:4])
            if len(slug) > 30:
                slug = slug[:30].rstrip("_")

        # Short hash suffix for uniqueness (4 chars)
        hash_suffix = cache_key[:4] if cache_key else "0000"

        # Build filename
        parts = [p for p in [model, slug, hash_suffix] if p]
        return "_".join(parts) + ".json"

    def export_entry(self, entry: LLMCacheEntry, path: Path) -> None:
        """Export an LLMCacheEntry to a JSON file.

        Args:
            entry: The cache entry to export.
            path: Destination file path.
        """
        self.export(entry.to_dict(), path)

    def import_entry(self, path: Path) -> LLMCacheEntry:
        """Import an LLMCacheEntry from a JSON file.

        Args:
            path: Source file path.

        Returns:
            Imported LLMCacheEntry.
        """
        data = self.import_(path)
        return LLMCacheEntry.from_dict(data)
