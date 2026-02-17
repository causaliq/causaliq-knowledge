"""
LLM-specific cache compressor and data structures.

This module provides the LLMCompressor for caching LLM requests and
responses with rich metadata for analysis.

Note: This module stays in causaliq-knowledge (LLM-specific).
The base cache infrastructure will migrate to causaliq-core.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from causaliq_core.cache.compressors import JsonCompressor

if TYPE_CHECKING:  # pragma: no cover
    from causaliq_core.cache.token_cache import TokenCache


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
        request_id: Optional identifier for the request (not in cache key).
    """

    provider: str = ""
    timestamp: str = ""
    latency_ms: int = 0
    tokens: LLMTokenUsage = field(default_factory=LLMTokenUsage)
    cost_usd: float = 0.0
    cache_hit: bool = False
    request_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialisation."""
        return {
            "provider": self.provider,
            "timestamp": self.timestamp,
            "latency_ms": self.latency_ms,
            "tokens": asdict(self.tokens),
            "cost_usd": self.cost_usd,
            "cache_hit": self.cache_hit,
            "request_id": self.request_id,
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
            request_id=data.get("request_id", ""),
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

    def to_export_dict(self) -> dict[str, Any]:
        """Convert to dictionary for export, parsing JSON content if valid.

        Unlike to_dict(), this attempts to parse the content as JSON
        for more readable exported files.
        """
        # Try to parse content as JSON for cleaner export
        try:
            parsed_content = json.loads(self.content)
        except (json.JSONDecodeError, TypeError):
            parsed_content = self.content

        return {
            "content": parsed_content,
            "finish_reason": self.finish_reason,
            "model_version": self.model_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LLMResponse:
        """Create from dictionary."""
        content = data.get("content", "")
        # Handle both string and parsed JSON content (from export files)
        if isinstance(content, dict):
            content = json.dumps(content)
        return cls(
            content=content,
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

    @staticmethod
    def _split_message_content(messages: list[dict[str, Any]]) -> list[Any]:
        """Convert message content with newlines into arrays of lines."""
        result = []
        for msg in messages:
            new_msg = dict(msg)
            content = new_msg.get("content", "")
            if isinstance(content, str) and "\n" in content:
                new_msg["content"] = content.split("\n")
            result.append(new_msg)
        return result

    @staticmethod
    def _join_message_content(messages: list[Any]) -> list[dict[str, Any]]:
        """Convert message content arrays back into strings with newlines."""
        result = []
        for msg in messages:
            new_msg = dict(msg)
            content = new_msg.get("content", "")
            if isinstance(content, list):
                new_msg["content"] = "\n".join(content)
            result.append(new_msg)
        return result

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

    def to_export_dict(self) -> dict[str, Any]:
        """Convert to dictionary for export with readable formatting.

        - Message content with newlines is split into arrays of lines
        - Response JSON content is parsed into a proper JSON structure
        """
        return {
            "cache_key": {
                "model": self.model,
                "messages": self._split_message_content(self.messages),
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
            "response": self.response.to_export_dict(),
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LLMCacheEntry:
        """Create from dictionary.

        Handles both internal format (string content) and export format
        (array of lines for content).
        """
        cache_key = data.get("cache_key", {})
        messages = cache_key.get("messages", [])
        # Handle export format where content is array of lines
        messages = cls._join_message_content(messages)
        return cls(
            model=cache_key.get("model", ""),
            messages=messages,
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
        request_id: str = "",
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
            request_id: Optional identifier for the request (not part of hash).

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
                request_id=request_id,
            ),
        )


class LLMCompressor(JsonCompressor):
    """Compressor for LLM cache entries.

    Extends JsonCompressor with LLM-specific convenience methods for
    compressing/decompressing LLMCacheEntry objects.

    The compressor stores data in the standard JSON tokenised format,
    achieving 50-70% compression through the shared token dictionary.

    Example:
        >>> from causaliq_core.cache import TokenCache
        >>> from causaliq_knowledge.llm.cache import (
        ...     LLMCompressor, LLMCacheEntry,
        ... )
        >>> with TokenCache(":memory:") as cache:
        ...     compressor = LLMCompressor()
        ...     entry = LLMCacheEntry.create(
        ...         model="gpt-4",
        ...         messages=[{"role": "user", "content": "Hello"}],
        ...         content="Hi there!",
        ...         provider="openai",
        ...     )
        ...     blob = compressor.compress(entry.to_dict(), cache)
        ...     data = compressor.decompress(blob, cache)
        ...     restored = LLMCacheEntry.from_dict(data)
    """

    def compress_entry(self, entry: LLMCacheEntry, cache: TokenCache) -> bytes:
        """Compress an LLMCacheEntry to bytes.

        Convenience method that handles to_dict conversion.

        Args:
            entry: The cache entry to compress.
            cache: TokenCache for token dictionary.

        Returns:
            Compressed bytes.
        """
        return self.compress(entry.to_dict(), cache)

    def decompress_entry(
        self, blob: bytes, cache: TokenCache
    ) -> LLMCacheEntry:
        """Decompress bytes to an LLMCacheEntry.

        Convenience method that handles from_dict conversion.

        Args:
            blob: Compressed bytes.
            cache: TokenCache for token dictionary.

        Returns:
            Decompressed LLMCacheEntry.
        """
        data = self.decompress(blob, cache)
        return LLMCacheEntry.from_dict(data)

    def generate_export_filename(
        self, entry: LLMCacheEntry, cache_key: str
    ) -> str:
        """Generate a human-readable filename for export.

        Creates a filename using request_id, timestamp, and provider:
            {request_id}_{yyyy-mm-dd-hhmmss}_{provider}.json

        If request_id is not set, falls back to a short hash prefix.

        Args:
            entry: The cache entry to generate filename for.
            cache_key: The cache key (hash) for fallback uniqueness.

        Returns:
            Human-readable filename with .json extension.

        Example:
            >>> compressor = LLMCompressor()
            >>> entry = LLMCacheEntry.create(
            ...     model="gpt-4",
            ...     messages=[{"role": "user", "content": "test"}],
            ...     content="Response",
            ...     provider="openai",
            ...     request_id="expt23",
            ... )
            >>> # Returns something like: expt23_2026-01-29-143052_openai.json
        """
        import re
        from datetime import datetime

        # Get request_id or use hash prefix as fallback
        request_id = entry.metadata.request_id or cache_key[:8]
        # Sanitise request_id (alphanumeric, hyphens, underscores only)
        request_id = re.sub(r"[^a-zA-Z0-9_-]", "", request_id)
        if not request_id:
            request_id = cache_key[:8] if cache_key else "unknown"

        # Parse timestamp and format as yyyy-mm-dd-hhmmss
        timestamp_str = entry.metadata.timestamp
        if timestamp_str:
            try:
                # Parse ISO format timestamp
                dt = datetime.fromisoformat(
                    timestamp_str.replace("Z", "+00:00")
                )
                formatted_ts = dt.strftime("%Y-%m-%d-%H%M%S")
            except ValueError:
                formatted_ts = "unknown"
        else:
            formatted_ts = "unknown"

        # Get provider, sanitised
        provider = entry.metadata.provider or "unknown"
        provider = re.sub(r"[^a-z0-9]", "", provider.lower())
        if not provider:
            provider = "unknown"

        # Build filename: id_timestamp_provider.json
        return f"{request_id}_{formatted_ts}_{provider}.json"

    def export_entry(self, entry: LLMCacheEntry, path: Path) -> None:
        """Export an LLMCacheEntry to a JSON file.

        Uses to_export_dict() to parse JSON content for readability.

        Args:
            entry: The cache entry to export.
            path: Destination file path.
        """
        self.export(entry.to_export_dict(), path)

    def import_entry(self, path: Path) -> LLMCacheEntry:
        """Import an LLMCacheEntry from a JSON file.

        Args:
            path: Source file path.

        Returns:
            Imported LLMCacheEntry.
        """
        data = self.import_(path)
        return LLMCacheEntry.from_dict(data)
