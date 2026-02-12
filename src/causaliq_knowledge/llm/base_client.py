"""Abstract base class for LLM clients.

This module defines the common interface that all LLM vendor clients
must implement. This provides a consistent API regardless of the
underlying LLM provider.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:  # pragma: no cover
    from causaliq_core.cache import TokenCache

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Base configuration for all LLM clients.

    This dataclass defines common configuration options shared by all
    LLM provider clients. Vendor-specific clients may extend this with
    additional options.

    Attributes:
        model: Model identifier (provider-specific format).
        temperature: Sampling temperature (0.0=deterministic, 1.0=creative).
        max_tokens: Maximum tokens in the response.
        timeout: Request timeout in seconds.
        api_key: API key for authentication (optional, can use env var).
    """

    model: str
    temperature: float = 0.1
    max_tokens: int = 500
    timeout: float = 30.0
    api_key: Optional[str] = None


@dataclass
class LLMResponse:
    """Standard response from any LLM client.

    This dataclass provides a unified response format across all LLM providers,
    abstracting away provider-specific response structures.

    Attributes:
        content: The text content of the response.
        model: The model that generated the response.
        input_tokens: Number of input/prompt tokens used.
        output_tokens: Number of output/completion tokens generated.
        cost: Estimated cost of the request (if available).
        finish_reason: Why generation stopped (stop, length, etc.).
        raw_response: The original provider-specific response (for debugging).
        llm_timestamp: Original LLM response timestamp (from cache or current).
        llm_latency_ms: Original LLM response latency (from cache or current).
    """

    content: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    finish_reason: str = "stop"
    raw_response: Optional[Dict[str, Any]] = field(default=None, repr=False)
    llm_timestamp: Optional[datetime] = field(default=None, repr=False)
    llm_latency_ms: Optional[int] = field(default=None, repr=False)

    def parse_json(self) -> Optional[Dict[str, Any]]:
        """Parse content as JSON, handling common formatting issues.

        LLMs sometimes wrap JSON in markdown code blocks. This method
        handles those cases and attempts to extract valid JSON.

        Returns:
            Parsed JSON as dict, or None if parsing fails.
        """
        try:
            # Clean up potential markdown code blocks
            text = self.content.strip()
            if text.startswith("```json"):
                text = text[7:]
            elif text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]

            return json.loads(text.strip())  # type: ignore[no-any-return]
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            return None


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients.

    All LLM vendor clients (OpenAI, Anthropic, Groq, Gemini, Llama, etc.)
    must implement this interface to ensure consistent behavior across
    the codebase.

    This abstraction allows:
    - Easy addition of new LLM providers
    - Consistent API for all providers
    - Provider-agnostic code in higher-level modules
    - Simplified testing with mock implementations

    Example:
        >>> class MyClient(BaseLLMClient):
        ...     def completion(self, messages, **kwargs):
        ...         # Implementation here
        ...         pass
        ...
        >>> client = MyClient(config)
        >>> msgs = [{"role": "user", "content": "Hello"}]
        >>> response = client.completion(msgs)
        >>> print(response.content)
    """

    @abstractmethod
    def __init__(self, config: LLMConfig) -> None:
        """Initialize the client with configuration.

        Args:
            config: Configuration for the LLM client.
        """
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of the LLM provider.

        Returns:
            Provider name (e.g., "openai", "anthropic", "groq").
        """
        pass

    @abstractmethod
    def completion(
        self, messages: List[Dict[str, str]], **kwargs: Any
    ) -> LLMResponse:
        """Make a chat completion request.

        This is the core method that sends a request to the LLM provider
        and returns a standardized response.

        Args:
            messages: List of message dicts with "role" and "content" keys.
                Roles can be: "system", "user", "assistant".
            **kwargs: Provider-specific options (temperature, max_tokens, etc.)
                that override the config defaults.

        Returns:
            LLMResponse with the generated content and metadata.

        Raises:
            ValueError: If the API request fails or returns an error.
        """
        pass

    def complete_json(
        self, messages: List[Dict[str, str]], **kwargs: Any
    ) -> tuple[Optional[Dict[str, Any]], LLMResponse]:
        """Make a completion request and parse response as JSON.

        Convenience method that calls completion() and attempts to parse
        the response content as JSON.

        Args:
            messages: List of message dicts with "role" and "content" keys.
            **kwargs: Provider-specific options passed to completion().

        Returns:
            Tuple of (parsed JSON dict or None, raw LLMResponse).
        """
        response = self.completion(messages, **kwargs)
        parsed = response.parse_json()
        return parsed, response

    @property
    @abstractmethod
    def call_count(self) -> int:
        """Return the number of API calls made by this client.

        Returns:
            Total number of completion calls made.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM provider is available and configured.

        This method checks whether the client can make API calls:
        - For cloud providers: checks if API key is set
        - For local providers: checks if server is running

        Returns:
            True if the provider is available and ready for requests.
        """
        pass

    @abstractmethod
    def list_models(self) -> List[str]:
        """List available models from the provider.

        Queries the provider's API to get the list of models accessible
        with the current API key or configuration. Results are filtered
        by the user's subscription/access level.

        Returns:
            List of model identifiers available for use.

        Raises:
            ValueError: If the API request fails.
        """
        pass

    @property
    def model_name(self) -> str:
        """Return the model name being used.

        Returns:
            Model identifier string.
        """
        return getattr(self, "config", LLMConfig(model="unknown")).model

    def _build_cache_key(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Build a deterministic cache key for the request.

        Creates a SHA-256 hash from the model, messages, temperature, and
        max_tokens. The hash is truncated to 16 hex characters (64 bits).

        Args:
            messages: List of message dicts with "role" and "content" keys.
            temperature: Sampling temperature (defaults to config value).
            max_tokens: Maximum tokens (defaults to config value).

        Returns:
            16-character hex string cache key.
        """
        config = getattr(self, "config", LLMConfig(model="unknown"))
        key_data = {
            "model": config.model,
            "messages": messages,
            "temperature": (
                temperature if temperature is not None else config.temperature
            ),
            "max_tokens": (
                max_tokens if max_tokens is not None else config.max_tokens
            ),
        }
        key_json = json.dumps(key_data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(key_json.encode()).hexdigest()[:16]

    def set_cache(
        self,
        cache: Optional["TokenCache"],
        use_cache: bool = True,
    ) -> None:
        """Configure caching for this client.

        Args:
            cache: TokenCache instance for caching, or None to disable.
            use_cache: Whether to use the cache (default True).
        """
        self._cache = cache
        self._use_cache = use_cache

    @property
    def cache(self) -> Optional["TokenCache"]:
        """Return the configured cache, if any."""
        return getattr(self, "_cache", None)

    @property
    def use_cache(self) -> bool:
        """Return whether caching is enabled."""
        return getattr(self, "_use_cache", True)

    def cached_completion(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> LLMResponse:
        """Make a completion request with caching.

        If caching is enabled and a cached response exists, returns
        the cached response without making an API call. Otherwise,
        makes the API call and caches the result.

        Args:
            messages: List of message dicts with "role" and "content" keys.
            **kwargs: Provider-specific options (temperature, max_tokens, etc.)
                Also accepts request_id (str) for identifying requests in
                exports. Note: request_id is NOT part of the cache key.

        Returns:
            LLMResponse with the generated content and metadata.
        """
        from causaliq_knowledge.llm.cache import LLMCacheEntry, LLMEntryEncoder

        cache = self.cache
        use_cache = self.use_cache

        # Extract request_id (not part of cache key)
        request_id = kwargs.pop("request_id", "")

        # Build cache key
        temperature = kwargs.get("temperature")
        max_tokens = kwargs.get("max_tokens")
        cache_key = self._build_cache_key(messages, temperature, max_tokens)

        # Check cache
        if use_cache and cache is not None:
            # Ensure encoder is registered
            if not cache.has_encoder("llm"):
                cache.register_encoder("llm", LLMEntryEncoder())

            if cache.exists(cache_key, "llm"):
                cached_data = cache.get_data(cache_key, "llm")
                if cached_data is not None:
                    entry = LLMCacheEntry.from_dict(cached_data)
                    # Parse original timestamp from cache
                    llm_ts: Optional[datetime] = None
                    if entry.metadata.timestamp:
                        try:
                            llm_ts = datetime.fromisoformat(
                                entry.metadata.timestamp.replace("Z", "+00:00")
                            )
                        except ValueError:
                            pass
                    return LLMResponse(
                        content=entry.response.content,
                        model=entry.model,
                        input_tokens=entry.metadata.tokens.input,
                        output_tokens=entry.metadata.tokens.output,
                        cost=entry.metadata.cost_usd or 0.0,
                        llm_timestamp=llm_ts,
                        llm_latency_ms=entry.metadata.latency_ms,
                    )

        # Make API call with timing
        request_time = datetime.now(timezone.utc)
        start_time = time.perf_counter()
        response = self.completion(messages, **kwargs)
        latency_ms = int((time.perf_counter() - start_time) * 1000)

        # Set original LLM metadata on response
        response.llm_timestamp = request_time
        response.llm_latency_ms = latency_ms

        # Store in cache
        if use_cache and cache is not None:
            config = getattr(self, "config", LLMConfig(model="unknown"))
            entry = LLMCacheEntry.create(
                model=config.model,
                messages=messages,
                content=response.content,
                temperature=(
                    temperature
                    if temperature is not None
                    else config.temperature
                ),
                max_tokens=(
                    max_tokens if max_tokens is not None else config.max_tokens
                ),
                provider=self.provider_name,
                latency_ms=latency_ms,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                cost_usd=response.cost,
                request_id=request_id,
            )
            cache.put_data(cache_key, "llm", entry.to_dict())

        return response
