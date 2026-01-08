"""Abstract base class for LLM clients.

This module defines the common interface that all LLM vendor clients
must implement. This provides a consistent API regardless of the
underlying LLM provider.
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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
        raw_response: The original provider-specific response (for debugging).
    """

    content: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    raw_response: Optional[Dict[str, Any]] = field(default=None, repr=False)

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

    @property
    def model_name(self) -> str:
        """Return the model name being used.

        Returns:
            Model identifier string.
        """
        return getattr(self, "config", LLMConfig(model="unknown")).model
