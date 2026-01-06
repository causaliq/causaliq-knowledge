"""LiteLLM client wrapper with configuration and cost tracking."""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import litellm
from pydantic import BaseModel

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for an LLM client.

    Attributes:
        model: LiteLLM model identifier (e.g., "gpt-4o-mini", "ollama/llama3")
        temperature: Sampling temperature (0.0 = deterministic)
        max_tokens: Maximum tokens in response
        timeout: Request timeout in seconds
        max_retries: Number of retries on failure
        api_base: Optional custom API base URL (for local models)
    """

    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout: float = 30.0
    max_retries: int = 3
    api_base: Optional[str] = None


@dataclass
class LLMResponse:
    """Response from an LLM call.

    Attributes:
        content: The text content of the response
        model: The model that generated the response
        input_tokens: Number of input tokens used
        output_tokens: Number of output tokens generated
        cost: Estimated cost in USD (if available)
        raw_response: The raw response object from LiteLLM
    """

    content: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    raw_response: Optional[Any] = field(default=None, repr=False)

    def parse_json(self) -> Optional[dict]:
        """Parse content as JSON.

        Returns:
            Parsed JSON dict, or None if parsing fails.
        """
        try:
            # Handle potential markdown code blocks
            text = self.content.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            result: dict = json.loads(text.strip())
            return result
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            return None


class LLMClient:
    """Client for making LLM calls via LiteLLM.

    This class wraps LiteLLM to provide:
    - Consistent configuration across calls
    - Cost tracking and token counting
    - JSON response parsing
    - Error handling with retries

    Example:
        >>> client = LLMClient(LLMConfig(model="gpt-4o-mini"))
        >>> response = client.complete(
        ...     system="You are a helpful assistant.",
        ...     user="What is 2+2?"
        ... )
        >>> print(response.content)
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        """Initialize the LLM client.

        Args:
            config: LLM configuration. Defaults to gpt-4o-mini.
        """
        self.config = config or LLMConfig()
        self._total_cost: float = 0.0
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._call_count: int = 0

    @property
    def total_cost(self) -> float:
        """Total cost in USD across all calls."""
        return self._total_cost

    @property
    def total_input_tokens(self) -> int:
        """Total input tokens across all calls."""
        return self._total_input_tokens

    @property
    def total_output_tokens(self) -> int:
        """Total output tokens across all calls."""
        return self._total_output_tokens

    @property
    def call_count(self) -> int:
        """Number of successful calls made."""
        return self._call_count

    def reset_stats(self) -> None:
        """Reset cost and token tracking statistics."""
        self._total_cost = 0.0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._call_count = 0

    def complete(
        self,
        user: str,
        system: Optional[str] = None,
        response_format: Optional[type[BaseModel]] = None,
    ) -> LLMResponse:
        """Make a completion request to the LLM.

        Args:
            user: The user message/prompt.
            system: Optional system message.
            response_format: Optional Pydantic model for structured output.

        Returns:
            LLMResponse with content and usage statistics.

        Raises:
            litellm.exceptions.APIError: On API failures after retries.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "timeout": self.config.timeout,
            "num_retries": self.config.max_retries,
        }

        if self.config.api_base:
            kwargs["api_base"] = self.config.api_base

        if response_format:
            kwargs["response_format"] = response_format

        logger.debug(f"Calling {self.config.model} with {len(messages)} msgs")

        response = litellm.completion(**kwargs)

        # Extract content
        content = response.choices[0].message.content or ""

        # Extract usage info
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        output_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

        # Calculate cost using LiteLLM's cost tracking
        cost = litellm.completion_cost(completion_response=response)

        # Update totals
        self._total_cost += cost
        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens
        self._call_count += 1

        logger.debug(
            f"Response: {input_tokens} in, {output_tokens} out, ${cost:.6f}"
        )

        return LLMResponse(
            content=content,
            model=response.model or self.config.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            raw_response=response,
        )

    def complete_json(
        self,
        user: str,
        system: Optional[str] = None,
    ) -> tuple[Optional[dict], LLMResponse]:
        """Make a completion request and parse response as JSON.

        Convenience method that calls complete() and parses the result.

        Args:
            user: The user message/prompt.
            system: Optional system message.

        Returns:
            Tuple of (parsed_json, response). parsed_json may be None
            if parsing fails.
        """
        response = self.complete(user=user, system=system)
        parsed = response.parse_json()
        return parsed, response

    def get_stats(self) -> dict:
        """Get current usage statistics.

        Returns:
            Dictionary with cost, token, and call count statistics.
        """
        return {
            "total_cost": self._total_cost,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "call_count": self._call_count,
            "model": self.config.model,
        }
