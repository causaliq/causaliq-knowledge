"""Direct Anthropic API client - clean and reliable."""

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

from causaliq_knowledge.llm.base_client import (
    BaseLLMClient,
    LLMConfig,
    LLMResponse,
)

logger = logging.getLogger(__name__)


@dataclass
class AnthropicConfig(LLMConfig):
    """Configuration for Anthropic API client.

    Extends LLMConfig with Anthropic-specific defaults.

    Attributes:
        model: Anthropic model identifier (default: claude-sonnet-4-20250514).
        temperature: Sampling temperature (default: 0.1).
        max_tokens: Maximum response tokens (default: 500).
        timeout: Request timeout in seconds (default: 30.0).
        api_key: Anthropic API key (falls back to ANTHROPIC_API_KEY env var).
    """

    model: str = "claude-sonnet-4-20250514"
    temperature: float = 0.1
    max_tokens: int = 500
    timeout: float = 30.0
    api_key: Optional[str] = None

    def __post_init__(self) -> None:
        """Set API key from environment if not provided."""
        if self.api_key is None:
            self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable is required"
            )


class AnthropicClient(BaseLLMClient):
    """Direct Anthropic API client.

    Implements the BaseLLMClient interface for Anthropic's Claude API.
    Uses httpx for HTTP requests.

    Example:
        >>> config = AnthropicConfig(model="claude-sonnet-4-20250514")
        >>> client = AnthropicClient(config)
        >>> msgs = [{"role": "user", "content": "Hello"}]
        >>> response = client.completion(msgs)
        >>> print(response.content)
    """

    BASE_URL = "https://api.anthropic.com/v1"
    API_VERSION = "2023-06-01"

    def __init__(self, config: Optional[AnthropicConfig] = None) -> None:
        """Initialize Anthropic client.

        Args:
            config: Anthropic configuration. If None, uses defaults with
                   API key from ANTHROPIC_API_KEY environment variable.
        """
        self.config = config or AnthropicConfig()
        self._total_calls = 0

    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return "anthropic"

    def completion(
        self, messages: List[Dict[str, str]], **kwargs: Any
    ) -> LLMResponse:
        """Make a chat completion request to Anthropic.

        Args:
            messages: List of message dicts with "role" and "content" keys.
            **kwargs: Override config options (temperature, max_tokens).

        Returns:
            LLMResponse with the generated content and metadata.

        Raises:
            ValueError: If the API request fails.
        """
        # Anthropic uses separate system parameter, not in messages
        system_content = None
        filtered_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                filtered_messages.append(msg)

        # Build request payload in Anthropic's format
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": filtered_messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
        }

        # Add system prompt if present
        if system_content:
            payload["system"] = system_content

        # api_key is guaranteed non-None after __post_init__ validation
        headers: dict[str, str] = {
            "x-api-key": self.config.api_key,  # type: ignore[dict-item]
            "anthropic-version": self.API_VERSION,
            "Content-Type": "application/json",
        }

        logger.debug(f"Calling Anthropic API with model: {self.config.model}")

        try:
            with httpx.Client(timeout=self.config.timeout) as client:
                response = client.post(
                    f"{self.BASE_URL}/messages",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()

                data = response.json()

                # Extract response content from Anthropic format
                content_blocks = data.get("content", [])
                content = ""
                for block in content_blocks:
                    if block.get("type") == "text":
                        content += block.get("text", "")

                # Extract usage info
                usage = data.get("usage", {})
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)

                self._total_calls += 1

                logger.debug(
                    f"Anthropic response: {input_tokens} in, "
                    f"{output_tokens} out"
                )

                return LLMResponse(
                    content=content,
                    model=data.get("model", self.config.model),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost=0.0,  # Cost calculation not implemented
                    raw_response=data,
                )

        except httpx.HTTPStatusError as e:
            try:
                error_data = e.response.json()
                error_msg = error_data.get("error", {}).get(
                    "message", e.response.text
                )
            except Exception:
                error_msg = e.response.text

            logger.error(
                f"Anthropic API HTTP error: {e.response.status_code} - "
                f"{error_msg}"
            )
            raise ValueError(
                f"Anthropic API error: {e.response.status_code} - {error_msg}"
            )
        except httpx.TimeoutException:
            raise ValueError("Anthropic API request timed out")
        except Exception as e:
            logger.error(f"Anthropic API unexpected error: {e}")
            raise ValueError(f"Anthropic API error: {str(e)}")

    def complete_json(
        self, messages: List[Dict[str, str]], **kwargs: Any
    ) -> tuple[Optional[Dict[str, Any]], LLMResponse]:
        """Make a completion request and parse response as JSON.

        Args:
            messages: List of message dicts with "role" and "content" keys.
            **kwargs: Override config options passed to completion().

        Returns:
            Tuple of (parsed JSON dict or None, raw LLMResponse).
        """
        response = self.completion(messages, **kwargs)
        parsed = response.parse_json()
        return parsed, response

    @property
    def call_count(self) -> int:
        """Return the number of API calls made."""
        return self._total_calls

    def is_available(self) -> bool:
        """Check if Anthropic API is available.

        Returns:
            True if ANTHROPIC_API_KEY is configured.
        """
        return bool(self.config.api_key)

    def list_models(self) -> List[str]:
        """List available Claude models from Anthropic API.

        Queries the Anthropic /v1/models endpoint to get available models.

        Returns:
            List of model identifiers
            (e.g., ['claude-sonnet-4-20250514', ...]).
        """
        if not self.config.api_key:
            return []

        headers: dict[str, str] = {
            "x-api-key": self.config.api_key,
            "anthropic-version": self.API_VERSION,
        }

        try:
            with httpx.Client(timeout=self.config.timeout) as client:
                response = client.get(
                    f"{self.BASE_URL}/models",
                    headers=headers,
                )
                response.raise_for_status()

                data = response.json()
                models = []
                for model_info in data.get("data", []):
                    model_id = model_info.get("id")
                    if model_id:
                        models.append(model_id)

                return sorted(models)

        except httpx.HTTPStatusError as e:
            logger.warning(f"Anthropic API error listing models: {e}")
            return []
        except Exception as e:
            logger.warning(f"Error listing Anthropic models: {e}")
            return []
