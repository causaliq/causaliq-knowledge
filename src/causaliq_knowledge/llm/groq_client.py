"""Direct Groq API client - clean and reliable."""

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
class GroqConfig(LLMConfig):
    """Configuration for Groq API client.

    Extends LLMConfig with Groq-specific defaults.

    Attributes:
        model: Groq model identifier (default: llama-3.1-8b-instant).
        temperature: Sampling temperature (default: 0.1).
        max_tokens: Maximum response tokens (default: 500).
        timeout: Request timeout in seconds (default: 30.0).
        api_key: Groq API key (falls back to GROQ_API_KEY env var).
    """

    model: str = "llama-3.1-8b-instant"
    temperature: float = 0.1
    max_tokens: int = 500
    timeout: float = 30.0
    api_key: Optional[str] = None

    def __post_init__(self) -> None:
        """Set API key from environment if not provided."""
        if self.api_key is None:
            self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable is required")


class GroqClient(BaseLLMClient):
    """Direct Groq API client.

    Implements the BaseLLMClient interface for Groq's API.
    Uses httpx for HTTP requests.

    Example:
        >>> config = GroqConfig(model="llama-3.1-8b-instant")
        >>> client = GroqClient(config)
        >>> msgs = [{"role": "user", "content": "Hello"}]
        >>> response = client.completion(msgs)
        >>> print(response.content)
    """

    BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(self, config: Optional[GroqConfig] = None) -> None:
        """Initialize Groq client.

        Args:
            config: Groq configuration. If None, uses defaults with
                   API key from GROQ_API_KEY environment variable.
        """
        self.config = config or GroqConfig()
        self._total_calls = 0

    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return "groq"

    def completion(
        self, messages: List[Dict[str, str]], **kwargs: Any
    ) -> LLMResponse:
        """Make a chat completion request to Groq.

        Args:
            messages: List of message dicts with "role" and "content" keys.
            **kwargs: Override config options (temperature, max_tokens).

        Returns:
            LLMResponse with the generated content and metadata.

        Raises:
            ValueError: If the API request fails.
        """
        # Build request payload
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "stream": False,
        }

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        logger.debug(f"Calling Groq API with model: {payload['model']}")

        try:
            with httpx.Client(timeout=self.config.timeout) as client:
                response = client.post(
                    f"{self.BASE_URL}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()

                data = response.json()

                # Extract response data
                content = data["choices"][0]["message"]["content"] or ""
                usage = data.get("usage", {})
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)

                self._total_calls += 1

                logger.debug(
                    f"Groq response: {input_tokens} in, {output_tokens} out"
                )

                return LLMResponse(
                    content=content,
                    model=data.get("model", self.config.model),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost=0.0,  # Free tier
                    raw_response=data,
                )

        except httpx.HTTPStatusError as e:
            logger.error(
                f"Groq API error: {e.response.status_code} - {e.response.text}"
            )
            raise ValueError(
                f"Groq API error: {e.response.status_code} - {e.response.text}"
            )
        except httpx.TimeoutException:
            raise ValueError("Groq API request timed out")
        except Exception as e:
            logger.error(f"Groq API unexpected error: {e}")
            raise ValueError(f"Groq API error: {str(e)}")

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
        """Check if Groq API is available.

        Returns:
            True if GROQ_API_KEY is configured.
        """
        return bool(self.config.api_key)

    def list_models(self) -> List[str]:
        """List available models from Groq API.

        Queries the Groq API to get models accessible with the current
        API key. Filters to only include text generation models.

        Returns:
            List of model identifiers (e.g., ['llama-3.1-8b-instant', ...]).

        Raises:
            ValueError: If the API request fails.
        """
        try:
            with httpx.Client(timeout=self.config.timeout) as client:
                response = client.get(
                    f"{self.BASE_URL}/models",
                    headers={"Authorization": f"Bearer {self.config.api_key}"},
                )
                response.raise_for_status()
                data = response.json()

                # Filter and sort models
                models = []
                for model in data.get("data", []):
                    model_id = model.get("id", "")
                    # Skip whisper (audio), guard, and safeguard models
                    if any(
                        x in model_id.lower()
                        for x in ["whisper", "guard", "embed"]
                    ):
                        continue
                    models.append(model_id)

                return sorted(models)

        except httpx.HTTPStatusError as e:
            raise ValueError(
                f"Groq API error: {e.response.status_code} - {e.response.text}"
            )
        except Exception as e:
            raise ValueError(f"Failed to list Groq models: {e}")
