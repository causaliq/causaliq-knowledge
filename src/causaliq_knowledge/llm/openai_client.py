"""Direct OpenAI API client - clean and reliable."""

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
class OpenAIConfig(LLMConfig):
    """Configuration for OpenAI API client.

    Extends LLMConfig with OpenAI-specific defaults.

    Attributes:
        model: OpenAI model identifier (default: gpt-4o-mini).
        temperature: Sampling temperature (default: 0.1).
        max_tokens: Maximum response tokens (default: 500).
        timeout: Request timeout in seconds (default: 30.0).
        api_key: OpenAI API key (falls back to OPENAI_API_KEY env var).
    """

    model: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_tokens: int = 500
    timeout: float = 30.0
    api_key: Optional[str] = None

    def __post_init__(self) -> None:
        """Set API key from environment if not provided."""
        if self.api_key is None:
            self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")


class OpenAIClient(BaseLLMClient):
    """Direct OpenAI API client.

    Implements the BaseLLMClient interface for OpenAI's API.
    Uses httpx for HTTP requests.

    Example:
        >>> config = OpenAIConfig(model="gpt-4o-mini")
        >>> client = OpenAIClient(config)
        >>> msgs = [{"role": "user", "content": "Hello"}]
        >>> response = client.completion(msgs)
        >>> print(response.content)
    """

    BASE_URL = "https://api.openai.com/v1"

    def __init__(self, config: Optional[OpenAIConfig] = None) -> None:
        """Initialize OpenAI client.

        Args:
            config: OpenAI configuration. If None, uses defaults with
                   API key from OPENAI_API_KEY environment variable.
        """
        self.config = config or OpenAIConfig()
        self._total_calls = 0

    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return "openai"

    def completion(
        self, messages: List[Dict[str, str]], **kwargs: Any
    ) -> LLMResponse:
        """Make a chat completion request to OpenAI.

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

        logger.debug(f"Calling OpenAI API with model: {payload['model']}")

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
                    f"OpenAI response: {input_tokens} in, {output_tokens} out"
                )

                # Calculate cost (approximate pricing)
                cost = self._calculate_cost(
                    self.config.model, input_tokens, output_tokens
                )

                return LLMResponse(
                    content=content,
                    model=data.get("model", self.config.model),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost=cost,
                    raw_response=data,
                )

        except httpx.HTTPStatusError as e:
            msg = f"OpenAI API error: {e.response.status_code}"
            logger.error(f"{msg} - {e.response.text}")
            raise ValueError(f"{msg} - {e.response.text}")
        except httpx.TimeoutException:
            raise ValueError("OpenAI API request timed out")
        except Exception as e:
            logger.error(f"OpenAI API unexpected error: {e}")
            raise ValueError(f"OpenAI API error: {str(e)}")

    def _calculate_cost(
        self, model: str, input_tokens: int, output_tokens: int
    ) -> float:
        """Calculate approximate cost for OpenAI API call.

        Args:
            model: Model identifier.
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.

        Returns:
            Estimated cost in USD.
        """
        # Pricing per 1M tokens (as of 2024)
        # Order matters - more specific prefixes must come first
        pricing = {
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
            "gpt-4o": {"input": 2.50, "output": 10.00},
            "gpt-4-turbo": {"input": 10.00, "output": 30.00},
            "gpt-4": {"input": 30.00, "output": 60.00},
            "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
            "o1-mini": {"input": 3.00, "output": 12.00},
            "o1": {"input": 15.00, "output": 60.00},
        }

        # Find matching pricing (check if model starts with known prefix)
        model_pricing = None
        for key in pricing:
            if model.startswith(key):
                model_pricing = pricing[key]
                break

        if not model_pricing:
            return 0.0

        input_cost = (input_tokens / 1_000_000) * model_pricing["input"]
        output_cost = (output_tokens / 1_000_000) * model_pricing["output"]

        return input_cost + output_cost

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
        """Check if OpenAI API is available.

        Returns:
            True if OPENAI_API_KEY is configured.
        """
        return bool(self.config.api_key)

    def list_models(self) -> List[str]:
        """List available models from OpenAI API.

        Queries the OpenAI API to get models accessible with the current
        API key. Filters to only include GPT chat models.

        Returns:
            List of model identifiers (e.g., ['gpt-4o', 'gpt-4o-mini', ...]).

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

                # Filter to chat completion models only
                models = []
                for model in data.get("data", []):
                    model_id = model.get("id", "")
                    # Include GPT and o1 models, exclude embeddings/whisper/etc
                    if any(
                        prefix in model_id
                        for prefix in ["gpt-4", "gpt-3.5", "o1", "o3"]
                    ):
                        # Exclude instruct variants and specific exclusions
                        if any(
                            x in model_id.lower()
                            for x in [
                                "instruct",
                                "vision",
                                "audio",
                                "realtime",
                            ]
                        ):
                            continue
                        models.append(model_id)

                return sorted(models)

        except httpx.HTTPStatusError as e:
            msg = f"OpenAI API error: {e.response.status_code}"
            raise ValueError(f"{msg} - {e.response.text}")
        except Exception as e:
            raise ValueError(f"Failed to list OpenAI models: {e}")
