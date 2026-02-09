"""Base class for OpenAI-compatible API clients.

This module provides a shared base class for LLM providers that implement
the OpenAI API format (OpenAI, DeepSeek, Mistral, etc.).
"""

import logging
from abc import abstractmethod
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
class OpenAICompatConfig(LLMConfig):
    """Base configuration for OpenAI-compatible API clients.

    Attributes:
        model: Model identifier.
        temperature: Sampling temperature (default: 0.1).
        max_tokens: Maximum response tokens (default: 500).
        timeout: Request timeout in seconds (default: 30.0).
        api_key: API key (provider-specific env var fallback).
    """

    model: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_tokens: int = 500
    timeout: float = 30.0
    api_key: Optional[str] = None


class OpenAICompatClient(BaseLLMClient):
    """Base class for OpenAI-compatible API clients.

    Provides shared implementation for providers that use the OpenAI API
    format (chat/completions endpoint, same request/response structure).

    Subclasses must implement:
        - BASE_URL: The API base URL
        - PROVIDER_NAME: Name for logging
        - ENV_VAR: Environment variable for API key
        - _get_pricing(): Return pricing dict for cost calculation
        - _filter_models(): Optional model list filtering

    Example:
        >>> class MyClient(OpenAICompatClient):
        ...     BASE_URL = "https://api.example.com/v1"
        ...     PROVIDER_NAME = "example"
        ...     ENV_VAR = "EXAMPLE_API_KEY"
    """

    # Subclasses must override these
    BASE_URL: str = ""
    PROVIDER_NAME: str = "openai-compat"
    ENV_VAR: str = "API_KEY"

    def __init__(self, config: Optional[OpenAICompatConfig] = None) -> None:
        """Initialize the client.

        Args:
            config: Client configuration. If None, uses defaults with
                   API key from environment variable.
        """
        self.config = config or self._default_config()
        self._total_calls = 0

    @abstractmethod
    def _default_config(self) -> OpenAICompatConfig:
        """Return default configuration for this provider."""
        pass

    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return self.PROVIDER_NAME

    def completion(
        self, messages: List[Dict[str, str]], **kwargs: Any
    ) -> LLMResponse:
        """Make a chat completion request.

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

        logger.debug(
            f"Calling {self.PROVIDER_NAME} API with model: {payload['model']}"
        )

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
                finish_reason = data["choices"][0].get("finish_reason", "stop")
                usage = data.get("usage", {})
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)

                self._total_calls += 1

                logger.debug(
                    f"{self.PROVIDER_NAME} response: "
                    f"{input_tokens} in, {output_tokens} out"
                )

                # Calculate cost
                cost = self._calculate_cost(
                    self.config.model, input_tokens, output_tokens
                )

                return LLMResponse(
                    content=content,
                    model=data.get("model", self.config.model),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost=cost,
                    finish_reason=finish_reason,
                    raw_response=data,
                )

        except httpx.HTTPStatusError as e:
            msg = f"{self.PROVIDER_NAME} API error: {e.response.status_code}"
            logger.error(f"{msg} - {e.response.text}")
            raise ValueError(f"{msg} - {e.response.text}")
        except httpx.TimeoutException:
            raise ValueError(f"{self.PROVIDER_NAME} API request timed out")
        except Exception as e:
            logger.error(f"{self.PROVIDER_NAME} API unexpected error: {e}")
            raise ValueError(f"{self.PROVIDER_NAME} API error: {str(e)}")

    def _calculate_cost(
        self, model: str, input_tokens: int, output_tokens: int
    ) -> float:
        """Calculate approximate cost for API call.

        Args:
            model: Model identifier.
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.

        Returns:
            Estimated cost in USD.
        """
        pricing = self._get_pricing()

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

    @abstractmethod
    def _get_pricing(self) -> Dict[str, Dict[str, float]]:
        """Return pricing dict for this provider.

        Returns:
            Dict mapping model prefixes to input/output costs per 1M tokens.
            Example: {"gpt-4o": {"input": 2.50, "output": 10.00}}
        """
        pass

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
        """Check if the API is available.

        Returns:
            True if API key is configured.
        """
        return bool(self.config.api_key)

    def list_models(self) -> List[str]:
        """List available models from the API.

        Queries the API to get models accessible with the current
        API key, then filters using _filter_models().

        Returns:
            List of model identifiers.

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

                # Get model IDs from response
                all_models = [
                    model.get("id", "") for model in data.get("data", [])
                ]

                # Filter using provider-specific logic
                models = self._filter_models(all_models)

                return sorted(models)

        except httpx.HTTPStatusError as e:
            msg = f"{self.PROVIDER_NAME} API error: {e.response.status_code}"
            raise ValueError(f"{msg} - {e.response.text}")
        except Exception as e:
            raise ValueError(
                f"Failed to list {self.PROVIDER_NAME} models: {e}"
            )

    def _filter_models(self, models: List[str]) -> List[str]:
        """Filter model list to relevant models.

        Override in subclasses to customize filtering.

        Args:
            models: List of all model IDs from API.

        Returns:
            Filtered list of relevant model IDs.
        """
        return models
