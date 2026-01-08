"""Local Ollama API client for running Llama models locally."""

import logging
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
class OllamaConfig(LLMConfig):
    """Configuration for Ollama API client.

    Extends LLMConfig with Ollama-specific defaults.

    Attributes:
        model: Ollama model identifier (default: llama3.2:1b).
        temperature: Sampling temperature (default: 0.1).
        max_tokens: Maximum response tokens (default: 500).
        timeout: Request timeout in seconds (default: 120.0, local).
        api_key: Not used for Ollama (local server).
        base_url: Ollama server URL (default: http://localhost:11434).
    """

    model: str = "llama3.2:1b"
    temperature: float = 0.1
    max_tokens: int = 500
    timeout: float = 120.0  # Local inference can be slow
    api_key: Optional[str] = None  # Not needed for local Ollama
    base_url: str = "http://localhost:11434"


class OllamaClient(BaseLLMClient):
    """Local Ollama API client.

    Implements the BaseLLMClient interface for locally running Ollama server.
    Uses httpx for HTTP requests to the local Ollama API.

    Ollama provides an OpenAI-compatible API for running open-source models
    like Llama locally without requiring API keys or internet access.

    Example:
        >>> config = OllamaConfig(model="llama3.2:1b")
        >>> client = OllamaClient(config)
        >>> msgs = [{"role": "user", "content": "Hello"}]
        >>> response = client.completion(msgs)
        >>> print(response.content)
    """

    def __init__(self, config: Optional[OllamaConfig] = None) -> None:
        """Initialize Ollama client.

        Args:
            config: Ollama configuration. If None, uses defaults connecting
                   to localhost:11434 with llama3.2:1b model.
        """
        self.config = config or OllamaConfig()
        self._total_calls = 0

    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return "ollama"

    def completion(
        self, messages: List[Dict[str, str]], **kwargs: Any
    ) -> LLMResponse:
        """Make a chat completion request to Ollama.

        Args:
            messages: List of message dicts with "role" and "content" keys.
            **kwargs: Override config options (temperature, max_tokens).

        Returns:
            LLMResponse with the generated content and metadata.

        Raises:
            ValueError: If the API request fails or Ollama is not running.
        """
        # Build request payload (Ollama uses similar format to OpenAI)
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": kwargs.get(
                    "temperature", self.config.temperature
                ),
                "num_predict": kwargs.get(
                    "max_tokens", self.config.max_tokens
                ),
            },
        }

        url = f"{self.config.base_url}/api/chat"
        headers = {"Content-Type": "application/json"}

        logger.debug(f"Calling Ollama API with model: {self.config.model}")

        try:
            with httpx.Client(timeout=self.config.timeout) as client:
                response = client.post(url, json=payload, headers=headers)
                response.raise_for_status()

                data = response.json()

                # Extract response content
                content = data.get("message", {}).get("content", "")

                # Extract token counts (Ollama provides these)
                input_tokens = data.get("prompt_eval_count", 0)
                output_tokens = data.get("eval_count", 0)

                self._total_calls += 1

                logger.debug(
                    f"Ollama response: {input_tokens} in, {output_tokens} out"
                )

                return LLMResponse(
                    content=content,
                    model=data.get("model", self.config.model),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost=0.0,  # Local inference is free
                    raw_response=data,
                )

        except httpx.ConnectError:
            raise ValueError(
                "Could not connect to Ollama. "
                "Make sure Ollama is running (run 'ollama serve' or start "
                "the Ollama app). "
                f"Tried to connect to: {self.config.base_url}"
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ValueError(
                    f"Model '{self.config.model}' not found. "
                    f"Run 'ollama pull {self.config.model}' to download it."
                )
            logger.error(
                f"Ollama API error: {e.response.status_code} - "
                f"{e.response.text}"
            )
            raise ValueError(
                f"Ollama API error: {e.response.status_code} - "
                f"{e.response.text}"
            )
        except httpx.TimeoutException:
            raise ValueError(
                "Ollama API request timed out. Local inference can be slow - "
                "try increasing the timeout in OllamaConfig."
            )
        except Exception as e:
            logger.error(f"Ollama API unexpected error: {e}")
            raise ValueError(f"Ollama API error: {str(e)}")

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
        """Check if Ollama server is running and model is available.

        Returns:
            True if Ollama is running and the configured model exists.
        """
        try:
            with httpx.Client(timeout=5.0) as client:
                # Check if server is running
                response = client.get(f"{self.config.base_url}/api/tags")
                if response.status_code != 200:
                    return False

                # Check if model is available
                data = response.json()
                models = [m.get("name", "") for m in data.get("models", [])]
                # Ollama model names can have :latest suffix
                model_name = self.config.model
                return any(
                    m == model_name or m.startswith(f"{model_name}:")
                    for m in models
                )
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """List installed models from Ollama.

        Queries the local Ollama server to get installed models.
        Unlike cloud providers, this returns only models the user
        has explicitly pulled/installed.

        Returns:
            List of model identifiers (e.g., ['llama3.2:1b', ...]).

        Raises:
            ValueError: If Ollama server is not running.
        """
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self.config.base_url}/api/tags")
                response.raise_for_status()
                data = response.json()

                models = [m.get("name", "") for m in data.get("models", [])]
                return sorted(models)

        except httpx.ConnectError:
            raise ValueError(
                "Ollama server not running. Start with: ollama serve"
            )
        except Exception as e:
            raise ValueError(f"Failed to list Ollama models: {e}")
