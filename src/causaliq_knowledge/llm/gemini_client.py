"""Direct Google Gemini API client - clean and reliable."""

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
class GeminiConfig(LLMConfig):
    """Configuration for Gemini API client.

    Extends LLMConfig with Gemini-specific defaults.

    Attributes:
        model: Gemini model identifier (default: gemini-2.5-flash).
        temperature: Sampling temperature (default: 0.1).
        max_tokens: Maximum response tokens (default: 500).
        timeout: Request timeout in seconds (default: 30.0).
        api_key: Gemini API key (falls back to GEMINI_API_KEY env var).
    """

    model: str = "gemini-2.5-flash"
    temperature: float = 0.1
    max_tokens: int = 500
    timeout: float = 30.0
    api_key: Optional[str] = None

    def __post_init__(self) -> None:
        """Set API key from environment if not provided."""
        if self.api_key is None:
            self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")


class GeminiClient(BaseLLMClient):
    """Direct Gemini API client.

    Implements the BaseLLMClient interface for Google's Gemini API.
    Uses httpx for HTTP requests.

    Example:
        >>> config = GeminiConfig(model="gemini-2.5-flash")
        >>> client = GeminiClient(config)
        >>> msgs = [{"role": "user", "content": "Hello"}]
        >>> response = client.completion(msgs)
        >>> print(response.content)
    """

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, config: Optional[GeminiConfig] = None) -> None:
        """Initialize Gemini client.

        Args:
            config: Gemini configuration. If None, uses defaults with
                   API key from GEMINI_API_KEY environment variable.
        """
        self.config = config or GeminiConfig()
        self._total_calls = 0

    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return "gemini"

    def completion(
        self, messages: List[Dict[str, str]], **kwargs: Any
    ) -> LLMResponse:
        """Make a chat completion request to Gemini.

        Args:
            messages: List of message dicts with "role" and "content" keys.
            **kwargs: Override config options (temperature, max_tokens).

        Returns:
            LLMResponse with the generated content and metadata.

        Raises:
            ValueError: If the API request fails.
        """

        # Convert OpenAI-style messages to Gemini format
        contents = []
        system_instruction = None

        for msg in messages:
            if msg["role"] == "system":
                # Gemini handles system prompts differently
                system_instruction = {"parts": [{"text": msg["content"]}]}
            elif msg["role"] == "user":
                contents.append(
                    {"role": "user", "parts": [{"text": msg["content"]}]}
                )
            elif msg["role"] == "assistant":
                contents.append(
                    {"role": "model", "parts": [{"text": msg["content"]}]}
                )

        # Build request payload in Gemini's format
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": kwargs.get(
                    "temperature", self.config.temperature
                ),
                "maxOutputTokens": kwargs.get(
                    "max_tokens", self.config.max_tokens
                ),
                "responseMimeType": "text/plain",
            },
        }

        # Add system instruction if present
        if system_instruction:
            payload["systemInstruction"] = system_instruction

        # API endpoint with model and key
        url = f"{self.BASE_URL}/{self.config.model}:generateContent"
        params = {"key": self.config.api_key}

        headers = {"Content-Type": "application/json"}

        logger.debug(f"Calling Gemini API with model: {self.config.model}")

        try:
            with httpx.Client(timeout=self.config.timeout) as client:
                response = client.post(
                    url, json=payload, headers=headers, params=params
                )
                response.raise_for_status()

                data = response.json()

                # Handle Gemini API errors
                if "error" in data:
                    error_msg = data["error"].get("message", "Unknown error")
                    raise ValueError(f"Gemini API error: {error_msg}")

                # Extract response data from Gemini format
                candidates = data.get("candidates", [])
                if not candidates:
                    raise ValueError("No candidates returned by Gemini API")

                candidate = candidates[0]
                if candidate.get("finishReason") == "SAFETY":
                    raise ValueError(
                        "Content was blocked by Gemini safety filters"
                    )

                # Extract text content
                parts = candidate.get("content", {}).get("parts", [])
                content = ""
                for part in parts:
                    if "text" in part:
                        content += part["text"]

                # Extract usage info and finish reason
                usage = data.get("usageMetadata", {})
                input_tokens = usage.get("promptTokenCount", 0)
                output_tokens = usage.get("candidatesTokenCount", 0)
                # Gemini uses finishReason
                finish_reason = candidate.get("finishReason", "STOP") or "stop"
                # Normalise to lowercase for consistency
                finish_reason = finish_reason.lower()

                self._total_calls += 1

                logger.debug(
                    f"Gemini response: {input_tokens} in, {output_tokens} out"
                )

                return LLMResponse(
                    content=content,
                    model=self.config.model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost=0.0,  # Free tier
                    finish_reason=finish_reason,
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
                f"Gemini API HTTP error: {e.response.status_code} - "
                f"{error_msg}"
            )
            raise ValueError(
                f"Gemini API error: {e.response.status_code} - {error_msg}"
            )
        except httpx.TimeoutException:
            raise ValueError("Gemini API request timed out")
        except Exception as e:
            logger.error(f"Gemini API unexpected error: {e}")
            raise ValueError(f"Gemini API error: {str(e)}")

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
        """Check if Gemini API is available.

        Returns:
            True if GEMINI_API_KEY is configured.
        """
        return bool(self.config.api_key)

    def list_models(self) -> List[str]:
        """List available models from Gemini API.

        Queries the Gemini API to get models accessible with the current
        API key. Filters to only include models that support generateContent.

        Returns:
            List of model identifiers (e.g., ['gemini-2.5-flash', ...]).

        Raises:
            ValueError: If the API request fails.
        """
        try:
            with httpx.Client(timeout=self.config.timeout) as client:
                response = client.get(
                    f"{self.BASE_URL}?key={self.config.api_key}",
                )
                response.raise_for_status()
                data = response.json()

                # Filter to models that support text generation
                models = []
                for model in data.get("models", []):
                    methods = model.get("supportedGenerationMethods", [])
                    if "generateContent" not in methods:
                        continue
                    # Extract model name (remove 'models/' prefix)
                    name = model.get("name", "").replace("models/", "")
                    # Skip embedding and TTS models
                    if any(x in name.lower() for x in ["embed", "tts", "aqa"]):
                        continue
                    models.append(name)

                return sorted(models)

        except httpx.HTTPStatusError as e:
            raise ValueError(
                f"Gemini API error: {e.response.status_code} - "
                f"{e.response.text}"
            )
        except Exception as e:
            raise ValueError(f"Failed to list Gemini models: {e}")
