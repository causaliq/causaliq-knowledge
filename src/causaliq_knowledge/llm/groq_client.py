"""Direct Groq API client - clean and reliable."""

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class GroqConfig:
    """Configuration for Groq API client."""

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


@dataclass
class GroqResponse:
    """Response from Groq API."""

    content: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0  # Groq free tier
    raw_response: Optional[Dict] = None

    def parse_json(self) -> Optional[Dict[str, Any]]:
        """Parse content as JSON, handling common formatting issues."""
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


class GroqClient:
    """Direct Groq API client."""

    BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(self, config: Optional[GroqConfig] = None):
        """Initialize Groq client."""
        self.config = config or GroqConfig()
        self._total_calls = 0

    def completion(
        self, messages: List[Dict[str, str]], **kwargs: Any
    ) -> GroqResponse:
        """Make a chat completion request to Groq."""

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

                return GroqResponse(
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
    ) -> tuple[Optional[Dict[str, Any]], GroqResponse]:
        """Make a completion request and parse response as JSON."""
        response = self.completion(messages, **kwargs)
        parsed = response.parse_json()
        return parsed, response

    @property
    def call_count(self) -> int:
        """Number of API calls made."""
        return self._total_calls
