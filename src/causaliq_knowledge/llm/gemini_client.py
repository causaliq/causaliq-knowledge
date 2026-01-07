"""Direct Google Gemini API client - clean and reliable."""

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class GeminiConfig:
    """Configuration for Gemini API client."""

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


@dataclass
class GeminiResponse:
    """Response from Gemini API."""

    content: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0  # Gemini free tier
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
        except json.JSONDecodeError:
            return None


class GeminiClient:
    """Direct Gemini API client."""

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, config: Optional[GeminiConfig] = None):
        """Initialize Gemini client."""
        self.config = config or GeminiConfig()
        self._total_calls = 0

    def completion(
        self, messages: List[Dict[str, str]], **kwargs: Any
    ) -> GeminiResponse:
        """Make a chat completion request to Gemini."""

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

                # Extract usage info
                usage = data.get("usageMetadata", {})
                input_tokens = usage.get("promptTokenCount", 0)
                output_tokens = usage.get("candidatesTokenCount", 0)

                self._total_calls += 1

                logger.debug(
                    f"Gemini response: {input_tokens} in, {output_tokens} out"
                )

                return GeminiResponse(
                    content=content,
                    model=self.config.model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost=0.0,  # Free tier
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
    ) -> tuple[Optional[Dict[str, Any]], GeminiResponse]:
        """Make a completion request and parse response as JSON."""
        response = self.completion(messages, **kwargs)
        parsed = response.parse_json()
        return parsed, response

    @property
    def call_count(self) -> int:
        """Number of API calls made."""
        return self._total_calls
