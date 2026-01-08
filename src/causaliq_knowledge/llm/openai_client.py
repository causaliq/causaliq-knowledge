"""Direct OpenAI API client - clean and reliable."""

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from causaliq_knowledge.llm.openai_compat_client import (
    OpenAICompatClient,
    OpenAICompatConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class OpenAIConfig(OpenAICompatConfig):
    """Configuration for OpenAI API client.

    Extends OpenAICompatConfig with OpenAI-specific defaults.

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


class OpenAIClient(OpenAICompatClient):
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
    PROVIDER_NAME = "openai"
    ENV_VAR = "OPENAI_API_KEY"

    def __init__(self, config: Optional[OpenAIConfig] = None) -> None:
        """Initialize OpenAI client.

        Args:
            config: OpenAI configuration. If None, uses defaults with
                   API key from OPENAI_API_KEY environment variable.
        """
        super().__init__(config)

    def _default_config(self) -> OpenAIConfig:
        """Return default OpenAI configuration."""
        return OpenAIConfig()

    def _get_pricing(self) -> Dict[str, Dict[str, float]]:
        """Return OpenAI pricing per 1M tokens.

        Returns:
            Dict mapping model prefixes to input/output costs.
        """
        # Order matters - more specific prefixes must come first
        return {
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
            "gpt-4o": {"input": 2.50, "output": 10.00},
            "gpt-4-turbo": {"input": 10.00, "output": 30.00},
            "gpt-4": {"input": 30.00, "output": 60.00},
            "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
            "o1-mini": {"input": 3.00, "output": 12.00},
            "o1": {"input": 15.00, "output": 60.00},
        }

    def _filter_models(self, models: List[str]) -> List[str]:
        """Filter to OpenAI chat models only.

        Args:
            models: List of all model IDs from API.

        Returns:
            Filtered list of GPT and o1/o3 models.
        """
        filtered = []
        for model_id in models:
            # Include GPT and o1/o3 models
            if any(
                prefix in model_id
                for prefix in ["gpt-4", "gpt-3.5", "o1", "o3"]
            ):
                # Exclude instruct variants and specific exclusions
                if any(
                    x in model_id.lower()
                    for x in ["instruct", "vision", "audio", "realtime"]
                ):
                    continue
                filtered.append(model_id)
        return filtered
