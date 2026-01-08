"""Direct Mistral AI API client - OpenAI-compatible API."""

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
class MistralConfig(OpenAICompatConfig):
    """Configuration for Mistral AI API client.

    Extends OpenAICompatConfig with Mistral-specific defaults.

    Attributes:
        model: Mistral model identifier (default: mistral-small-latest).
        temperature: Sampling temperature (default: 0.1).
        max_tokens: Maximum response tokens (default: 500).
        timeout: Request timeout in seconds (default: 30.0).
        api_key: Mistral API key (falls back to MISTRAL_API_KEY env var).
    """

    model: str = "mistral-small-latest"
    temperature: float = 0.1
    max_tokens: int = 500
    timeout: float = 30.0
    api_key: Optional[str] = None

    def __post_init__(self) -> None:
        """Set API key from environment if not provided."""
        if self.api_key is None:
            self.api_key = os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError(
                "MISTRAL_API_KEY environment variable is required"
            )


class MistralClient(OpenAICompatClient):
    """Direct Mistral AI API client.

    Mistral AI is a French company providing high-quality LLMs with an
    OpenAI-compatible API.

    Available models:
        - mistral-small-latest: Fast, cost-effective
        - mistral-medium-latest: Balanced performance
        - mistral-large-latest: Most capable
        - codestral-latest: Optimized for code

    Example:
        >>> config = MistralConfig(model="mistral-small-latest")
        >>> client = MistralClient(config)
        >>> msgs = [{"role": "user", "content": "Hello"}]
        >>> response = client.completion(msgs)
        >>> print(response.content)
    """

    BASE_URL = "https://api.mistral.ai/v1"
    PROVIDER_NAME = "mistral"
    ENV_VAR = "MISTRAL_API_KEY"

    def __init__(self, config: Optional[MistralConfig] = None) -> None:
        """Initialize Mistral client.

        Args:
            config: Mistral configuration. If None, uses defaults with
                   API key from MISTRAL_API_KEY environment variable.
        """
        super().__init__(config)

    def _default_config(self) -> MistralConfig:
        """Return default Mistral configuration."""
        return MistralConfig()

    def _get_pricing(self) -> Dict[str, Dict[str, float]]:
        """Return Mistral pricing per 1M tokens.

        Returns:
            Dict mapping model prefixes to input/output costs.
        """
        # Mistral pricing as of Jan 2025
        return {
            "mistral-large": {"input": 2.00, "output": 6.00},
            "mistral-medium": {"input": 2.70, "output": 8.10},
            "mistral-small": {"input": 0.20, "output": 0.60},
            "codestral": {"input": 0.20, "output": 0.60},
            "open-mistral-nemo": {"input": 0.15, "output": 0.15},
            "open-mixtral-8x22b": {"input": 2.00, "output": 6.00},
            "open-mixtral-8x7b": {"input": 0.70, "output": 0.70},
            "ministral-3b": {"input": 0.04, "output": 0.04},
            "ministral-8b": {"input": 0.10, "output": 0.10},
        }

    def _filter_models(self, models: List[str]) -> List[str]:
        """Filter to Mistral chat models only.

        Args:
            models: List of all model IDs from API.

        Returns:
            Filtered list of Mistral models.
        """
        filtered = []
        for model_id in models:
            # Include mistral and codestral models
            if any(
                prefix in model_id.lower()
                for prefix in ["mistral", "codestral", "ministral", "mixtral"]
            ):
                # Exclude embedding models
                if "embed" in model_id.lower():
                    continue
                filtered.append(model_id)
        return filtered
