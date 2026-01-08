"""Direct DeepSeek API client - OpenAI-compatible API."""

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
class DeepSeekConfig(OpenAICompatConfig):
    """Configuration for DeepSeek API client.

    Extends OpenAICompatConfig with DeepSeek-specific defaults.

    Attributes:
        model: DeepSeek model identifier (default: deepseek-chat).
        temperature: Sampling temperature (default: 0.1).
        max_tokens: Maximum response tokens (default: 500).
        timeout: Request timeout in seconds (default: 30.0).
        api_key: DeepSeek API key (falls back to DEEPSEEK_API_KEY env var).
    """

    model: str = "deepseek-chat"
    temperature: float = 0.1
    max_tokens: int = 500
    timeout: float = 30.0
    api_key: Optional[str] = None

    def __post_init__(self) -> None:
        """Set API key from environment if not provided."""
        if self.api_key is None:
            self.api_key = os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY environment variable is required"
            )


class DeepSeekClient(OpenAICompatClient):
    """Direct DeepSeek API client.

    DeepSeek uses an OpenAI-compatible API, making integration straightforward.
    Known for excellent reasoning capabilities (R1) at low cost.

    Available models:
        - deepseek-chat: General purpose (DeepSeek-V3)
        - deepseek-reasoner: Advanced reasoning (DeepSeek-R1)

    Example:
        >>> config = DeepSeekConfig(model="deepseek-chat")
        >>> client = DeepSeekClient(config)
        >>> msgs = [{"role": "user", "content": "Hello"}]
        >>> response = client.completion(msgs)
        >>> print(response.content)
    """

    BASE_URL = "https://api.deepseek.com"
    PROVIDER_NAME = "deepseek"
    ENV_VAR = "DEEPSEEK_API_KEY"

    def __init__(self, config: Optional[DeepSeekConfig] = None) -> None:
        """Initialize DeepSeek client.

        Args:
            config: DeepSeek configuration. If None, uses defaults with
                   API key from DEEPSEEK_API_KEY environment variable.
        """
        super().__init__(config)

    def _default_config(self) -> DeepSeekConfig:
        """Return default DeepSeek configuration."""
        return DeepSeekConfig()

    def _get_pricing(self) -> Dict[str, Dict[str, float]]:
        """Return DeepSeek pricing per 1M tokens.

        Returns:
            Dict mapping model prefixes to input/output costs.
        """
        # DeepSeek pricing as of Jan 2025
        # Note: Cache hits are much cheaper but we use regular pricing
        return {
            "deepseek-reasoner": {"input": 0.55, "output": 2.19},
            "deepseek-chat": {"input": 0.14, "output": 0.28},
        }

    def _filter_models(self, models: List[str]) -> List[str]:
        """Filter to DeepSeek chat models only.

        Args:
            models: List of all model IDs from API.

        Returns:
            Filtered list of DeepSeek models.
        """
        filtered = []
        for model_id in models:
            # Include deepseek chat and reasoner models
            if model_id.startswith("deepseek-"):
                filtered.append(model_id)
        return filtered
