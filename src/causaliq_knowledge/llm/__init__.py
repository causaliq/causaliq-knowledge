"""LLM integration module for causaliq-knowledge."""

from causaliq_knowledge.llm.anthropic_client import (
    AnthropicClient,
    AnthropicConfig,
)
from causaliq_knowledge.llm.base_client import (
    BaseLLMClient,
    LLMConfig,
    LLMResponse,
)
from causaliq_knowledge.llm.deepseek_client import (
    DeepSeekClient,
    DeepSeekConfig,
)
from causaliq_knowledge.llm.gemini_client import GeminiClient, GeminiConfig
from causaliq_knowledge.llm.groq_client import GroqClient, GroqConfig
from causaliq_knowledge.llm.mistral_client import MistralClient, MistralConfig
from causaliq_knowledge.llm.ollama_client import OllamaClient, OllamaConfig
from causaliq_knowledge.llm.openai_client import OpenAIClient, OpenAIConfig

__all__ = [
    # Abstract base
    "BaseLLMClient",
    "LLMConfig",
    "LLMResponse",
    # Anthropic
    "AnthropicClient",
    "AnthropicConfig",
    # DeepSeek
    "DeepSeekClient",
    "DeepSeekConfig",
    # Gemini
    "GeminiClient",
    "GeminiConfig",
    # Groq
    "GroqClient",
    "GroqConfig",
    # Mistral
    "MistralClient",
    "MistralConfig",
    # Ollama (local)
    "OllamaClient",
    "OllamaConfig",
    # OpenAI
    "OpenAIClient",
    "OpenAIConfig",
]
