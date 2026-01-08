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
from causaliq_knowledge.llm.prompts import EdgeQueryPrompt, parse_edge_response
from causaliq_knowledge.llm.provider import (
    CONSENSUS_STRATEGIES,
    LLMKnowledge,
    highest_confidence,
    weighted_vote,
)

__all__ = [
    # Abstract base
    "BaseLLMClient",
    "LLMConfig",
    "LLMResponse",
    # Anthropic
    "AnthropicClient",
    "AnthropicConfig",
    # Consensus
    "CONSENSUS_STRATEGIES",
    # DeepSeek
    "DeepSeekClient",
    "DeepSeekConfig",
    "EdgeQueryPrompt",
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
    # Provider
    "LLMKnowledge",
    "highest_confidence",
    "parse_edge_response",
    "weighted_vote",
]
