"""LLM integration module for causaliq-knowledge."""

from causaliq_knowledge.llm.base_client import (
    BaseLLMClient,
    LLMConfig,
    LLMResponse,
)
from causaliq_knowledge.llm.gemini_client import GeminiClient, GeminiConfig
from causaliq_knowledge.llm.groq_client import GroqClient, GroqConfig
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
    # Consensus
    "CONSENSUS_STRATEGIES",
    "EdgeQueryPrompt",
    # Gemini
    "GeminiClient",
    "GeminiConfig",
    # Groq
    "GroqClient",
    "GroqConfig",
    # Provider
    "LLMKnowledge",
    "highest_confidence",
    "parse_edge_response",
    "weighted_vote",
]
