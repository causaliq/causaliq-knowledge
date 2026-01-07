"""LLM integration module for causaliq-knowledge."""

from causaliq_knowledge.llm.gemini_client import (
    GeminiClient,
    GeminiConfig,
    GeminiResponse,
)
from causaliq_knowledge.llm.groq_client import (
    GroqClient,
    GroqConfig,
    GroqResponse,
)
from causaliq_knowledge.llm.prompts import EdgeQueryPrompt, parse_edge_response
from causaliq_knowledge.llm.provider import (
    CONSENSUS_STRATEGIES,
    LLMKnowledge,
    highest_confidence,
    weighted_vote,
)

__all__ = [
    "CONSENSUS_STRATEGIES",
    "EdgeQueryPrompt",
    "GeminiClient",
    "GeminiConfig",
    "GeminiResponse",
    "GroqClient",
    "GroqConfig",
    "GroqResponse",
    "LLMKnowledge",
    "highest_confidence",
    "parse_edge_response",
    "weighted_vote",
]
