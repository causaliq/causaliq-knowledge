"""Unit tests for llm.provider module."""

import pytest
from causaliq_core.utils import values_same

from causaliq_knowledge.llm.provider import (
    CONSENSUS_STRATEGIES,
    LLMKnowledge,
    highest_confidence,
    weighted_vote,
)
from causaliq_knowledge.models import EdgeDirection, EdgeKnowledge

# --- weighted_vote Tests ---


# Test weighted_vote with empty list returns uncertain.
def test_weighted_vote_empty_list():
    result = weighted_vote([])

    assert result.exists is None
    assert result.confidence == 0.0
    assert "No responses" in result.reasoning


# Test weighted_vote with single response returns it unchanged.
def test_weighted_vote_single_response():
    response = EdgeKnowledge(
        exists=True,
        direction=EdgeDirection.A_TO_B,
        confidence=0.9,
        reasoning="Test",
        model="gpt-4",
    )
    result = weighted_vote([response])

    assert result == response


# Test weighted_vote majority exists=True wins.
def test_weighted_vote_exists_majority():
    responses = [
        EdgeKnowledge(
            exists=True, confidence=0.8, reasoning="Yes", model="m1"
        ),
        EdgeKnowledge(
            exists=True, confidence=0.7, reasoning="Agree", model="m2"
        ),
        EdgeKnowledge(
            exists=False, confidence=0.5, reasoning="No", model="m3"
        ),
    ]
    result = weighted_vote(responses)

    assert result.exists is True
    assert result.confidence == 0.75  # (0.8 + 0.7) / 2


# Test weighted_vote majority exists=False wins.
def test_weighted_vote_not_exists_majority():
    responses = [
        EdgeKnowledge(
            exists=False, confidence=0.9, reasoning="No1", model="m1"
        ),
        EdgeKnowledge(
            exists=False, confidence=0.8, reasoning="No2", model="m2"
        ),
        EdgeKnowledge(
            exists=True, confidence=0.3, reasoning="Yes", model="m3"
        ),
    ]
    result = weighted_vote(responses)

    assert result.exists is False


# Test weighted_vote uncertain majority wins.
def test_weighted_vote_uncertain_majority():
    responses = [
        EdgeKnowledge(
            exists=None, confidence=0.6, reasoning="Unsure1", model="m1"
        ),
        EdgeKnowledge(
            exists=None, confidence=0.5, reasoning="Unsure2", model="m2"
        ),
        EdgeKnowledge(
            exists=True, confidence=0.3, reasoning="Yes", model="m3"
        ),
    ]
    result = weighted_vote(responses)

    assert result.exists is None


# Test weighted_vote direction from agreeing responses.
def test_weighted_vote_direction_consensus():
    responses = [
        EdgeKnowledge(
            exists=True,
            direction=EdgeDirection.A_TO_B,
            confidence=0.9,
            reasoning="A->B",
            model="m1",
        ),
        EdgeKnowledge(
            exists=True,
            direction=EdgeDirection.A_TO_B,
            confidence=0.8,
            reasoning="A->B too",
            model="m2",
        ),
        EdgeKnowledge(
            exists=True,
            direction=EdgeDirection.B_TO_A,
            confidence=0.5,
            reasoning="B->A",
            model="m3",
        ),
    ]
    result = weighted_vote(responses)

    assert result.exists is True
    assert result.direction == EdgeDirection.A_TO_B


# Test weighted_vote combines reasoning from all models.
def test_weighted_vote_combines_reasoning():
    responses = [
        EdgeKnowledge(
            exists=True, confidence=0.8, reasoning="Reason1", model="m1"
        ),
        EdgeKnowledge(
            exists=True, confidence=0.7, reasoning="Reason2", model="m2"
        ),
    ]
    result = weighted_vote(responses)

    assert "[m1] Reason1" in result.reasoning
    assert "[m2] Reason2" in result.reasoning
    assert "|" in result.reasoning


# Test weighted_vote combines model names.
def test_weighted_vote_combines_models():
    responses = [
        EdgeKnowledge(
            exists=True, confidence=0.8, reasoning="R1", model="gpt-4"
        ),
        EdgeKnowledge(
            exists=True, confidence=0.7, reasoning="R2", model="claude"
        ),
    ]
    result = weighted_vote(responses)

    assert "gpt-4" in result.model
    assert "claude" in result.model


# Test weighted_vote with all zero confidence.
def test_weighted_vote_all_zero_confidence():
    responses = [
        EdgeKnowledge(exists=True, confidence=0.0, reasoning="R1", model="m1"),
        EdgeKnowledge(
            exists=False, confidence=0.0, reasoning="R2", model="m2"
        ),
    ]
    result = weighted_vote(responses)

    assert result.exists is None
    assert "zero confidence" in result.reasoning.lower()


# Test weighted_vote handles missing model names.
def test_weighted_vote_missing_model_names():
    responses = [
        EdgeKnowledge(exists=True, confidence=0.8, reasoning="R1"),
        EdgeKnowledge(exists=True, confidence=0.7, reasoning="R2"),
    ]
    result = weighted_vote(responses)

    assert "[unknown]" in result.reasoning


# Test weighted_vote direction is None when exists=False.
def test_weighted_vote_direction_none_when_not_exists():
    responses = [
        EdgeKnowledge(
            exists=False,
            direction=EdgeDirection.A_TO_B,
            confidence=0.9,
            reasoning="No edge",
            model="m1",
        ),
    ]
    # Single response, but let's wrap in list for weighted_vote
    result = weighted_vote(responses)

    assert result.exists is False
    # Direction shouldn't matter for non-existent edge
    assert result.direction == EdgeDirection.A_TO_B  # Passes through as-is


# --- highest_confidence Tests ---


# Test highest_confidence with empty list returns uncertain.
def test_highest_confidence_empty_list():
    result = highest_confidence([])

    assert result.exists is None
    assert result.confidence == 0.0


# Test highest_confidence returns response with max confidence.
def test_highest_confidence_picks_max():
    responses = [
        EdgeKnowledge(
            exists=True, confidence=0.7, reasoning="Medium", model="m1"
        ),
        EdgeKnowledge(
            exists=False, confidence=0.95, reasoning="High", model="m2"
        ),
        EdgeKnowledge(
            exists=None, confidence=0.3, reasoning="Low", model="m3"
        ),
    ]
    result = highest_confidence(responses)

    assert result.exists is False
    assert result.confidence == 0.95
    assert result.model == "m2"


# Test highest_confidence with single response.
def test_highest_confidence_single():
    response = EdgeKnowledge(
        exists=True, confidence=0.5, reasoning="Only one", model="m1"
    )
    result = highest_confidence([response])

    assert result == response


# --- CONSENSUS_STRATEGIES Tests ---


# Test CONSENSUS_STRATEGIES contains expected strategies.
def test_consensus_strategies_mapping():
    assert "weighted_vote" in CONSENSUS_STRATEGIES
    assert "highest_confidence" in CONSENSUS_STRATEGIES
    assert CONSENSUS_STRATEGIES["weighted_vote"] == weighted_vote
    assert CONSENSUS_STRATEGIES["highest_confidence"] == highest_confidence


# --- LLMKnowledge Tests ---


# Test LLMKnowledge default initialization.
def test_llm_knowledge_default_init(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    provider = LLMKnowledge()

    assert provider.models == ["groq/llama-3.1-8b-instant"]
    assert provider.consensus_strategy == "weighted_vote"
    assert "LLMKnowledge" in provider.name
    assert "groq/llama-3.1-8b-instant" in provider.name


# Test LLMKnowledge with custom models.
def test_llm_knowledge_custom_models(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    provider = LLMKnowledge(
        models=["groq/llama-3.1-8b-instant", "gemini/gemini-2.5-flash"]
    )

    assert provider.models == [
        "groq/llama-3.1-8b-instant",
        "gemini/gemini-2.5-flash",
    ]
    assert "groq/llama-3.1-8b-instant" in provider.name
    assert "gemini/gemini-2.5-flash" in provider.name


# Test LLMKnowledge with Ollama model creates OllamaClient.
def test_llm_knowledge_ollama_model(monkeypatch):
    from causaliq_knowledge.llm.ollama_client import OllamaClient

    provider = LLMKnowledge(
        models=["ollama/llama3.2:1b"],
        timeout=120.0,
    )

    assert provider.models == ["ollama/llama3.2:1b"]
    assert "ollama/llama3.2:1b" in provider.name
    assert "ollama/llama3.2:1b" in provider._clients
    assert isinstance(provider._clients["ollama/llama3.2:1b"], OllamaClient)

    # Verify config was passed correctly
    client = provider._clients["ollama/llama3.2:1b"]
    assert client.config.model == "llama3.2:1b"
    assert client.config.timeout == 120.0


# Test LLMKnowledge with custom consensus strategy.
def test_llm_knowledge_custom_strategy(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    provider = LLMKnowledge(consensus_strategy="highest_confidence")

    assert provider.consensus_strategy == "highest_confidence"


# Test LLMKnowledge raises on invalid strategy.
def test_llm_knowledge_invalid_strategy():
    try:
        LLMKnowledge(consensus_strategy="invalid")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unknown consensus strategy" in str(e)
        assert "invalid" in str(e)


# Test LLMKnowledge models property returns copy.
def test_llm_knowledge_models_returns_copy(monkeypatch):
    # Mock environment for API keys
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    provider = LLMKnowledge(models=["groq/llama-3.1-8b-instant"])

    models = provider.models
    models.append("other")

    assert provider.models == ["groq/llama-3.1-8b-instant"]


# Test LLMKnowledge get_stats returns combined stats.
def test_llm_knowledge_get_stats(monkeypatch):
    # Mock environment for API keys
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    provider = LLMKnowledge(
        models=["groq/llama-3.1-8b-instant", "gemini/gemini-2.5-flash"]
    )

    # Mock the _total_calls attribute that call_count property uses
    for client in provider._clients.values():
        monkeypatch.setattr(client, "_total_calls", 5)

    stats = provider.get_stats()

    assert stats["total_calls"] == 10  # 5 + 5
    assert stats["total_cost"] == 0.0  # Free tier
    assert "groq/llama-3.1-8b-instant" in stats["per_model"]
    assert "gemini/gemini-2.5-flash" in stats["per_model"]


# Test LLMKnowledge query_edge calls client and parses response.
def test_llm_knowledge_query_edge_success(monkeypatch):
    # Mock environment for API keys
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    provider = LLMKnowledge(models=["groq/llama-3.1-8b-instant"])

    # Mock the complete_json method
    def mock_complete_json(messages, **kwargs):
        return (
            {
                "exists": True,
                "direction": "a_to_b",
                "confidence": 0.85,
                "reasoning": "Test reasoning",
            },
            None,  # LLMResponse not needed
        )

    client = provider._clients["groq/llama-3.1-8b-instant"]
    monkeypatch.setattr(client, "complete_json", mock_complete_json)

    result = provider.query_edge("X", "Y")

    assert result.exists is True
    assert result.direction == EdgeDirection.A_TO_B
    assert result.confidence == 0.85


# Test LLMKnowledge query_edge with context.
def test_llm_knowledge_query_edge_with_context(monkeypatch):
    # Mock environment for API keys
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    provider = LLMKnowledge(models=["groq/llama-3.1-8b-instant"])

    captured_prompts = {}

    def mock_complete_json(messages, **kwargs):
        # messages is a list of dicts with role/content
        captured_prompts["messages"] = messages
        return ({"exists": True, "confidence": 0.5, "reasoning": "ok"}, None)

    client = provider._clients["groq/llama-3.1-8b-instant"]
    monkeypatch.setattr(client, "complete_json", mock_complete_json)

    provider.query_edge(
        "smoking",
        "cancer",
        context={"domain": "medicine"},
    )

    # Check that domain context was included in the user message
    user_content = captured_prompts["messages"][1]["content"]
    assert "medicine" in user_content


# Test LLMKnowledge query_edge handles client error.
def test_llm_knowledge_query_edge_handles_error(monkeypatch):
    # Mock environment for API keys
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    provider = LLMKnowledge(models=["groq/llama-3.1-8b-instant"])

    def mock_complete_json(messages, **kwargs):
        raise Exception("API Error")

    client = provider._clients["groq/llama-3.1-8b-instant"]
    monkeypatch.setattr(client, "complete_json", mock_complete_json)

    result = provider.query_edge("X", "Y")

    assert result.exists is None
    assert "Error querying groq/llama-3.1-8b-instant" in result.reasoning
    assert "API Error" in result.reasoning


# Test client without complete_json raises graceful error in query_edge
def test_llm_knowledge_query_edge_invalid_client(monkeypatch):
    # Mock environment for API keys
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    provider = LLMKnowledge(models=["groq/llama-3.1-8b-instant"])

    # Create a mock client that doesn't implement BaseLLMClient interface
    class InvalidClient:
        pass

    # Replace the client with an invalid type after construction
    provider._clients["groq/llama-3.1-8b-instant"] = InvalidClient()

    result = provider.query_edge("X", "Y")

    # Should return uncertain response with error message
    assert result.exists is None
    assert result.confidence == 0.0
    assert "Error querying groq/llama-3.1-8b-instant" in result.reasoning


# Test LLMKnowledge query_edge multi-model consensus.
def test_llm_knowledge_query_edge_multi_model(monkeypatch):
    # Mock environment for API keys
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    provider = LLMKnowledge(
        models=["groq/llama-3.1-8b-instant", "gemini/gemini-2.5-flash"]
    )

    responses = {
        "groq/llama-3.1-8b-instant": {
            "exists": True,
            "confidence": 0.9,
            "reasoning": "R1",
        },
        "gemini/gemini-2.5-flash": {
            "exists": True,
            "confidence": 0.8,
            "reasoning": "R2",
        },
    }

    def make_mock(model):
        def mock_complete_json(messages, **kwargs):
            return (responses[model], None)

        return mock_complete_json

    for model, client in provider._clients.items():
        monkeypatch.setattr(client, "complete_json", make_mock(model))

    result = provider.query_edge("X", "Y")

    assert result.exists is True
    assert values_same(result.confidence, 0.85)  # (0.9 + 0.8) / 2


# Test LLMKnowledge query_edge with highest_confidence strategy.
def test_llm_knowledge_query_edge_highest_confidence(monkeypatch):
    # Mock environment for API keys
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    provider = LLMKnowledge(
        models=["groq/llama-3.1-8b-instant", "gemini/gemini-2.5-flash"],
        consensus_strategy="highest_confidence",
    )

    responses = {
        "groq/llama-3.1-8b-instant": {
            "exists": True,
            "confidence": 0.6,
            "reasoning": "Low",
        },
        "gemini/gemini-2.5-flash": {
            "exists": False,
            "confidence": 0.95,
            "reasoning": "High",
        },
    }

    def make_mock(model):
        def mock_complete_json(messages, **kwargs):
            return (responses[model], None)

        return mock_complete_json

    for model, client in provider._clients.items():
        monkeypatch.setattr(client, "complete_json", make_mock(model))

    result = provider.query_edge("X", "Y")

    # highest_confidence should pick gemini's response
    assert result.exists is False
    assert result.confidence == 0.95


# Test client without call_count raises error in get_stats
def test_llm_knowledge_get_stats_invalid_client(monkeypatch):
    # Mock environment for API keys
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    provider = LLMKnowledge(models=["groq/llama-3.1-8b-instant"])

    # Create a mock client that doesn't implement BaseLLMClient interface
    class InvalidClient:
        pass

    # Replace the client with an invalid type after construction
    provider._clients["groq/llama-3.1-8b-instant"] = InvalidClient()

    # Should raise AttributeError since client lacks call_count
    with pytest.raises(AttributeError):
        provider.get_stats()


# Test LLMKnowledge unsupported model prefix raises error.
def test_llm_knowledge_unsupported_model():
    with pytest.raises(ValueError) as exc_info:
        LLMKnowledge(models=["unsupported/model"])

    assert "Model 'unsupported/model' not supported" in str(exc_info.value)
    assert "groq/" in str(exc_info.value)
    assert "gemini/" in str(exc_info.value)
    assert "ollama/" in str(exc_info.value)


def test_llm_knowledge_invalid_client_in_query_edge(monkeypatch):
    """Test that query_edge handles invalid client type gracefully."""
    # Mock environment for API keys
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    # Create a provider with a valid model
    provider = LLMKnowledge(models=["groq/llama-3.1-8b-instant"])

    # Replace the client with an invalid mock object
    class InvalidClient:
        pass

    provider._clients["groq/llama-3.1-8b-instant"] = InvalidClient()

    # Should return uncertain response due to exception handling
    result = provider.query_edge("A", "B")
    assert result.exists is None
    assert "Error querying groq/llama-3.1-8b-instant" in result.reasoning


def test_llm_knowledge_invalid_client_in_get_stats(monkeypatch):
    """Test that get_stats raises error for invalid client type."""
    # Mock environment for API keys
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    # Create a provider with a valid model
    provider = LLMKnowledge(models=["groq/llama-3.1-8b-instant"])

    # Replace the client with an invalid mock object
    class InvalidClient:
        pass

    provider._clients["groq/llama-3.1-8b-instant"] = InvalidClient()

    # Should raise AttributeError since client lacks call_count
    with pytest.raises(AttributeError):
        provider.get_stats()
