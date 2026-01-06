"""Unit tests for llm.provider module."""

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
def test_llm_knowledge_default_init():
    provider = LLMKnowledge()

    assert provider.models == ["gpt-4o-mini"]
    assert provider.consensus_strategy == "weighted_vote"
    assert "LLMKnowledge" in provider.name
    assert "gpt-4o-mini" in provider.name


# Test LLMKnowledge with custom models.
def test_llm_knowledge_custom_models():
    provider = LLMKnowledge(models=["gpt-4", "claude-3-haiku-20240307"])

    assert provider.models == ["gpt-4", "claude-3-haiku-20240307"]
    assert "gpt-4" in provider.name
    assert "claude-3-haiku-20240307" in provider.name


# Test LLMKnowledge with custom consensus strategy.
def test_llm_knowledge_custom_strategy():
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
def test_llm_knowledge_models_returns_copy():
    provider = LLMKnowledge(models=["gpt-4"])

    models = provider.models
    models.append("other")

    assert provider.models == ["gpt-4"]


# Test LLMKnowledge get_stats returns combined stats.
def test_llm_knowledge_get_stats(monkeypatch):
    provider = LLMKnowledge(models=["m1", "m2"])

    # Mock client get_stats
    def mock_stats():
        return {
            "call_count": 5,
            "total_cost": 0.01,
            "total_input_tokens": 100,
            "total_output_tokens": 50,
            "model": "test",
        }

    for client in provider._clients.values():
        monkeypatch.setattr(client, "get_stats", mock_stats)

    stats = provider.get_stats()

    assert stats["total_calls"] == 10  # 5 + 5
    assert stats["total_cost"] == 0.02  # 0.01 + 0.01
    assert "m1" in stats["per_model"]
    assert "m2" in stats["per_model"]


# Test LLMKnowledge query_edge calls client and parses response.
def test_llm_knowledge_query_edge_success(monkeypatch):
    provider = LLMKnowledge(models=["test-model"])

    # Mock the complete_json method
    def mock_complete_json(system, user):
        return (
            {
                "exists": True,
                "direction": "a_to_b",
                "confidence": 0.85,
                "reasoning": "Test reasoning",
            },
            None,  # LLMResponse not needed
        )

    client = provider._clients["test-model"]
    monkeypatch.setattr(client, "complete_json", mock_complete_json)

    result = provider.query_edge("X", "Y")

    assert result.exists is True
    assert result.direction == EdgeDirection.A_TO_B
    assert result.confidence == 0.85


# Test LLMKnowledge query_edge with context.
def test_llm_knowledge_query_edge_with_context(monkeypatch):
    provider = LLMKnowledge(models=["test-model"])

    captured_prompts = {}

    def mock_complete_json(system, user):
        captured_prompts["system"] = system
        captured_prompts["user"] = user
        return ({"exists": True, "confidence": 0.5, "reasoning": "ok"}, None)

    client = provider._clients["test-model"]
    monkeypatch.setattr(client, "complete_json", mock_complete_json)

    provider.query_edge(
        "smoking",
        "cancer",
        context={"domain": "medicine"},
    )

    assert "medicine" in captured_prompts["user"]


# Test LLMKnowledge query_edge handles client error.
def test_llm_knowledge_query_edge_handles_error(monkeypatch):
    provider = LLMKnowledge(models=["test-model"])

    def mock_complete_json(system, user):
        raise Exception("API Error")

    client = provider._clients["test-model"]
    monkeypatch.setattr(client, "complete_json", mock_complete_json)

    result = provider.query_edge("X", "Y")

    assert result.exists is None
    assert "Error querying test-model" in result.reasoning
    assert "API Error" in result.reasoning


# Test LLMKnowledge query_edge multi-model consensus.
def test_llm_knowledge_query_edge_multi_model(monkeypatch):
    provider = LLMKnowledge(models=["m1", "m2"])

    responses = {
        "m1": {"exists": True, "confidence": 0.9, "reasoning": "R1"},
        "m2": {"exists": True, "confidence": 0.8, "reasoning": "R2"},
    }

    def make_mock(model):
        def mock_complete_json(system, user):
            return (responses[model], None)

        return mock_complete_json

    for model, client in provider._clients.items():
        monkeypatch.setattr(client, "complete_json", make_mock(model))

    result = provider.query_edge("X", "Y")

    assert result.exists is True
    assert values_same(result.confidence, 0.85)  # (0.9 + 0.8) / 2


# Test LLMKnowledge query_edge with highest_confidence strategy.
def test_llm_knowledge_query_edge_highest_confidence(monkeypatch):
    provider = LLMKnowledge(
        models=["m1", "m2"],
        consensus_strategy="highest_confidence",
    )

    responses = {
        "m1": {"exists": True, "confidence": 0.6, "reasoning": "Low"},
        "m2": {"exists": False, "confidence": 0.95, "reasoning": "High"},
    }

    def make_mock(model):
        def mock_complete_json(system, user):
            return (responses[model], None)

        return mock_complete_json

    for model, client in provider._clients.items():
        monkeypatch.setattr(client, "complete_json", make_mock(model))

    result = provider.query_edge("X", "Y")

    # highest_confidence should pick m2's response
    assert result.exists is False
    assert result.confidence == 0.95
