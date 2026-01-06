"""Unit tests for core models."""

import pytest

from causaliq_knowledge.models import EdgeDirection, EdgeKnowledge

# ============================================================================
# EdgeDirection tests
# ============================================================================


# Check enum values are as expected
def test_direction_values():
    assert EdgeDirection.A_TO_B.value == "a_to_b"
    assert EdgeDirection.B_TO_A.value == "b_to_a"
    assert EdgeDirection.UNDIRECTED.value == "undirected"


# Check enum can be compared with strings
def test_direction_string_comparison():
    assert EdgeDirection.A_TO_B == "a_to_b"
    assert EdgeDirection.B_TO_A == "b_to_a"


# ============================================================================
# EdgeKnowledge tests
# ============================================================================


# Check creating an EdgeKnowledge instance with all fields
def test_knowledge_basic_creation():
    knowledge = EdgeKnowledge(
        exists=True,
        direction=EdgeDirection.A_TO_B,
        confidence=0.85,
        reasoning="Smoking causes cancer",
        model="gpt-4o-mini",
    )
    assert knowledge.exists is True
    assert knowledge.direction == EdgeDirection.A_TO_B
    assert knowledge.confidence == 0.85
    assert knowledge.reasoning == "Smoking causes cancer"
    assert knowledge.model == "gpt-4o-mini"


# Check creating with string direction auto-converts to enum
def test_knowledge_creation_with_string_direction():
    knowledge = EdgeKnowledge(
        exists=True,
        direction="a_to_b",
        confidence=0.9,
        reasoning="Test",
    )
    assert knowledge.direction == EdgeDirection.A_TO_B


# Check direction validator handles uppercase strings
def test_knowledge_creation_with_uppercase_direction():
    knowledge = EdgeKnowledge(
        exists=True,
        direction="A_TO_B",
        confidence=0.9,
        reasoning="Test",
    )
    assert knowledge.direction == EdgeDirection.A_TO_B


# Check default values when no arguments provided
def test_knowledge_defaults():
    knowledge = EdgeKnowledge()
    assert knowledge.exists is None
    assert knowledge.direction is None
    assert knowledge.confidence == 0.0
    assert knowledge.reasoning == ""
    assert knowledge.model is None


# Check confidence accepts values in valid range [0, 1]
def test_knowledge_confidence_bounds_valid():
    assert EdgeKnowledge(confidence=0.0).confidence == 0.0
    assert EdgeKnowledge(confidence=0.5).confidence == 0.5
    assert EdgeKnowledge(confidence=1.0).confidence == 1.0


# Check confidence below 0 raises validation error
def test_knowledge_confidence_below_zero_raises():
    with pytest.raises(ValueError):
        EdgeKnowledge(confidence=-0.1)


# Check confidence above 1 raises validation error
def test_knowledge_confidence_above_one_raises():
    with pytest.raises(ValueError):
        EdgeKnowledge(confidence=1.1)


# Check is_uncertain returns True when exists is None
def test_knowledge_is_uncertain_when_exists_none():
    knowledge = EdgeKnowledge(exists=None, confidence=0.8)
    assert knowledge.is_uncertain() is True


# Check is_uncertain returns True when confidence < 0.5
def test_knowledge_is_uncertain_when_low_confidence():
    knowledge = EdgeKnowledge(exists=True, confidence=0.4)
    assert knowledge.is_uncertain() is True


# Check is_uncertain returns False when confident
def test_knowledge_is_uncertain_false_when_confident():
    knowledge = EdgeKnowledge(exists=True, confidence=0.5)
    assert knowledge.is_uncertain() is False

    knowledge = EdgeKnowledge(exists=False, confidence=0.8)
    assert knowledge.is_uncertain() is False


# Check to_dict returns correct dictionary representation
def test_knowledge_to_dict():
    knowledge = EdgeKnowledge(
        exists=True,
        direction=EdgeDirection.A_TO_B,
        confidence=0.85,
        reasoning="Test reasoning",
        model="test-model",
    )
    result = knowledge.to_dict()
    assert result == {
        "exists": True,
        "direction": "a_to_b",
        "confidence": 0.85,
        "reasoning": "Test reasoning",
        "model": "test-model",
    }


# Check to_dict handles None direction correctly
def test_knowledge_to_dict_with_none_direction():
    knowledge = EdgeKnowledge(exists=None, confidence=0.0)
    result = knowledge.to_dict()
    assert result["direction"] is None


# Check uncertain() factory creates correct instance
def test_knowledge_uncertain_factory():
    knowledge = EdgeKnowledge.uncertain()
    assert knowledge.exists is None
    assert knowledge.direction is None
    assert knowledge.confidence == 0.0
    assert knowledge.reasoning == "Unable to determine"
    assert knowledge.model is None


# Check uncertain() factory with custom reasoning and model
def test_knowledge_uncertain_factory_with_args():
    knowledge = EdgeKnowledge.uncertain(
        reasoning="API error occurred",
        model="gpt-4o-mini",
    )
    assert knowledge.exists is None
    assert knowledge.confidence == 0.0
    assert knowledge.reasoning == "API error occurred"
    assert knowledge.model == "gpt-4o-mini"


# Check invalid direction string raises error
def test_knowledge_invalid_direction_raises():
    with pytest.raises(ValueError):
        EdgeKnowledge(direction="invalid_direction")


# Check non-string direction type raises error
def test_knowledge_invalid_direction_type_raises():
    with pytest.raises(ValueError, match="Invalid direction"):
        EdgeKnowledge(direction=123)


# Check model can be serialized to JSON via Pydantic
def test_knowledge_model_dump_json_serializable():
    knowledge = EdgeKnowledge(
        exists=True,
        direction=EdgeDirection.A_TO_B,
        confidence=0.85,
        reasoning="Test",
    )
    json_str = knowledge.model_dump_json()
    assert "exists" in json_str
    assert "true" in json_str.lower()
