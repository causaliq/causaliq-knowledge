"""Unit tests for core models."""

import pytest

from causaliq_knowledge.models import EdgeDirection, EdgeKnowledge


class TestEdgeDirection:
    """Tests for EdgeDirection enum."""

    def test_values(self):
        """Test enum values are as expected."""
        assert EdgeDirection.A_TO_B.value == "a_to_b"
        assert EdgeDirection.B_TO_A.value == "b_to_a"
        assert EdgeDirection.UNDIRECTED.value == "undirected"

    def test_string_comparison(self):
        """Test enum can be compared with strings."""
        assert EdgeDirection.A_TO_B == "a_to_b"
        assert EdgeDirection.B_TO_A == "b_to_a"


class TestEdgeKnowledge:
    """Tests for EdgeKnowledge model."""

    def test_basic_creation(self):
        """Test creating an EdgeKnowledge instance."""
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

    def test_creation_with_string_direction(self):
        """Test creating with string direction (auto-converted to enum)."""
        knowledge = EdgeKnowledge(
            exists=True,
            direction="a_to_b",
            confidence=0.9,
            reasoning="Test",
        )
        assert knowledge.direction == EdgeDirection.A_TO_B

    def test_creation_with_uppercase_direction(self):
        """Test direction validator handles uppercase."""
        knowledge = EdgeKnowledge(
            exists=True,
            direction="A_TO_B",
            confidence=0.9,
            reasoning="Test",
        )
        assert knowledge.direction == EdgeDirection.A_TO_B

    def test_defaults(self):
        """Test default values."""
        knowledge = EdgeKnowledge()
        assert knowledge.exists is None
        assert knowledge.direction is None
        assert knowledge.confidence == 0.0
        assert knowledge.reasoning == ""
        assert knowledge.model is None

    def test_confidence_bounds_valid(self):
        """Test confidence accepts values in [0, 1]."""
        assert EdgeKnowledge(confidence=0.0).confidence == 0.0
        assert EdgeKnowledge(confidence=0.5).confidence == 0.5
        assert EdgeKnowledge(confidence=1.0).confidence == 1.0

    def test_confidence_below_zero_raises(self):
        """Test confidence below 0 raises validation error."""
        with pytest.raises(ValueError):
            EdgeKnowledge(confidence=-0.1)

    def test_confidence_above_one_raises(self):
        """Test confidence above 1 raises validation error."""
        with pytest.raises(ValueError):
            EdgeKnowledge(confidence=1.1)

    def test_is_uncertain_when_exists_none(self):
        """Test is_uncertain returns True when exists is None."""
        knowledge = EdgeKnowledge(exists=None, confidence=0.8)
        assert knowledge.is_uncertain() is True

    def test_is_uncertain_when_low_confidence(self):
        """Test is_uncertain returns True when confidence < 0.5."""
        knowledge = EdgeKnowledge(exists=True, confidence=0.4)
        assert knowledge.is_uncertain() is True

    def test_is_uncertain_false_when_confident(self):
        """Test is_uncertain returns False when confident."""
        knowledge = EdgeKnowledge(exists=True, confidence=0.5)
        assert knowledge.is_uncertain() is False

        knowledge = EdgeKnowledge(exists=False, confidence=0.8)
        assert knowledge.is_uncertain() is False

    def test_to_dict(self):
        """Test to_dict method."""
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

    def test_to_dict_with_none_direction(self):
        """Test to_dict when direction is None."""
        knowledge = EdgeKnowledge(exists=None, confidence=0.0)
        result = knowledge.to_dict()
        assert result["direction"] is None

    def test_uncertain_factory(self):
        """Test uncertain() class method."""
        knowledge = EdgeKnowledge.uncertain()
        assert knowledge.exists is None
        assert knowledge.direction is None
        assert knowledge.confidence == 0.0
        assert knowledge.reasoning == "Unable to determine"
        assert knowledge.model is None

    def test_uncertain_factory_with_args(self):
        """Test uncertain() with custom reasoning and model."""
        knowledge = EdgeKnowledge.uncertain(
            reasoning="API error occurred",
            model="gpt-4o-mini",
        )
        assert knowledge.exists is None
        assert knowledge.confidence == 0.0
        assert knowledge.reasoning == "API error occurred"
        assert knowledge.model == "gpt-4o-mini"

    def test_invalid_direction_raises(self):
        """Test invalid direction string raises error."""
        with pytest.raises(ValueError):
            EdgeKnowledge(direction="invalid_direction")

    def test_invalid_direction_type_raises(self):
        """Test non-string direction type raises error."""
        with pytest.raises(ValueError, match="Invalid direction"):
            EdgeKnowledge(direction=123)

    def test_model_dump_json_serializable(self):
        """Test model can be serialized to JSON via Pydantic."""
        knowledge = EdgeKnowledge(
            exists=True,
            direction=EdgeDirection.A_TO_B,
            confidence=0.85,
            reasoning="Test",
        )
        json_str = knowledge.model_dump_json()
        assert "exists" in json_str
        assert "true" in json_str.lower()
