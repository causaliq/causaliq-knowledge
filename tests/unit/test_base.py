"""Unit tests for the abstract base class."""

from typing import Optional

import pytest

from causaliq_knowledge.base import KnowledgeProvider
from causaliq_knowledge.models import EdgeDirection, EdgeKnowledge


class MockKnowledgeProvider(KnowledgeProvider):
    """A mock implementation for testing the abstract interface."""

    def __init__(self, default_response: Optional[EdgeKnowledge] = None):
        self._default_response = default_response or EdgeKnowledge(
            exists=True,
            direction=EdgeDirection.A_TO_B,
            confidence=0.8,
            reasoning="Mock response",
            model="mock-model",
        )
        self.query_history: list[tuple[str, str, Optional[dict]]] = []

    def query_edge(
        self,
        node_a: str,
        node_b: str,
        context: Optional[dict] = None,
    ) -> EdgeKnowledge:
        """Mock implementation that records queries and returns default."""
        self.query_history.append((node_a, node_b, context))
        return self._default_response


class TestKnowledgeProvider:
    """Tests for the KnowledgeProvider abstract base class."""

    def test_cannot_instantiate_abstract(self):
        """Test that KnowledgeProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            KnowledgeProvider()

    def test_mock_implementation(self):
        """Test that mock implementation works."""
        provider = MockKnowledgeProvider()
        result = provider.query_edge("smoking", "cancer")

        assert result.exists is True
        assert result.direction == EdgeDirection.A_TO_B
        assert result.confidence == 0.8
        assert len(provider.query_history) == 1
        assert provider.query_history[0] == ("smoking", "cancer", None)

    def test_query_edge_with_context(self):
        """Test query_edge with context dictionary."""
        provider = MockKnowledgeProvider()
        context = {
            "domain": "medicine",
            "descriptions": {
                "smoking": "Tobacco consumption",
                "cancer": "Lung cancer diagnosis",
            },
        }
        result = provider.query_edge("smoking", "cancer", context)

        assert result is not None
        assert provider.query_history[0] == ("smoking", "cancer", context)

    def test_query_edges_default_implementation(self):
        """Test default query_edges calls query_edge for each pair."""
        provider = MockKnowledgeProvider()
        edges = [("A", "B"), ("C", "D"), ("E", "F")]

        results = provider.query_edges(edges)

        assert len(results) == 3
        assert len(provider.query_history) == 3
        assert provider.query_history[0] == ("A", "B", None)
        assert provider.query_history[1] == ("C", "D", None)
        assert provider.query_history[2] == ("E", "F", None)

    def test_query_edges_with_context(self):
        """Test query_edges passes context to all queries."""
        provider = MockKnowledgeProvider()
        edges = [("A", "B"), ("C", "D")]
        context = {"domain": "test"}

        provider.query_edges(edges, context)

        assert provider.query_history[0] == ("A", "B", context)
        assert provider.query_history[1] == ("C", "D", context)

    def test_name_property(self):
        """Test name property returns class name."""
        provider = MockKnowledgeProvider()
        assert provider.name == "MockKnowledgeProvider"

    def test_custom_response(self):
        """Test mock provider with custom response."""
        custom_response = EdgeKnowledge(
            exists=False,
            direction=None,
            confidence=0.95,
            reasoning="No causal relationship",
            model="custom-model",
        )
        provider = MockKnowledgeProvider(default_response=custom_response)

        result = provider.query_edge("A", "B")

        assert result.exists is False
        assert result.confidence == 0.95


class TestKnowledgeProviderSubclass:
    """Tests to verify subclass requirements."""

    def test_subclass_must_implement_query_edge(self):
        """Test that subclass without query_edge raises TypeError."""

        class IncompleteProvider(KnowledgeProvider):
            pass

        with pytest.raises(TypeError):
            IncompleteProvider()

    def test_subclass_with_query_edge_works(self):
        """Test that subclass with query_edge can be instantiated."""

        class CompleteProvider(KnowledgeProvider):
            def query_edge(self, node_a, node_b, context=None):
                return EdgeKnowledge.uncertain()

        provider = CompleteProvider()
        result = provider.query_edge("A", "B")
        assert result.exists is None
