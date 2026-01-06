"""Unit tests for the abstract base class."""

from typing import Optional

import pytest

from causaliq_knowledge.base import KnowledgeProvider
from causaliq_knowledge.models import EdgeDirection, EdgeKnowledge


# Helper mock provider for testing
def _create_mock_provider(default_response=None):
    """Create a mock KnowledgeProvider for testing."""

    class MockKnowledgeProvider(KnowledgeProvider):
        def __init__(self):
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
            self.query_history.append((node_a, node_b, context))
            return self._default_response

    return MockKnowledgeProvider()


# ============================================================================
# KnowledgeProvider tests
# ============================================================================


# Check that KnowledgeProvider cannot be instantiated directly
def test_provider_cannot_instantiate_abstract():
    with pytest.raises(TypeError):
        KnowledgeProvider()


# Check that mock implementation works correctly
def test_provider_mock_implementation():
    provider = _create_mock_provider()
    result = provider.query_edge("smoking", "cancer")

    assert result.exists is True
    assert result.direction == EdgeDirection.A_TO_B
    assert result.confidence == 0.8
    assert len(provider.query_history) == 1
    assert provider.query_history[0] == ("smoking", "cancer", None)


# Check query_edge with context dictionary
def test_provider_query_edge_with_context():
    provider = _create_mock_provider()
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


# Check default query_edges calls query_edge for each pair
def test_provider_query_edges_default_implementation():
    provider = _create_mock_provider()
    edges = [("A", "B"), ("C", "D"), ("E", "F")]

    results = provider.query_edges(edges)

    assert len(results) == 3
    assert len(provider.query_history) == 3
    assert provider.query_history[0] == ("A", "B", None)
    assert provider.query_history[1] == ("C", "D", None)
    assert provider.query_history[2] == ("E", "F", None)


# Check query_edges passes context to all queries
def test_provider_query_edges_with_context():
    provider = _create_mock_provider()
    edges = [("A", "B"), ("C", "D")]
    context = {"domain": "test"}

    provider.query_edges(edges, context)

    assert provider.query_history[0] == ("A", "B", context)
    assert provider.query_history[1] == ("C", "D", context)


# Check name property returns class name
def test_provider_name_property():
    provider = _create_mock_provider()
    assert provider.name == "MockKnowledgeProvider"


# Check mock provider with custom response
def test_provider_custom_response():
    custom_response = EdgeKnowledge(
        exists=False,
        direction=None,
        confidence=0.95,
        reasoning="No causal relationship",
        model="custom-model",
    )
    provider = _create_mock_provider(default_response=custom_response)

    result = provider.query_edge("A", "B")

    assert result.exists is False
    assert result.confidence == 0.95


# Check that subclass without query_edge raises TypeError
def test_provider_subclass_must_implement_query_edge():
    class IncompleteProvider(KnowledgeProvider):
        pass

    with pytest.raises(TypeError):
        IncompleteProvider()


# Check that subclass with query_edge can be instantiated
def test_provider_subclass_with_query_edge_works():
    class CompleteProvider(KnowledgeProvider):
        def query_edge(self, node_a, node_b, context=None):
            return EdgeKnowledge.uncertain()

    provider = CompleteProvider()
    result = provider.query_edge("A", "B")
    assert result.exists is None
