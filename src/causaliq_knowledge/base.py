"""Abstract base class for knowledge providers."""

from abc import ABC, abstractmethod
from typing import Optional

from causaliq_knowledge.models import EdgeKnowledge


class KnowledgeProvider(ABC):
    """Abstract interface for all knowledge sources.

    This is the base class that all knowledge providers must implement.
    Knowledge providers can be LLM-based, rule-based, human-input based,
    or any other source of causal knowledge.

    The primary method is `query_edge()` which asks about the causal
    relationship between two variables.

    Example:
        >>> class MyKnowledgeProvider(KnowledgeProvider):
        ...     def query_edge(self, node_a, node_b, context=None):
        ...         # Implementation here
        ...         return EdgeKnowledge(exists=True, confidence=0.8, ...)
        ...
        >>> provider = MyKnowledgeProvider()
        >>> result = provider.query_edge("smoking", "cancer")
    """

    @abstractmethod
    def query_edge(
        self,
        node_a: str,
        node_b: str,
        context: Optional[dict] = None,
    ) -> EdgeKnowledge:
        """Query whether a causal edge exists between two nodes.

        Args:
            node_a: Name of the first variable.
            node_b: Name of the second variable.
            context: Optional context dictionary that may include:
                - domain: The domain (e.g., "medicine", "economics")
                - descriptions: Dict mapping variable names to descriptions
                - additional_info: Any other relevant context

        Returns:
            EdgeKnowledge with:
                - exists: True, False, or None (uncertain)
                - direction: "a_to_b", "b_to_a", "undirected", or None
                - confidence: 0.0 to 1.0
                - reasoning: Human-readable explanation
                - model: Source identifier (optional)

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        pass

    def query_edges(
        self,
        edges: list[tuple[str, str]],
        context: Optional[dict] = None,
    ) -> list[EdgeKnowledge]:
        """Query multiple edges at once.

        Default implementation calls query_edge for each pair.
        Subclasses may override for batch optimization.

        Args:
            edges: List of (node_a, node_b) tuples to query.
            context: Optional context dictionary (shared across all queries).

        Returns:
            List of EdgeKnowledge results, one per edge pair.
        """
        return [self.query_edge(a, b, context) for a, b in edges]

    @property
    def name(self) -> str:
        """Return the name of this knowledge provider.

        Returns:
            Class name by default. Subclasses may override.
        """
        return self.__class__.__name__
