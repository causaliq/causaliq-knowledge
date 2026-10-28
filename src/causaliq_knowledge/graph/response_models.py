"""Data models for LLM graph generation responses.

This module holds the Pydantic and dataclass models that describe proposed
edges, generated graphs, generation metadata, and PDG generation results.
The parsing functions that build these models live in
`causaliq_knowledge.graph.response_parsing`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from causaliq_core.graph.pdg import PDG
from pydantic import BaseModel, Field, field_validator


class ProposedEdge(BaseModel):
    """A proposed causal edge from LLM graph generation.

    Represents a single directed edge in the proposed causal graph,
    with confidence score and optional reasoning.

    Attributes:
        source: The name of the source variable (cause).
        target: The name of the target variable (effect).
        confidence: Confidence score from 0.0 to 1.0.
        reasoning: Optional explanation for this specific edge.

    Example:
        >>> edge = ProposedEdge(
        ...     source="smoking",
        ...     target="lung_cancer",
        ...     confidence=0.95,
        ... )
        >>> print(f"{edge.source} -> {edge.target}: {edge.confidence}")
        smoking -> lung_cancer: 0.95
    """

    source: str = Field(..., description="Source variable name (cause)")
    target: str = Field(..., description="Target variable name (effect)")
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence score from 0.0 to 1.0",
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Optional explanation for this edge",
    )

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, v: Any) -> float:
        """Clamp confidence values to [0.0, 1.0] range."""
        if v is None:
            return 0.5
        try:
            val = float(v)
            return max(0.0, min(1.0, val))
        except (TypeError, ValueError):
            return 0.5


@dataclass
class GenerationMetadata:
    """Metadata about a graph generation request.

    Attributes:
        model: The LLM model used for generation.
        provider: The LLM provider (e.g., "groq", "gemini").
        timestamp: When this request was made (current request time).
        llm_timestamp: When the LLM originally responded (from cache if maybe).
        llm_latency_ms: Original LLM response latency in milliseconds.
        input_tokens: Number of input tokens used.
        output_tokens: Number of output tokens generated.
        from_cache: Whether the response was from cache.
        messages: The messages sent to the LLM.
        temperature: Sampling temperature used.
        max_tokens: Maximum tokens requested.
        finish_reason: Why generation stopped (stop, length, etc.).
        llm_cost_usd: Cost when the LLM request was originally made.
    """

    model: str
    provider: str = ""
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    llm_timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    llm_latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    from_cache: bool = False
    messages: List[Dict[str, Any]] = field(default_factory=list)
    temperature: float = 0.1
    max_tokens: int = 2000
    finish_reason: str = "stop"
    llm_cost_usd: float = 0.0

    # Backward compatibility aliases
    @property
    def initial_cost_usd(self) -> float:
        """Alias for llm_cost_usd (backward compatibility)."""
        return self.llm_cost_usd

    @property
    def latency_ms(self) -> int:
        """Alias for llm_latency_ms (backward compatibility)."""
        return self.llm_latency_ms

    @staticmethod
    def _split_message_content(
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Split message content strings at newlines for readability.

        Converts content strings containing newlines into lists of strings,
        making the JSON output more human-readable.

        Args:
            messages: List of message dicts with "role" and "content" keys.

        Returns:
            Messages with content split into string arrays where applicable.
        """
        result = []
        for msg in messages:
            new_msg = dict(msg)
            content = new_msg.get("content", "")
            if isinstance(content, str) and "\n" in content:
                new_msg["content"] = content.split("\n")
            result.append(new_msg)
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Convert generation metadata to a dictionary.

        Returns a dictionary suitable for JSON serialisation, containing
        all generation provenance information.

        Returns:
            Dictionary with all metadata fields.
        """
        return {
            "llm_model": self.model,
            "llm_provider": self.provider,
            "timestamp": self.timestamp.isoformat(),
            "llm_timestamp": self.llm_timestamp.isoformat(),
            "llm_latency_ms": self.llm_latency_ms,
            "llm_input_tokens": self.input_tokens,
            "llm_output_tokens": self.output_tokens,
            "from_cache": self.from_cache,
            "llm_messages": self._split_message_content(self.messages),
            "llm_temperature": self.temperature,
            "llm_max_tokens": self.max_tokens,
            "llm_finish_reason": self.finish_reason,
            "llm_cost_usd": self.llm_cost_usd,
        }


@dataclass
class GeneratedGraph:
    """A complete causal graph generated by an LLM.

    Represents the full output from an LLM graph generation query,
    including all proposed edges, metadata, and the LLM's reasoning.

    Attributes:
        edges: List of proposed causal edges.
        variables: List of variable names in the graph.
        reasoning: Overall reasoning provided by the LLM.
        metadata: Generation metadata (model, timing, etc.).
        raw_response: The original LLM response for debugging.

    Example:
        >>> edge1 = ProposedEdge(
        ...     source="age", target="income", confidence=0.7
        ... )
        >>> edge2 = ProposedEdge(
        ...     source="education", target="income", confidence=0.9
        ... )
        >>> graph = GeneratedGraph(
        ...     edges=[edge1, edge2],
        ...     variables=["age", "education", "income"],
        ...     reasoning="Age and education both influence income.",
        ...     metadata=GenerationMetadata(model="llama-3.1-8b-instant"),
        ... )
        >>> print(f"Generated {len(graph.edges)} edges")
        Generated 2 edges
    """

    edges: List[ProposedEdge]
    variables: List[str]
    reasoning: str = ""
    metadata: Optional[GenerationMetadata] = None
    raw_response: Optional[Dict[str, Any]] = field(default=None, repr=False)

    def get_adjacency_matrix(self) -> List[List[float]]:
        """Convert edges to an adjacency matrix.

        Creates a square matrix where entry (i,j) represents the
        confidence that variable i causes variable j.

        Returns:
            Square matrix of confidence scores.
        """
        n = len(self.variables)
        var_to_idx = {var: i for i, var in enumerate(self.variables)}
        matrix = [[0.0] * n for _ in range(n)]

        for edge in self.edges:
            src_idx = var_to_idx.get(edge.source)
            tgt_idx = var_to_idx.get(edge.target)
            if src_idx is not None and tgt_idx is not None:
                matrix[src_idx][tgt_idx] = edge.confidence

        return matrix

    def get_edge_list(self) -> List[tuple[str, str, float]]:
        """Get edges as a list of tuples.

        Returns:
            List of (source, target, confidence) tuples.
        """
        return [
            (edge.source, edge.target, edge.confidence) for edge in self.edges
        ]

    def filter_by_confidence(self, threshold: float = 0.5) -> "GeneratedGraph":
        """Return a new graph with only edges above the threshold.

        Args:
            threshold: Minimum confidence score to include.

        Returns:
            New GeneratedGraph with filtered edges.
        """
        filtered_edges = [e for e in self.edges if e.confidence >= threshold]
        return GeneratedGraph(
            edges=filtered_edges,
            variables=self.variables,
            reasoning=self.reasoning,
            metadata=self.metadata,
            raw_response=self.raw_response,
        )


@dataclass
class PDGGenerationResult:
    """Result of PDG generation including graph and metadata.

    Combines the generated PDG with comprehensive generation metadata
    for full provenance tracking.

    Attributes:
        pdg: The generated Probabilistic Dependency Graph.
        metadata: Generation metadata (model, timing, cost, etc.).
        raw_response: The original LLM response for debugging.
    """

    pdg: PDG
    metadata: GenerationMetadata
    raw_response: Optional[str] = field(default=None, repr=False)
