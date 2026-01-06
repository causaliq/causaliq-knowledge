"""Pydantic models for causaliq-knowledge."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class EdgeDirection(str, Enum):
    """Direction of a causal edge between two nodes."""

    A_TO_B = "a_to_b"
    B_TO_A = "b_to_a"
    UNDIRECTED = "undirected"


class EdgeKnowledge(BaseModel):
    """Structured knowledge about a potential causal edge.

    This model represents the result of querying a knowledge source
    about whether a causal relationship exists between two variables.

    Attributes:
        exists: Whether a causal edge exists. True, False, or None (uncertain).
        direction: The direction of the causal relationship if it exists.
            "a_to_b" means node_a causes node_b, "b_to_a" means the reverse,
            "undirected" means bidirectional or direction unknown.
        confidence: Confidence score from 0.0 (no confidence) to 1.0 (certain).
        reasoning: Human-readable explanation for the knowledge assessment.
        model: The LLM or knowledge source that provided this response.

    Example:
        >>> knowledge = EdgeKnowledge(
        ...     exists=True,
        ...     direction="a_to_b",
        ...     confidence=0.85,
        ...     reasoning="Smoking causes lung cancer.",
        ...     model="gpt-4o-mini"
        ... )
    """

    exists: Optional[bool] = Field(
        default=None,
        description="Whether a causal edge exists. None = uncertain.",
    )
    direction: Optional[EdgeDirection] = Field(
        default=None,
        description="Direction of the causal relationship if it exists.",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score from 0.0 to 1.0.",
    )
    reasoning: str = Field(
        default="",
        description="Human-readable explanation for the assessment.",
    )
    model: Optional[str] = Field(
        default=None,
        description="The model/source that provided this knowledge.",
    )

    @field_validator("direction", mode="before")
    @classmethod
    def validate_direction(cls, v: Optional[str]) -> Optional[EdgeDirection]:
        """Convert string direction to EdgeDirection enum."""
        if v is None:
            return None
        if isinstance(v, EdgeDirection):
            return v
        if isinstance(v, str):
            # Handle both "a_to_b" and "A_TO_B" formats
            return EdgeDirection(v.lower())
        raise ValueError(f"Invalid direction: {v}")

    def is_uncertain(self) -> bool:
        """Check if this knowledge is uncertain.

        Returns:
            True if exists is None or confidence is below 0.5.
        """
        return self.exists is None or self.confidence < 0.5

    def to_dict(self) -> dict:
        """Convert to dictionary with string direction.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        return {
            "exists": self.exists,
            "direction": self.direction.value if self.direction else None,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "model": self.model,
        }

    @classmethod
    def uncertain(
        cls,
        reasoning: str = "Unable to determine",
        model: Optional[str] = None,
    ) -> "EdgeKnowledge":
        """Create an uncertain EdgeKnowledge instance.

        Useful for error cases or when knowledge source cannot
        provide an answer.

        Args:
            reasoning: Explanation for why the result is uncertain.
            model: The model/source that was queried.

        Returns:
            EdgeKnowledge with exists=None and confidence=0.0.
        """
        return cls(
            exists=None,
            direction=None,
            confidence=0.0,
            reasoning=reasoning,
            model=model,
        )
