"""Prompt templates for LLM edge queries."""

from dataclasses import dataclass
from typing import Optional

from causaliq_knowledge.models import EdgeDirection, EdgeKnowledge

# Default system prompt for edge existence queries
DEFAULT_SYSTEM_PROMPT = """You are an expert in causal reasoning and domain \
knowledge.
Your task is to assess whether a causal relationship exists between two \
variables.

Respond ONLY with valid JSON in this exact format:
{
  "exists": true or false or null,
  "direction": "a_to_b" or "b_to_a" or "undirected" or null,
  "confidence": 0.0 to 1.0,
  "reasoning": "brief explanation"
}

Guidelines:
- exists: true if causal relationship, false if not, null if uncertain
- direction: "a_to_b" means the first variable causes the second
- direction: "b_to_a" means the second variable causes the first
- direction: "undirected" means bidirectional or direction unknown
- confidence: your confidence level from 0.0 (none) to 1.0 (certain)
- reasoning: a brief explanation of your assessment"""

# Template for the user prompt
USER_PROMPT_TEMPLATE = """\
Does a causal relationship exist between "{node_a}" and "{node_b}"?

Consider:
- Direct causation ({node_a} causes {node_b})
- Reverse causation ({node_b} causes {node_a})
- Bidirectional/feedback relationships
- No causal relationship (correlation only or independence)"""

# Template with domain context
USER_PROMPT_WITH_DOMAIN_TEMPLATE = """In the domain of {domain}:

Does a causal relationship exist between "{node_a}" and "{node_b}"?

Consider:
- Direct causation ({node_a} causes {node_b})
- Reverse causation ({node_b} causes {node_a})
- Bidirectional/feedback relationships
- No causal relationship (correlation only or independence)"""

# Template addition for variable descriptions
VARIABLE_DESCRIPTIONS_TEMPLATE = """

Variable descriptions:
- {node_a}: {desc_a}
- {node_b}: {desc_b}"""


@dataclass
class EdgeQueryPrompt:
    """Builder for edge existence/orientation query prompts.

    This class constructs system and user prompts for querying an LLM
    about causal relationships between variables.

    Attributes:
        node_a: Name of the first variable.
        node_b: Name of the second variable.
        domain: Optional domain context (e.g., "medicine", "economics").
        descriptions: Optional dict mapping variable names to descriptions.
        system_prompt: Custom system prompt (uses default if None).

    Example:
        >>> prompt = EdgeQueryPrompt("smoking", "cancer", domain="medicine")
        >>> system, user = prompt.build()
        >>> # Use with LLMClient
        >>> response = client.complete(system=system, user=user)
    """

    node_a: str
    node_b: str
    domain: Optional[str] = None
    descriptions: Optional[dict[str, str]] = None
    system_prompt: Optional[str] = None

    def build(self) -> tuple[str, str]:
        """Build the system and user prompts.

        Returns:
            Tuple of (system_prompt, user_prompt).
        """
        system = self.system_prompt or DEFAULT_SYSTEM_PROMPT

        # Build user prompt
        if self.domain:
            user = USER_PROMPT_WITH_DOMAIN_TEMPLATE.format(
                domain=self.domain,
                node_a=self.node_a,
                node_b=self.node_b,
            )
        else:
            user = USER_PROMPT_TEMPLATE.format(
                node_a=self.node_a,
                node_b=self.node_b,
            )

        # Add variable descriptions if provided
        if self.descriptions:
            desc_a = self.descriptions.get(self.node_a, "No description")
            desc_b = self.descriptions.get(self.node_b, "No description")
            user += VARIABLE_DESCRIPTIONS_TEMPLATE.format(
                node_a=self.node_a,
                desc_a=desc_a,
                node_b=self.node_b,
                desc_b=desc_b,
            )

        return system, user

    @classmethod
    def from_context(
        cls,
        node_a: str,
        node_b: str,
        context: Optional[dict] = None,
    ) -> "EdgeQueryPrompt":
        """Create an EdgeQueryPrompt from a context dictionary.

        This is a convenience method for creating prompts from the
        context dict used by KnowledgeProvider.query_edge().

        Args:
            node_a: Name of the first variable.
            node_b: Name of the second variable.
            context: Optional context dict with keys:
                - domain: str
                - descriptions: dict[str, str]
                - system_prompt: str

        Returns:
            EdgeQueryPrompt instance.
        """
        if context is None:
            return cls(node_a=node_a, node_b=node_b)

        return cls(
            node_a=node_a,
            node_b=node_b,
            domain=context.get("domain"),
            descriptions=context.get("descriptions"),
            system_prompt=context.get("system_prompt"),
        )


def parse_edge_response(
    json_data: Optional[dict],
    model: Optional[str] = None,
) -> EdgeKnowledge:
    """Parse a JSON response dict into an EdgeKnowledge object.

    Args:
        json_data: Parsed JSON dict from LLM response, or None if parsing
            failed.
        model: Optional model identifier to include in the result.

    Returns:
        EdgeKnowledge object. Returns uncertain result if json_data is None
        or missing required fields.
    """
    if json_data is None:
        return EdgeKnowledge.uncertain(
            reasoning="Failed to parse LLM response as JSON",
            model=model,
        )

    # Extract fields with defaults
    exists = json_data.get("exists")
    direction_str = json_data.get("direction")
    confidence = json_data.get("confidence", 0.0)
    reasoning = json_data.get("reasoning", "")

    # Validate confidence is a number
    try:
        confidence = float(confidence)
        confidence = max(0.0, min(1.0, confidence))
    except (TypeError, ValueError):
        confidence = 0.0

    # Convert direction string to enum
    direction = None
    if direction_str:
        try:
            direction = EdgeDirection(direction_str.lower())
        except ValueError:
            # Invalid direction, leave as None
            pass

    return EdgeKnowledge(
        exists=exists,
        direction=direction,
        confidence=confidence,
        reasoning=str(reasoning),
        model=model,
    )
