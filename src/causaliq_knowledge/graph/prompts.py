"""Prompt templates for LLM graph generation queries.

This module provides prompt builders for generating complete causal
graphs from variable specifications, distinct from the edge-by-edge
queries in the llm.prompts module.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:  # pragma: no cover
    from causaliq_knowledge.graph.models import NetworkContext

from causaliq_knowledge.graph.view_filter import PromptDetail, ViewFilter


class OutputFormat(str, Enum):
    """Output format for graph generation responses.

    Determines the structure of the JSON response expected from the LLM.

    Attributes:
        EDGE_LIST: Graph as a list of edges with source, target, confidence.
        ADJACENCY_MATRIX: Graph as a square matrix where entry (i,j)
            represents confidence that variable i causes variable j.
        PDG: Probabilistic Dependency Graph with existence and orientation
            probabilities for each proposed edge.
    """

    EDGE_LIST = "edge_list"
    ADJACENCY_MATRIX = "adjacency_matrix"
    PDG = "pdg"


# System prompt for PDG (Probabilistic Dependency Graph) format
GRAPH_SYSTEM_PROMPT_PDG = """\
You are an expert in causal reasoning and domain knowledge.
Your task is to propose causal relationships between variables with \
uncertainty estimates.

Respond ONLY with valid JSON in this exact format:
{
  "edges": [
    {
      "source": "variable_name_1",
      "target": "variable_name_2",
      "existence": 0.0 to 1.0,
      "orientation": 0.0 to 1.0
    }
  ],
  "reasoning": "brief explanation of your approach"
}

Guidelines:
- source/target: the variable names (source is your proposed cause)
- existence: probability that ANY causal relationship exists between these \
variables (0.0 = definitely no relationship, 1.0 = definitely related)
- orientation: given a relationship exists, confidence that source causes \
target rather than target causes source (0.5 = uncertain direction, 1.0 = \
certain source causes target, 0.0 = certain target causes source)
- Include ONLY direct causal relationships, not indirect ones
- Use the exact variable names provided
- Do not add edges between a variable and itself
- Consider domain knowledge and temporal ordering
- Omit pairs where no causal relationship exists (existence would be 0)"""


# System prompt for graph generation (edge list format)
GRAPH_SYSTEM_PROMPT_EDGE_LIST = """\
You are an expert in causal reasoning and domain knowledge.
Your task is to propose a complete causal graph structure given a set of \
variables.

Respond ONLY with valid JSON in this exact format:
{
  "edges": [
    {
      "source": "variable_name_1",
      "target": "variable_name_2",
      "confidence": 0.0 to 1.0
    }
  ],
  "reasoning": "brief explanation of your approach"
}

Guidelines:
- Each edge represents a direct causal relationship (source causes target)
- Include ONLY direct causal relationships, not indirect ones
- confidence: your confidence in each edge from 0.0 (low) to 1.0 (certain)
- Use the exact variable names provided
- Do not add edges between a variable and itself
- Consider domain knowledge and temporal ordering
- Omit edges where no causal relationship exists"""

# System prompt for graph generation (adjacency matrix format)
GRAPH_SYSTEM_PROMPT_ADJACENCY = """\
You are an expert in causal reasoning and domain knowledge.
Your task is to propose a complete causal graph structure given a set of \
variables.

Respond ONLY with valid JSON in this exact format:
{
  "variables": ["var1", "var2", "var3"],
  "adjacency_matrix": [
    [0.0, 0.8, 0.0],
    [0.0, 0.0, 0.6],
    [0.0, 0.0, 0.0]
  ],
  "reasoning": "brief explanation of your approach"
}

Guidelines:
- List variables in the order you will use in the matrix
- adjacency_matrix(i,j) = confidence that variables(i) causes variables(j)
- Values range from 0.0 (no edge) to 1.0 (certain edge)
- Diagonal elements must be 0.0 (no self-loops)
- Consider domain knowledge and temporal ordering
- Use 0.0 for pairs with no causal relationship"""

# User prompt template for minimal context (names only)
USER_PROMPT_MINIMAL = """\
Propose a causal graph for the following variables:

Variables: {variable_names}

Based on your domain knowledge, identify which variables directly cause \
others."""

# User prompt template for minimal context with domain
USER_PROMPT_MINIMAL_WITH_DOMAIN = """\
In the domain of {domain}:

Propose a causal graph for the following variables:

Variables: {variable_names}

Based on your domain knowledge, identify which variables directly cause \
others."""

# User prompt template for standard context
USER_PROMPT_STANDARD = """\
Propose a causal graph for the following variables:

{variable_details}

Based on the variable types and descriptions, identify which variables \
directly cause others."""

# User prompt template for standard context with domain
USER_PROMPT_STANDARD_WITH_DOMAIN = """\
In the domain of {domain}:

Propose a causal graph for the following variables:

{variable_details}

Based on the variable types and descriptions, identify which variables \
directly cause others."""

# User prompt template for rich context
USER_PROMPT_RICH = """\
Propose a causal graph for the following variables:

{variable_details}

Consider:
- Variable roles (exogenous variables have no parents)
- Domain-specific causal mechanisms
- Temporal ordering where applicable
- Related domain knowledge provided

Identify which variables directly cause others."""

# User prompt template for rich context with domain
USER_PROMPT_RICH_WITH_DOMAIN = """\
In the domain of {domain}:

Propose a causal graph for the following variables:

{variable_details}

Consider:
- Variable roles (exogenous variables have no parents)
- Domain-specific causal mechanisms
- Temporal ordering where applicable
- Related domain knowledge provided

Identify which variables directly cause others."""


def _format_variable_details(
    variables: list[dict[str, Any]],
    level: PromptDetail,
) -> str:
    """Format variable details for prompt inclusion.

    Args:
        variables: List of filtered variable dictionaries.
        level: The view level for formatting style.

    Returns:
        Formatted string of variable details.
    """
    lines = []

    for var in variables:
        name = var.get("name", "unknown")

        if level == PromptDetail.MINIMAL:
            lines.append(f"- {name}")
        elif level == PromptDetail.STANDARD:
            var_type = var.get("type", "")
            desc = var.get("short_description", "")
            states = var.get("states", [])

            parts = [f"- {name}"]
            if var_type:
                parts.append(f"  Type: {var_type}")
            if desc:
                parts.append(f"  Description: {desc}")
            if states:
                parts.append(f"  States: {', '.join(str(s) for s in states)}")
            lines.append("\n".join(parts))
        else:  # RICH
            var_type = var.get("type", "")
            role = var.get("role", "")
            category = var.get("category", "")
            short_desc = var.get("short_description", "")
            extended_desc = var.get("extended_description", "")
            states = var.get("states", [])
            hints = var.get("sensitivity_hints", "")
            knowledge = var.get("related_domain_knowledge", [])

            parts = [f"- {name}"]
            if var_type:
                parts.append(f"  Type: {var_type}")
            if role:
                parts.append(f"  Role: {role}")
            if category:
                parts.append(f"  Category: {category}")
            if short_desc:
                parts.append(f"  Description: {short_desc}")
            if extended_desc:
                parts.append(f"  Extended: {extended_desc}")
            if states:
                parts.append(f"  States: {', '.join(str(s) for s in states)}")
            if hints:
                parts.append(f"  Causal hints: {hints}")
            if knowledge:
                knowledge_str = "; ".join(str(k) for k in knowledge)
                parts.append(f"  Domain knowledge: {knowledge_str}")
            lines.append("\n".join(parts))

    return "\n\n".join(lines)


@dataclass
class GraphQueryPrompt:
    """Builder for graph generation query prompts.

    This class constructs system and user prompts for querying an LLM
    to generate a complete causal graph from variable specifications.

    Attributes:
        variables: List of filtered variable dictionaries.
        level: The view level (minimal, standard, rich).
        domain: Optional domain context.
        output_format: Desired output format (edge_list or adjacency_matrix).
        system_prompt: Custom system prompt (uses default if None).

    Example:
        >>> context = NetworkContext.load("model.json")
        >>> view_filter = ViewFilter(context)
        >>> variables = view_filter.filter_variables(PromptDetail.STANDARD)
        >>> prompt = GraphQueryPrompt(
        ...     variables=variables,
        ...     level=PromptDetail.STANDARD,
        ...     domain=context.domain,
        ... )
        >>> system, user = prompt.build()
    """

    variables: list[dict[str, Any]]
    level: PromptDetail = PromptDetail.STANDARD
    domain: Optional[str] = None
    output_format: OutputFormat = OutputFormat.EDGE_LIST
    system_prompt: Optional[str] = None

    def build(self) -> tuple[str, str]:
        """Build the system and user prompts for LLM graph generation.

        Constructs a system prompt (instructions for the LLM) and a user
        prompt (the actual query with variable information) based on the
        configured output format, view level, and domain context.

        Returns:
            Tuple of (system_prompt, user_prompt) strings ready for use
            with an LLM client.

        Example:
            >>> prompt = GraphQueryPrompt(
            ...     variables=[{"name": "age"}, {"name": "income"}],
            ...     level=PromptDetail.MINIMAL,
            ... )
            >>> system, user = prompt.build()
            >>> # system contains JSON format instructions
            >>> # user contains the variable query
        """
        # Select system prompt based on output format
        if self.system_prompt:
            system = self.system_prompt
        elif self.output_format == OutputFormat.ADJACENCY_MATRIX:
            system = GRAPH_SYSTEM_PROMPT_ADJACENCY
        elif self.output_format == OutputFormat.PDG:
            system = GRAPH_SYSTEM_PROMPT_PDG
        else:
            system = GRAPH_SYSTEM_PROMPT_EDGE_LIST

        # Build user prompt based on view level
        user = self._build_user_prompt()

        return system, user

    def _build_user_prompt(self) -> str:
        """Build the user prompt based on view level and domain.

        Selects the appropriate template based on view level (minimal,
        standard, or rich) and whether a domain context is provided.
        Formats variable information according to the selected template.

        Returns:
            The formatted user prompt string.
        """
        if self.level == PromptDetail.MINIMAL:
            # Extract just the names for minimal view
            names = [v.get("name", "unknown") for v in self.variables]
            variable_names = ", ".join(names)

            if self.domain:
                return USER_PROMPT_MINIMAL_WITH_DOMAIN.format(
                    domain=self.domain,
                    variable_names=variable_names,
                )
            return USER_PROMPT_MINIMAL.format(variable_names=variable_names)

        # Standard and Rich views use detailed variable info
        variable_details = _format_variable_details(self.variables, self.level)

        if self.level == PromptDetail.STANDARD:
            if self.domain:
                return USER_PROMPT_STANDARD_WITH_DOMAIN.format(
                    domain=self.domain,
                    variable_details=variable_details,
                )
            return USER_PROMPT_STANDARD.format(
                variable_details=variable_details,
            )

        # Rich level
        if self.domain:
            return USER_PROMPT_RICH_WITH_DOMAIN.format(
                domain=self.domain,
                variable_details=variable_details,
            )
        return USER_PROMPT_RICH.format(variable_details=variable_details)

    def get_variable_names(self) -> list[str]:
        """Get the list of variable names from the filtered variables.

        Extracts the name field from each variable dictionary. Useful for
        validating LLM responses to ensure they reference valid variables.

        Returns:
            List of variable names. Returns "unknown" for any variable
            missing a name field.

        Example:
            >>> prompt = GraphQueryPrompt(
            ...     variables=[{"name": "age"}, {"name": "income"}],
            ...     level=PromptDetail.MINIMAL,
            ... )
            >>> prompt.get_variable_names()
            ['age', 'income']
        """
        return [v.get("name", "unknown") for v in self.variables]

    @classmethod
    def from_context(
        cls,
        context: "NetworkContext",
        level: PromptDetail = PromptDetail.STANDARD,
        output_format: OutputFormat = OutputFormat.EDGE_LIST,
        system_prompt: Optional[str] = None,
        use_llm_names: bool = True,
    ) -> "GraphQueryPrompt":
        """Create a GraphQueryPrompt from a NetworkContext.

        Convenience factory method that automatically applies view
        filtering to extract variables at the specified level. This is
        the recommended way to create prompts when working with
        NetworkContext objects.

        Args:
            context: The network context containing variable definitions.
            level: The view level determining context depth. Defaults to
                STANDARD which includes names, types, and descriptions.
            output_format: Desired output format for LLM response. Defaults
                to EDGE_LIST.
            system_prompt: Custom system prompt to override the default.
                If None, uses the appropriate default for the output format.
            use_llm_names: If True (default), use llm_name for variables in
                the prompt. If False, use benchmark names directly.

        Returns:
            GraphQueryPrompt instance configured with filtered variables
            and domain context from the context.

        Example:
            >>> context = NetworkContext.load("asia.json")
            >>> prompt = GraphQueryPrompt.from_context(
            ...     context,
            ...     level=PromptDetail.RICH,
            ... )
            >>> system, user = prompt.build()
        """
        view_filter = ViewFilter(context, use_llm_names=use_llm_names)
        variables = view_filter.filter_variables(level)

        return cls(
            variables=variables,
            level=level,
            domain=context.domain,
            output_format=output_format,
            system_prompt=system_prompt,
        )


# Expected response schemas for documentation and validation
EDGE_LIST_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["edges"],
    "properties": {
        "edges": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["source", "target"],
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                },
            },
        },
        "reasoning": {"type": "string"},
    },
}

ADJACENCY_MATRIX_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["variables", "adjacency_matrix"],
    "properties": {
        "variables": {
            "type": "array",
            "items": {"type": "string"},
        },
        "adjacency_matrix": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number", "minimum": 0, "maximum": 1},
            },
        },
        "reasoning": {"type": "string"},
    },
}
