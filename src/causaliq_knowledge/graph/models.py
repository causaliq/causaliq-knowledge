"""Pydantic models for network context schemas.

This module defines the data models for loading and validating
network context JSON files used for LLM graph generation.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

if TYPE_CHECKING:  # pragma: no cover
    pass


class VariableType(str, Enum):
    """Type of variable in the model."""

    BINARY = "binary"
    CATEGORICAL = "categorical"
    ORDINAL = "ordinal"
    CONTINUOUS = "continuous"


class VariableRole(str, Enum):
    """Role of variable in the causal structure."""

    EXOGENOUS = "exogenous"  # No parents (root cause)
    ENDOGENOUS = "endogenous"  # Has parents (caused by other variables)
    LATENT = "latent"  # Unobserved variable


class VariableSpec(BaseModel):
    """Specification for a single variable in the causal model.

    This model captures all metadata about a variable that can be used
    to provide context to LLMs for graph generation.

    Attributes:
        name: Benchmark/literature name used for ground truth and reporting.
        llm_name: Name used when querying LLMs (prevents memorisation).
            Defaults to name if not specified.
        display_name: Human-readable name for display.
        aliases: Alternative names for the variable.
        type: Variable type (binary, categorical, ordinal, continuous).
        states: Possible values/states for discrete variables.
        role: Causal role (exogenous, endogenous, latent).
        category: Domain-specific category (e.g., "environmental_exposure").
        short_description: Brief description of the variable.
        extended_description: Detailed description with domain context.
        base_rate: Prior probabilities for each state.
        conditional_rates: Conditional probabilities given parent states.
        sensitivity_hints: Hints about causal relationships.
        related_domain_knowledge: Domain knowledge statements.
        references: Literature references.

    Example:
        >>> var = VariableSpec(
        ...     name="smoke",
        ...     llm_name="tobacco_history",
        ...     type="binary",
        ...     states=["never", "ever"],
        ...     role="exogenous",
        ...     short_description="Patient has history of tobacco smoking."
        ... )
    """

    name: str = Field(
        ..., description="Benchmark/literature name for ground truth"
    )
    llm_name: str = Field(
        default="",
        description="Name used for LLM queries (defaults to name)",
    )
    display_name: Optional[str] = Field(
        default=None, description="Human-readable display name"
    )
    aliases: list[str] = Field(
        default_factory=list, description="Alternative names"
    )
    type: VariableType = Field(..., description="Variable type")
    states: list[str] = Field(
        default_factory=list,
        description="Possible states for discrete variables",
    )
    role: Optional[VariableRole] = Field(
        default=None, description="Causal role in the structure"
    )

    @model_validator(mode="after")
    def set_llm_name_default(self) -> "VariableSpec":
        """Set llm_name to name if not specified or empty."""
        if not self.llm_name:
            # Use object.__setattr__ since Pydantic models may be frozen
            object.__setattr__(self, "llm_name", self.name)
        return self

    category: Optional[str] = Field(
        default=None, description="Domain-specific category"
    )
    short_description: Optional[str] = Field(
        default=None, description="Brief description"
    )
    extended_description: Optional[str] = Field(
        default=None, description="Detailed description with domain context"
    )
    base_rate: Optional[dict[str, float]] = Field(
        default=None, description="Prior probabilities for each state"
    )
    conditional_rates: Optional[dict[str, Any]] = Field(
        default=None, description="Conditional probabilities"
    )
    sensitivity_hints: Optional[str] = Field(
        default=None, description="Hints about causal relationships"
    )
    related_domain_knowledge: list[str] = Field(
        default_factory=list, description="Domain knowledge statements"
    )
    references: list[str] = Field(
        default_factory=list, description="Literature references"
    )

    @field_validator("type", mode="before")
    @classmethod
    def validate_type(cls, v: str | VariableType) -> VariableType:
        """Convert string type to VariableType enum."""
        if isinstance(v, VariableType):
            return v
        return VariableType(v.lower())

    @field_validator("role", mode="before")
    @classmethod
    def validate_role(
        cls, v: str | VariableRole | None
    ) -> VariableRole | None:
        """Convert string role to VariableRole enum."""
        if v is None:
            return None
        if isinstance(v, VariableRole):
            return v
        return VariableRole(v.lower())


class Provenance(BaseModel):
    """Provenance information for the model specification.

    Attributes:
        source_network: Name of the source benchmark network.
        source_reference: Citation for the original source.
        source_url: URL to the source data.
        disguise_strategy: Strategy used for variable name disguising.
        memorization_risk: Risk level for LLM memorization.
        notes: Additional notes about the source.
    """

    source_network: Optional[str] = Field(
        default=None, description="Source benchmark network name"
    )
    source_reference: Optional[str] = Field(
        default=None, description="Citation for original source"
    )
    source_url: Optional[str] = Field(
        default=None, description="URL to source data"
    )
    disguise_strategy: Optional[str] = Field(
        default=None, description="Variable name disguising strategy"
    )
    memorization_risk: Optional[str] = Field(
        default=None, description="LLM memorization risk level"
    )
    notes: Optional[str] = Field(default=None, description="Additional notes")


class LLMGuidance(BaseModel):
    """Guidance for LLMs when processing the model.

    Attributes:
        usage_notes: Notes about how to use the model.
        do_not_provide: Information that should not be given to LLMs.
    """

    usage_notes: list[str] = Field(
        default_factory=list, description="Usage guidance for LLMs"
    )
    do_not_provide: list[str] = Field(
        default_factory=list, description="Information to withhold from LLMs"
    )


class ViewDefinition(BaseModel):
    """Definition of a view (minimal, standard, rich).

    Attributes:
        description: Description of what this view includes.
        include_fields: List of VariableSpec fields to include in this view.
    """

    description: Optional[str] = Field(
        default=None, description="Description of this view"
    )
    include_fields: list[str] = Field(
        default_factory=list, description="Fields to include in this view"
    )


class PromptDetails(BaseModel):
    """Collection of prompt detail definitions.

    Attributes:
        minimal: Minimal view (typically just variable names).
        standard: Standard view (names, types, descriptions, states).
        rich: Rich view (all available metadata).
    """

    minimal: ViewDefinition = Field(
        default_factory=lambda: ViewDefinition(include_fields=["name"]),
        description="Minimal context view",
    )
    standard: ViewDefinition = Field(
        default_factory=lambda: ViewDefinition(
            include_fields=["name", "type", "short_description", "states"]
        ),
        description="Standard context view",
    )
    rich: ViewDefinition = Field(
        default_factory=lambda: ViewDefinition(
            include_fields=[
                "name",
                "display_name",
                "type",
                "role",
                "category",
                "short_description",
                "extended_description",
                "states",
                "base_rate",
                "conditional_rates",
                "sensitivity_hints",
                "related_domain_knowledge",
                "references",
            ]
        ),
        description="Rich context view",
    )


class Constraints(BaseModel):
    """Structural constraints for the causal model.

    Attributes:
        forbidden_edges: Pairs of variables that cannot have direct edges.
        partial_order: Pairs indicating causal ordering (a must precede b).
        tiers: Grouping of variables into causal tiers.
        notes: Additional notes about constraints.
    """

    forbidden_edges: list[list[str]] = Field(
        default_factory=list,
        description="Variable pairs that cannot have edges",
    )
    partial_order: list[list[str]] = Field(
        default_factory=list, description="Causal ordering constraints"
    )
    tiers: dict[str, list[str]] = Field(
        default_factory=dict, description="Variable tier groupings"
    )
    notes: Optional[str] = Field(
        default=None, description="Notes about constraints"
    )


class CausalPrinciple(BaseModel):
    """A causal principle that applies to the domain.

    Attributes:
        id: Unique identifier for the principle.
        statement: The causal principle statement.
        references: Supporting literature references.
    """

    id: str = Field(..., description="Principle identifier")
    statement: str = Field(..., description="The causal principle")
    references: list[str] = Field(
        default_factory=list, description="Literature references"
    )


class GroundTruth(BaseModel):
    """Ground truth structure for evaluation.

    Note: This should NOT be provided to LLMs during generation.

    Attributes:
        edges: Ground truth edges using benchmark variable names.
        v_structures: V-structure definitions.
        adjacency_matrix: Adjacency matrix representation.
    """

    edges: list[list[str]] = Field(
        default_factory=list, description="Edges with benchmark variable names"
    )
    v_structures: list[dict[str, Any]] = Field(
        default_factory=list, description="V-structure definitions"
    )
    adjacency_matrix: Optional[dict[str, Any]] = Field(
        default=None, description="Adjacency matrix representation"
    )


class NetworkLoadError(Exception):
    """Error raised when network context loading fails.

    Attributes:
        message: Error description.
        path: Path to the file that failed to load.
        details: Additional error details.
    """

    def __init__(
        self,
        message: str,
        path: Path | str | None = None,
        details: str | None = None,
    ) -> None:
        self.message = message
        self.path = path
        self.details = details
        full_message = message
        if path:
            full_message = f"{message}: {path}"
        if details:
            full_message = f"{full_message}\n  Details: {details}"
        super().__init__(full_message)


class NetworkContext(BaseModel):
    """Network context for LLM-based causal graph generation.

    Provides domain and variable information needed to generate causal
    graphs using LLMs. This is not the network itself, but the context
    required to generate one.

    Attributes:
        schema_version: Version of the context schema.
        network: Identifier for the benchmark network (e.g., "asia").
        domain: Domain of the causal model (e.g., "pulmonary_oncology").
        purpose: Purpose of this context specification.
        provenance: Source and provenance information.
        llm_guidance: Guidance for LLM usage.
        prompt_details: Prompt detail definitions.
        variables: List of variable specifications.
        constraints: Structural constraints.
        causal_principles: Domain causal principles.
        ground_truth: Ground truth for evaluation (not for LLMs).

    Example:
        >>> context = NetworkContext(
        ...     network="cancer",
        ...     domain="pulmonary_oncology",
        ...     variables=[
        ...         VariableSpec(
        ...             name="smoking", llm_name="tobacco_use", type="binary"
        ...         ),
        ...         VariableSpec(
        ...             name="cancer", llm_name="malignancy", type="binary"
        ...         ),
        ...     ]
        ... )

        >>> # Load from file
        >>> context = NetworkContext.load("asia.json")
    """

    schema_version: str = Field(default="2.0", description="Schema version")
    network: str = Field(..., description="Network identifier")
    domain: str = Field(..., description="Domain of the causal model")
    purpose: Optional[str] = Field(
        default=None, description="Purpose of this specification"
    )
    provenance: Optional[Provenance] = Field(
        default=None, description="Source and provenance information"
    )
    llm_guidance: Optional[LLMGuidance] = Field(
        default=None, description="Guidance for LLM usage"
    )
    prompt_details: PromptDetails = Field(
        default_factory=PromptDetails,
        description="Prompt detail definitions",
        alias="prompt_details",
    )
    variables: list[VariableSpec] = Field(
        default_factory=list, description="Variable specifications"
    )
    constraints: Optional[Constraints] = Field(
        default=None, description="Structural constraints"
    )
    causal_principles: list[CausalPrinciple] = Field(
        default_factory=list, description="Domain causal principles"
    )
    ground_truth: Optional[GroundTruth] = Field(
        default=None, description="Ground truth for evaluation"
    )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "NetworkContext":
        """Load a network context from a JSON file.

        Args:
            path: Path to the JSON file.

        Returns:
            Validated NetworkContext instance.

        Raises:
            NetworkLoadError: If the file cannot be loaded or validated.

        Example:
            >>> context = NetworkContext.load("asia.json")
            >>> print(context.network)
            'asia'
        """
        path = Path(path)

        if not path.exists():
            raise NetworkLoadError("Network context file not found", path)

        if path.suffix.lower() != ".json":
            raise NetworkLoadError(
                "Network context must be a JSON file",
                path,
                f"Got extension: {path.suffix}",
            )

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise NetworkLoadError(
                "Invalid JSON in network context",
                path,
                str(e),
            ) from e
        except OSError as e:
            raise NetworkLoadError(
                "Failed to read network context file",
                path,
                str(e),
            ) from e

        return cls.from_dict(data, source_path=path)

    @classmethod
    def from_dict(
        cls,
        data: dict,
        source_path: Path | str | None = None,
    ) -> "NetworkContext":
        """Create a NetworkContext from a dictionary.

        Args:
            data: Dictionary containing network context.
            source_path: Optional source path for error messages.

        Returns:
            Validated NetworkContext instance.

        Raises:
            NetworkLoadError: If validation fails.

        Example:
            >>> context = NetworkContext.from_dict({
            ...     "network": "test",
            ...     "domain": "testing",
            ...     "variables": [{"name": "X", "type": "binary"}]
            ... })
        """
        required_fields = ["network", "domain"]
        missing = [f for f in required_fields if f not in data]
        if missing:
            raise NetworkLoadError(
                f"Missing required fields: {', '.join(missing)}",
                source_path,
            )

        try:
            return cls.model_validate(data)
        except Exception as e:
            raise NetworkLoadError(
                "Network context validation failed",
                source_path,
                str(e),
            ) from e

    def validate_variables(self) -> list[str]:
        """Validate variable specifications and return warnings.

        Performs additional validation beyond Pydantic schema:
        - Checks for duplicate variable names
        - Checks that states are defined for discrete variables
        - Checks for empty variable list

        Returns:
            List of warning messages (empty if no issues).

        Raises:
            NetworkLoadError: If critical validation errors found.
        """
        warnings: list[str] = []

        if not self.variables:
            raise NetworkLoadError("Network context has no variables defined")

        names = [v.name for v in self.variables]
        duplicates = [n for n in names if names.count(n) > 1]
        if duplicates:
            raise NetworkLoadError(
                f"Duplicate variable names: {', '.join(set(duplicates))}"
            )

        for var in self.variables:
            if (
                var.type
                in (
                    "binary",
                    "categorical",
                    "ordinal",
                )
                and not var.states
            ):
                warnings.append(
                    f"Variable '{var.name}' is {var.type} "
                    "but has no states defined"
                )

        for var in self.variables:
            if var.type == "binary" and var.states and len(var.states) != 2:
                warnings.append(
                    f"Variable '{var.name}' is binary "
                    f"but has {len(var.states)} states"
                )

        return warnings

    @classmethod
    def load_and_validate(
        cls, path: Union[str, Path]
    ) -> tuple["NetworkContext", list[str]]:
        """Load and fully validate a network context.

        Combines loading with additional validation checks.

        Args:
            path: Path to the JSON file.

        Returns:
            Tuple of (NetworkContext, list of warnings).

        Raises:
            NetworkLoadError: If loading or validation fails.
        """
        context = cls.load(path)
        warnings = context.validate_variables()
        return context, warnings

    def get_variable(self, name: str) -> VariableSpec | None:
        """Get a variable by name.

        Args:
            name: Variable name to look up.

        Returns:
            VariableSpec if found, None otherwise.
        """
        for var in self.variables:
            if var.name == name:
                return var
        return None

    def get_variable_names(self) -> list[str]:
        """Get list of all benchmark variable names.

        Returns:
            List of variable names.
        """
        return [var.name for var in self.variables]

    def get_llm_names(self) -> list[str]:
        """Get list of all LLM variable names.

        Returns:
            List of llm_name values.
        """
        return [var.llm_name for var in self.variables]

    def get_llm_to_name_mapping(self) -> dict[str, str]:
        """Get mapping from LLM names to benchmark names.

        Returns:
            Dict mapping llm_name -> name.
        """
        return {var.llm_name: var.name for var in self.variables}

    def get_name_to_llm_mapping(self) -> dict[str, str]:
        """Get mapping from benchmark names to LLM names.

        Returns:
            Dict mapping name -> llm_name.
        """
        return {var.name: var.llm_name for var in self.variables}

    def uses_distinct_llm_names(self) -> bool:
        """Check if any variable has a different llm_name from name.

        Returns:
            True if at least one variable has llm_name != name.
        """
        return any(var.llm_name != var.name for var in self.variables)
