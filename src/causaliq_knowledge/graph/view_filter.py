"""View filtering for model specifications.

This module provides functionality to extract filtered views
(minimal, standard, rich) from model specifications.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from causaliq_knowledge.graph.models import ModelSpec, VariableSpec


class ViewLevel(str, Enum):
    """Level of detail for variable context views."""

    MINIMAL = "minimal"
    STANDARD = "standard"
    RICH = "rich"


class ViewFilter:
    """Filter model specifications to extract specific view levels.

    This class extracts variable information according to the view
    definitions in the model specification (minimal, standard, rich).

    Example:
        >>> spec = ModelLoader.load("model.json")
        >>> view_filter = ViewFilter(spec)
        >>> minimal_vars = view_filter.filter_variables(ViewLevel.MINIMAL)
        >>> # Returns list of dicts with only 'name' field
    """

    def __init__(self, spec: ModelSpec) -> None:
        """Initialise the view filter.

        Args:
            spec: The model specification to filter.
        """
        self._spec = spec

    @property
    def spec(self) -> ModelSpec:
        """Return the model specification."""
        return self._spec

    def get_include_fields(self, level: ViewLevel) -> list[str]:
        """Get the fields to include for a given view level.

        Args:
            level: The view level (minimal, standard, rich).

        Returns:
            List of field names to include.
        """
        if level == ViewLevel.MINIMAL:
            return self._spec.views.minimal.include_fields
        elif level == ViewLevel.STANDARD:
            return self._spec.views.standard.include_fields
        elif level == ViewLevel.RICH:
            return self._spec.views.rich.include_fields
        else:  # pragma: no cover
            raise ValueError(f"Unknown view level: {level}")

    def filter_variable(
        self,
        variable: VariableSpec,
        level: ViewLevel,
    ) -> dict[str, Any]:
        """Filter a single variable to include only specified fields.

        Args:
            variable: The variable specification to filter.
            level: The view level determining which fields to include.

        Returns:
            Dictionary with only the fields specified by the view level.
            Enum values are converted to their string representations.
        """
        include_fields = self.get_include_fields(level)
        # Use mode="json" to convert enums to their string values
        var_dict = variable.model_dump(mode="json")

        return {
            key: value
            for key, value in var_dict.items()
            if key in include_fields and value is not None
        }

    def filter_variables(self, level: ViewLevel) -> list[dict[str, Any]]:
        """Filter all variables to the specified view level.

        Args:
            level: The view level (minimal, standard, rich).

        Returns:
            List of filtered variable dictionaries.
        """
        return [
            self.filter_variable(var, level) for var in self._spec.variables
        ]

    def get_variable_names(self) -> list[str]:
        """Get all variable names from the specification.

        Returns:
            List of variable names.
        """
        return self._spec.get_variable_names()

    def get_domain(self) -> str:
        """Get the domain from the specification.

        Returns:
            The domain string.
        """
        return self._spec.domain

    def get_context_summary(self, level: ViewLevel) -> dict[str, Any]:
        """Get a complete context summary for LLM prompts.

        Args:
            level: The view level for variable filtering.

        Returns:
            Dictionary with domain and filtered variables.
        """
        return {
            "domain": self._spec.domain,
            "dataset_id": self._spec.dataset_id,
            "variables": self.filter_variables(level),
        }
