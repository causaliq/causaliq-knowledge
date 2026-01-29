"""Variable disguising for LLM queries.

This module provides functionality to obfuscate variable names
when sending to LLMs and translate responses back to original names.
"""

from __future__ import annotations

import random
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from causaliq_knowledge.graph.models import ModelSpec
    from causaliq_knowledge.graph.response import GeneratedGraph


class VariableDisguiser:
    """Obfuscate variable names for LLM queries.

    This class creates a reproducible mapping from original variable
    names to disguised names (V1, V2, etc.) and provides methods
    to translate between the two representations.

    Example:
        >>> spec = ModelLoader.load("model.json")
        >>> disguiser = VariableDisguiser(spec, seed=42)
        >>> disguised = disguiser.disguise_text("smoking causes lung cancer")
        >>> # Returns text with original names replaced
        >>> original = disguiser.reveal_text(disguised)
        >>> # Translates back to original names
    """

    def __init__(
        self,
        spec: ModelSpec,
        seed: int | None = None,
        prefix: str = "V",
    ) -> None:
        """Initialise the variable disguiser.

        Args:
            spec: The model specification containing variables.
            seed: Random seed for reproducible disguising. If None,
                a random mapping is generated.
            prefix: Prefix for disguised names (default: "V").
        """
        self._spec = spec
        self._seed = seed
        self._prefix = prefix
        self._original_to_disguised: dict[str, str] = {}
        self._disguised_to_original: dict[str, str] = {}
        self._build_mapping()

    def _build_mapping(self) -> None:
        """Build the bidirectional mapping between names."""
        names = self._spec.get_variable_names()
        indices = list(range(1, len(names) + 1))

        if self._seed is not None:
            rng = random.Random(self._seed)
            rng.shuffle(indices)

        for name, idx in zip(names, indices):
            disguised = f"{self._prefix}{idx}"
            self._original_to_disguised[name] = disguised
            self._disguised_to_original[disguised] = name

    @property
    def seed(self) -> int | None:
        """Return the random seed used for mapping."""
        return self._seed

    @property
    def prefix(self) -> str:
        """Return the prefix used for disguised names."""
        return self._prefix

    @property
    def original_to_disguised(self) -> dict[str, str]:
        """Return the original-to-disguised mapping."""
        return self._original_to_disguised.copy()

    @property
    def disguised_to_original(self) -> dict[str, str]:
        """Return the disguised-to-original mapping."""
        return self._disguised_to_original.copy()

    def disguise_name(self, original: str) -> str:
        """Convert an original variable name to its disguised form.

        Args:
            original: The original variable name.

        Returns:
            The disguised variable name.

        Raises:
            KeyError: If the original name is not in the mapping.
        """
        if original not in self._original_to_disguised:
            raise KeyError(f"Unknown variable: {original}")
        return self._original_to_disguised[original]

    def reveal_name(self, disguised: str) -> str:
        """Convert a disguised variable name back to its original form.

        Args:
            disguised: The disguised variable name.

        Returns:
            The original variable name.

        Raises:
            KeyError: If the disguised name is not in the mapping.
        """
        if disguised not in self._disguised_to_original:
            raise KeyError(f"Unknown disguised name: {disguised}")
        return self._disguised_to_original[disguised]

    def disguise_text(self, text: str) -> str:
        """Replace all original variable names in text with disguised names.

        Args:
            text: Text containing original variable names.

        Returns:
            Text with original names replaced by disguised names.
        """
        result = text
        # Sort by length descending to replace longer names first
        # This prevents partial replacements of substrings
        sorted_names = sorted(
            self._original_to_disguised.keys(),
            key=len,
            reverse=True,
        )
        for original in sorted_names:
            disguised = self._original_to_disguised[original]
            # Use word boundary matching for safety
            pattern = re.compile(re.escape(original), re.IGNORECASE)
            result = pattern.sub(disguised, result)
        return result

    def reveal_text(self, text: str) -> str:
        """Replace all disguised variable names in text with original names.

        Args:
            text: Text containing disguised variable names.

        Returns:
            Text with disguised names replaced by original names.
        """
        result = text
        # Sort by length descending to replace longer names first
        sorted_names = sorted(
            self._disguised_to_original.keys(),
            key=len,
            reverse=True,
        )
        for disguised in sorted_names:
            original = self._disguised_to_original[disguised]
            # Use word boundary matching
            pattern = re.compile(re.escape(disguised), re.IGNORECASE)
            result = pattern.sub(original, result)
        return result

    def disguise_names_list(self, names: list[str]) -> list[str]:
        """Convert a list of original names to disguised names.

        Args:
            names: List of original variable names.

        Returns:
            List of disguised variable names.
        """
        return [self.disguise_name(name) for name in names]

    def reveal_names_list(self, names: list[str]) -> list[str]:
        """Convert a list of disguised names to original names.

        Args:
            names: List of disguised variable names.

        Returns:
            List of original variable names.
        """
        return [self.reveal_name(name) for name in names]

    def disguise_spec(self) -> "ModelSpec":
        """Create a copy of the ModelSpec with disguised variable names.

        Returns:
            New ModelSpec with variables renamed to disguised names.
        """
        from causaliq_knowledge.graph.models import ModelSpec, VariableSpec

        disguised_vars = []
        for var in self._spec.variables:
            # Create new variable with disguised name
            disguised_name = self._original_to_disguised[var.name]
            var_dict = var.model_dump()
            var_dict["name"] = disguised_name
            # Clear descriptions to avoid leaking information
            var_dict["short_description"] = f"Variable {disguised_name}"
            var_dict["long_description"] = None
            disguised_vars.append(VariableSpec(**var_dict))

        return ModelSpec(
            dataset_id=self._spec.dataset_id,
            domain=self._spec.domain,
            variables=disguised_vars,
            views=self._spec.views,
            provenance=self._spec.provenance,
        )

    def undisguise_graph(self, graph: "GeneratedGraph") -> "GeneratedGraph":
        """Convert a graph with disguised names back to original names.

        Args:
            graph: GeneratedGraph with disguised variable names.

        Returns:
            New GeneratedGraph with original variable names.
        """
        from causaliq_knowledge.graph.response import (
            GeneratedGraph,
            ProposedEdge,
        )

        undisguised_edges = []
        for edge in graph.edges:
            source = self._disguised_to_original.get(edge.source, edge.source)
            target = self._disguised_to_original.get(edge.target, edge.target)
            undisguised_edges.append(
                ProposedEdge(
                    source=source,
                    target=target,
                    confidence=edge.confidence,
                    reasoning=self.reveal_text(edge.reasoning or ""),
                )
            )

        # Convert disguised variable names back to original names
        undisguised_vars = [
            self._disguised_to_original.get(v, v) for v in graph.variables
        ]

        return GeneratedGraph(
            edges=undisguised_edges,
            variables=undisguised_vars,
            metadata=graph.metadata,
            raw_response=graph.raw_response,
        )
