"""Model specification loader with validation.

This module provides functionality to load and validate model
specification JSON files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from causaliq_knowledge.graph.models import ModelSpec


class ModelLoadError(Exception):
    """Error raised when model loading fails.

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


class ModelLoader:
    """Loader for model specification JSON files.

    This class provides methods to load, validate, and access
    model specifications from JSON files.

    Example:
        >>> loader = ModelLoader()
        >>> spec = loader.load("path/to/model.json")
        >>> print(spec.dataset_id)
        'cancer'

        >>> # Or load from dict
        >>> spec = loader.from_dict(
        ...     {"dataset_id": "test", "domain": "test", ...}
        ... )
    """

    @staticmethod
    def load(path: Union[str, Path]) -> ModelSpec:
        """Load a model specification from a JSON file.

        Args:
            path: Path to the JSON file.

        Returns:
            Validated ModelSpec instance.

        Raises:
            ModelLoadError: If the file cannot be loaded or validated.
        """
        path = Path(path)

        # Check file exists
        if not path.exists():
            raise ModelLoadError("Model specification file not found", path)

        # Check file extension
        if path.suffix.lower() != ".json":
            raise ModelLoadError(
                "Model specification must be a JSON file",
                path,
                f"Got extension: {path.suffix}",
            )

        # Load JSON
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ModelLoadError(
                "Invalid JSON in model specification",
                path,
                str(e),
            ) from e
        except OSError as e:
            raise ModelLoadError(
                "Failed to read model specification file",
                path,
                str(e),
            ) from e

        # Validate and create ModelSpec
        return ModelLoader.from_dict(data, source_path=path)

    @staticmethod
    def from_dict(
        data: dict,
        source_path: Path | str | None = None,
    ) -> ModelSpec:
        """Create a ModelSpec from a dictionary.

        Args:
            data: Dictionary containing model specification.
            source_path: Optional source path for error messages.

        Returns:
            Validated ModelSpec instance.

        Raises:
            ModelLoadError: If validation fails.
        """
        # Check required fields
        required_fields = ["dataset_id", "domain"]
        missing = [f for f in required_fields if f not in data]
        if missing:
            raise ModelLoadError(
                f"Missing required fields: {', '.join(missing)}",
                source_path,
            )

        # Validate with Pydantic
        try:
            return ModelSpec.model_validate(data)
        except Exception as e:
            raise ModelLoadError(
                "Model specification validation failed",
                source_path,
                str(e),
            ) from e

    @staticmethod
    def validate_variables(spec: ModelSpec) -> list[str]:
        """Validate variable specifications and return warnings.

        Performs additional validation beyond Pydantic schema:
        - Checks for duplicate variable names
        - Checks that states are defined for discrete variables
        - Checks for empty variable list

        Args:
            spec: ModelSpec to validate.

        Returns:
            List of warning messages (empty if no issues).

        Raises:
            ModelLoadError: If critical validation errors found.
        """
        warnings: list[str] = []

        # Check for empty variables
        if not spec.variables:
            raise ModelLoadError(
                "Model specification has no variables defined"
            )

        # Check for duplicate names
        names = [v.name for v in spec.variables]
        duplicates = [n for n in names if names.count(n) > 1]
        if duplicates:
            raise ModelLoadError(
                f"Duplicate variable names found: {', '.join(set(duplicates))}"
            )

        # Check states for discrete variables
        for var in spec.variables:
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

        # Check binary variables have exactly 2 states
        for var in spec.variables:
            if var.type == "binary" and var.states and len(var.states) != 2:
                warnings.append(
                    f"Variable '{var.name}' is binary "
                    f"but has {len(var.states)} states"
                )

        return warnings

    @staticmethod
    def load_and_validate(
        path: Union[str, Path],
    ) -> tuple[ModelSpec, list[str]]:
        """Load and fully validate a model specification.

        Combines loading with additional validation checks.

        Args:
            path: Path to the JSON file.

        Returns:
            Tuple of (ModelSpec, list of warnings).

        Raises:
            ModelLoadError: If loading or validation fails.
        """
        spec = ModelLoader.load(path)
        warnings = ModelLoader.validate_variables(spec)
        return spec, warnings
