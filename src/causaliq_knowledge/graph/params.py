"""Shared parameter models for graph generation.

This module provides Pydantic models for validating graph generation
parameters, shared between CLI commands and workflow actions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

from causaliq_knowledge.graph.view_filter import PromptDetail


class GenerateGraphParams(BaseModel):
    """Parameters for graph generation - shared by CLI and Action.

    This model provides validation for all graph generation parameters,
    ensuring consistent behaviour between CLI invocation and workflow
    action execution.

    Attributes:
        model_spec: Path to model specification JSON file.
        prompt_detail: Detail level for variable information in prompts.
        use_benchmark_names: Use benchmark names instead of LLM names.
        llm_model: LLM model identifier with provider prefix.
        output: Output destination - .json file path or "none" for stdout.
        llm_cache: Path to cache database file (.db) or "none" to disable.
        llm_temperature: LLM sampling temperature.

    Example:
        >>> params = GenerateGraphParams(
        ...     model_spec=Path("model.json"),
        ...     prompt_detail=PromptDetail.STANDARD,
        ...     llm_model="groq/llama-3.1-8b-instant",
        ...     output="none",
        ...     llm_cache="cache.db",
        ... )
    """

    model_spec: Path = Field(
        ...,
        description="Path to model specification JSON file",
    )
    prompt_detail: PromptDetail = Field(
        default=PromptDetail.STANDARD,
        description="Detail level for variable information in prompts",
    )
    use_benchmark_names: bool = Field(
        default=False,
        description="Use benchmark names instead of LLM names",
    )
    llm_model: str = Field(
        default="groq/llama-3.1-8b-instant",
        description="LLM model identifier with provider prefix",
    )
    output: str = Field(
        ...,
        description="Output destination: .json file path or 'none' for stdout",
    )
    llm_cache: str = Field(
        ...,
        description="Path to cache database file (.db) or 'none' to disable",
    )
    llm_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="LLM sampling temperature (0.0-2.0)",
    )

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("llm_model")
    @classmethod
    def validate_llm_model_format(cls, v: str) -> str:
        """Validate LLM model identifier has provider prefix."""
        valid_prefixes = (
            "anthropic/",
            "deepseek/",
            "gemini/",
            "groq/",
            "mistral/",
            "ollama/",
            "openai/",
        )
        if not v.startswith(valid_prefixes):
            raise ValueError(
                f"LLM model must start with provider prefix. "
                f"Valid prefixes: {', '.join(valid_prefixes)}. Got: {v}"
            )
        return v

    @field_validator("llm_cache")
    @classmethod
    def validate_llm_cache_format(cls, v: str) -> str:
        """Validate llm_cache is 'none' or a path ending with .db."""
        if v.lower() == "none":
            return "none"
        if not v.endswith(".db"):
            raise ValueError(
                f"llm_cache must be 'none' or a path ending with .db. "
                f"Got: {v}"
            )
        return v

    @field_validator("output")
    @classmethod
    def validate_output_format(cls, v: str) -> str:
        """Validate output is 'none' or a path ending with .json."""
        if v.lower() == "none":
            return "none"
        if not v.endswith(".json"):
            raise ValueError(
                f"output must be 'none' or a path ending with .json. "
                f"Got: {v}"
            )
        return v

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GenerateGraphParams":
        """Create params from dictionary with string-to-enum conversion.

        This method handles conversion of string values to enum types,
        useful when receiving parameters from workflow inputs.

        Args:
            data: Dictionary of parameter values.

        Returns:
            Validated GenerateGraphParams instance.

        Raises:
            ValueError: If validation fails.
        """
        # Convert string values to enums where needed
        processed = dict(data)

        # Convert prompt_detail string to PromptDetail enum
        if "prompt_detail" in processed and isinstance(
            processed["prompt_detail"], str
        ):
            processed["prompt_detail"] = PromptDetail(
                processed["prompt_detail"].lower()
            )

        # Convert model_spec string to Path
        if "model_spec" in processed and isinstance(
            processed["model_spec"], str
        ):
            processed["model_spec"] = Path(processed["model_spec"])

        return cls(**processed)

    def get_effective_cache_path(self) -> Optional[Path]:
        """Get the effective cache path.

        Returns:
            Path to cache database, or None if caching is disabled.
        """
        if self.llm_cache.lower() == "none":
            return None
        return Path(self.llm_cache)

    def get_effective_output_path(self) -> Optional[Path]:
        """Get the effective output path.

        Returns:
            Path to output JSON file, or None for stdout.
        """
        if self.output.lower() == "none":
            return None
        return Path(self.output)
