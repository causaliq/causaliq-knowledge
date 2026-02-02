"""Shared parameter models for graph generation.

This module provides Pydantic models for validating graph generation
parameters, shared between CLI commands and workflow actions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

from causaliq_knowledge.graph.prompts import OutputFormat
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
        llm: LLM model identifier with provider prefix.
        output: Output file path for results.
        output_format: Format for generated graph output.
        cache: Enable LLM response caching.
        cache_path: Path to cache database file.
        temperature: LLM sampling temperature.
        request_id: Identifier for requests (stored in metadata).

    Example:
        >>> params = GenerateGraphParams(
        ...     model_spec=Path("model.json"),
        ...     prompt_detail=PromptDetail.STANDARD,
        ...     llm="groq/llama-3.1-8b-instant",
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
    llm: str = Field(
        default="groq/llama-3.1-8b-instant",
        description="LLM model identifier with provider prefix",
    )
    output: Optional[Path] = Field(
        default=None,
        description="Output file path for results",
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.EDGE_LIST,
        description="Format for generated graph output",
    )
    cache: bool = Field(
        default=True,
        description="Enable LLM response caching",
    )
    cache_path: Optional[Path] = Field(
        default=None,
        description="Path to cache database file",
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="LLM sampling temperature (0.0-2.0)",
    )
    request_id: str = Field(
        default="",
        description="Identifier for requests (stored in metadata)",
    )

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("llm")
    @classmethod
    def validate_llm_format(cls, v: str) -> str:
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

        # Convert output_format string to OutputFormat enum
        if "output_format" in processed and isinstance(
            processed["output_format"], str
        ):
            processed["output_format"] = OutputFormat(
                processed["output_format"].lower()
            )

        # Convert model_spec string to Path
        if "model_spec" in processed and isinstance(
            processed["model_spec"], str
        ):
            processed["model_spec"] = Path(processed["model_spec"])

        # Convert output string to Path
        if "output" in processed and isinstance(processed["output"], str):
            processed["output"] = Path(processed["output"])

        # Convert cache_path string to Path
        if "cache_path" in processed and isinstance(
            processed["cache_path"], str
        ):
            processed["cache_path"] = Path(processed["cache_path"])

        return cls(**processed)

    def get_effective_cache_path(self) -> Optional[Path]:
        """Get the effective cache path, deriving default if not specified.

        If cache is enabled but no cache_path is specified, returns a
        default path based on the model_spec location.

        Returns:
            Path to cache database, or None if caching is disabled.
        """
        if not self.cache:
            return None

        if self.cache_path is not None:
            return self.cache_path

        # Default: alongside model spec, e.g. cancer.json -> cancer_llm.db
        stem = self.model_spec.stem
        return self.model_spec.parent / f"{stem}_llm.db"
