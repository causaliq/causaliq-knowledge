"""Unit tests for graph generation parameter validation.

Tests for the GenerateGraphParams model which provides shared
validation for CLI and workflow action parameters.
"""

from pathlib import Path
from typing import Any, Dict

import pytest
from pydantic import ValidationError

from causaliq_knowledge.graph.params import GenerateGraphParams
from causaliq_knowledge.graph.prompts import OutputFormat
from causaliq_knowledge.graph.view_filter import PromptDetail


# Test creating params with minimal required fields.
def test_params_minimal_required() -> None:
    """Test creating params with only required fields."""
    params = GenerateGraphParams(model_spec=Path("model.json"))

    assert params.model_spec == Path("model.json")
    assert params.prompt_detail == PromptDetail.STANDARD
    assert params.disguise is False
    assert params.use_benchmark_names is False
    assert params.seed is None
    assert params.llm == "groq/llama-3.1-8b-instant"
    assert params.output is None
    assert params.output_format == OutputFormat.EDGE_LIST
    assert params.cache is True
    assert params.cache_path is None
    assert params.temperature == 0.1
    assert params.request_id == ""


# Test creating params with all fields specified.
def test_params_all_fields() -> None:
    """Test creating params with all fields explicitly set."""
    params = GenerateGraphParams(
        model_spec=Path("test/model.json"),
        prompt_detail=PromptDetail.RICH,
        disguise=True,
        use_benchmark_names=False,
        seed=42,
        llm="gemini/gemini-2.5-flash",
        output=Path("output/graph.json"),
        output_format=OutputFormat.ADJACENCY_MATRIX,
        cache=False,
        cache_path=Path("cache/test.db"),
        temperature=0.5,
        request_id="test-123",
    )

    assert params.model_spec == Path("test/model.json")
    assert params.prompt_detail == PromptDetail.RICH
    assert params.disguise is True
    assert params.seed == 42
    assert params.llm == "gemini/gemini-2.5-flash"
    assert params.output == Path("output/graph.json")
    assert params.output_format == OutputFormat.ADJACENCY_MATRIX
    assert params.cache is False
    assert params.cache_path == Path("cache/test.db")
    assert params.temperature == 0.5
    assert params.request_id == "test-123"


# Test LLM model validation with valid providers.
@pytest.mark.parametrize(
    "llm",
    [
        "anthropic/claude-3-5-sonnet-20241022",
        "deepseek/deepseek-chat",
        "gemini/gemini-2.5-flash",
        "groq/llama-3.1-8b-instant",
        "mistral/mistral-small-latest",
        "ollama/llama3.2:1b",
        "openai/gpt-4o",
    ],
)
def test_params_valid_llm_providers(llm: str) -> None:
    """Test that all valid LLM provider prefixes are accepted."""
    params = GenerateGraphParams(model_spec=Path("model.json"), llm=llm)
    assert params.llm == llm


# Test LLM model validation rejects invalid providers.
def test_params_invalid_llm_provider() -> None:
    """Test that invalid LLM provider prefixes are rejected."""
    with pytest.raises(ValidationError) as exc_info:
        GenerateGraphParams(
            model_spec=Path("model.json"),
            llm="invalid/model-name",
        )

    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert "provider prefix" in errors[0]["msg"].lower()


# Test LLM model validation rejects missing provider.
def test_params_llm_missing_provider() -> None:
    """Test that LLM model without provider prefix is rejected."""
    with pytest.raises(ValidationError) as exc_info:
        GenerateGraphParams(
            model_spec=Path("model.json"),
            llm="llama-3.1-8b-instant",
        )

    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert "provider prefix" in errors[0]["msg"].lower()


# Test seed without disguise validation error.
def test_params_seed_requires_disguise() -> None:
    """Test that seed parameter requires disguise to be enabled."""
    with pytest.raises(ValidationError) as exc_info:
        GenerateGraphParams(
            model_spec=Path("model.json"),
            seed=42,
            disguise=False,
        )

    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert "disguise" in errors[0]["msg"].lower()


# Test seed with disguise is valid.
def test_params_seed_with_disguise_valid() -> None:
    """Test that seed with disguise enabled is accepted."""
    params = GenerateGraphParams(
        model_spec=Path("model.json"),
        seed=42,
        disguise=True,
    )
    assert params.seed == 42
    assert params.disguise is True


# Test temperature validation bounds.
def test_params_temperature_bounds() -> None:
    """Test temperature validation accepts valid range."""
    # Valid minimum
    params_min = GenerateGraphParams(
        model_spec=Path("model.json"),
        temperature=0.0,
    )
    assert params_min.temperature == 0.0

    # Valid maximum
    params_max = GenerateGraphParams(
        model_spec=Path("model.json"),
        temperature=2.0,
    )
    assert params_max.temperature == 2.0

    # Valid middle
    params_mid = GenerateGraphParams(
        model_spec=Path("model.json"),
        temperature=1.0,
    )
    assert params_mid.temperature == 1.0


# Test temperature below minimum is rejected.
def test_params_temperature_below_minimum() -> None:
    """Test temperature below 0.0 is rejected."""
    with pytest.raises(ValidationError) as exc_info:
        GenerateGraphParams(
            model_spec=Path("model.json"),
            temperature=-0.1,
        )

    errors = exc_info.value.errors()
    assert len(errors) == 1


# Test temperature above maximum is rejected.
def test_params_temperature_above_maximum() -> None:
    """Test temperature above 2.0 is rejected."""
    with pytest.raises(ValidationError) as exc_info:
        GenerateGraphParams(
            model_spec=Path("model.json"),
            temperature=2.1,
        )

    errors = exc_info.value.errors()
    assert len(errors) == 1


# Test from_dict with string values.
def test_params_from_dict_strings() -> None:
    """Test from_dict converts string values to correct types."""
    data: Dict[str, Any] = {
        "model_spec": "test/model.json",
        "prompt_detail": "rich",
        "output_format": "adjacency_matrix",
        "output": "output/graph.json",
        "cache_path": "cache/test.db",
        "llm": "groq/llama-3.1-8b-instant",
    }

    params = GenerateGraphParams.from_dict(data)

    assert params.model_spec == Path("test/model.json")
    assert params.prompt_detail == PromptDetail.RICH
    assert params.output_format == OutputFormat.ADJACENCY_MATRIX
    assert params.output == Path("output/graph.json")
    assert params.cache_path == Path("cache/test.db")


# Test from_dict with already-typed values.
def test_params_from_dict_typed() -> None:
    """Test from_dict accepts already-typed values."""
    data: Dict[str, Any] = {
        "model_spec": Path("test/model.json"),
        "prompt_detail": PromptDetail.MINIMAL,
        "output_format": OutputFormat.EDGE_LIST,
    }

    params = GenerateGraphParams.from_dict(data)

    assert params.model_spec == Path("test/model.json")
    assert params.prompt_detail == PromptDetail.MINIMAL
    assert params.output_format == OutputFormat.EDGE_LIST


# Test from_dict with case-insensitive enum values.
def test_params_from_dict_case_insensitive() -> None:
    """Test from_dict handles case-insensitive enum values."""
    data: Dict[str, Any] = {
        "model_spec": "model.json",
        "prompt_detail": "STANDARD",
        "output_format": "Edge_List",
    }

    params = GenerateGraphParams.from_dict(data)

    assert params.prompt_detail == PromptDetail.STANDARD
    assert params.output_format == OutputFormat.EDGE_LIST


# Test from_dict validation error propagation.
def test_params_from_dict_validation_error() -> None:
    """Test from_dict propagates validation errors."""
    data: Dict[str, Any] = {
        "model_spec": "model.json",
        "llm": "invalid/provider",
    }

    with pytest.raises(ValidationError):
        GenerateGraphParams.from_dict(data)


# Test get_effective_cache_path with cache disabled.
def test_params_cache_path_disabled() -> None:
    """Test get_effective_cache_path returns None when cache disabled."""
    params = GenerateGraphParams(
        model_spec=Path("test/model.json"),
        cache=False,
    )

    assert params.get_effective_cache_path() is None


# Test get_effective_cache_path with explicit path.
def test_params_cache_path_explicit() -> None:
    """Test get_effective_cache_path returns explicit path."""
    params = GenerateGraphParams(
        model_spec=Path("test/model.json"),
        cache=True,
        cache_path=Path("custom/cache.db"),
    )

    assert params.get_effective_cache_path() == Path("custom/cache.db")


# Test get_effective_cache_path derives default from model_spec.
def test_params_cache_path_default() -> None:
    """Test get_effective_cache_path derives path from model_spec."""
    params = GenerateGraphParams(
        model_spec=Path("test/cancer.json"),
        cache=True,
    )

    expected = Path("test/cancer_llm.db")
    assert params.get_effective_cache_path() == expected


# Test get_effective_cache_path with nested model_spec path.
def test_params_cache_path_nested() -> None:
    """Test cache path derivation with deeply nested model_spec."""
    params = GenerateGraphParams(
        model_spec=Path("research/models/asia/asia.json"),
        cache=True,
    )

    expected = Path("research/models/asia/asia_llm.db")
    assert params.get_effective_cache_path() == expected


# Test prompt_detail level enum values.
@pytest.mark.parametrize(
    "prompt_detail",
    [PromptDetail.MINIMAL, PromptDetail.STANDARD, PromptDetail.RICH],
)
def test_params_prompt_detail_levels(prompt_detail: PromptDetail) -> None:
    """Test all prompt_detail levels are accepted."""
    params = GenerateGraphParams(
        model_spec=Path("model.json"), prompt_detail=prompt_detail
    )
    assert params.prompt_detail == prompt_detail


# Test output format enum values.
@pytest.mark.parametrize(
    "output_format",
    [OutputFormat.EDGE_LIST, OutputFormat.ADJACENCY_MATRIX],
)
def test_params_output_formats(output_format: OutputFormat) -> None:
    """Test all output formats are accepted."""
    params = GenerateGraphParams(
        model_spec=Path("model.json"),
        output_format=output_format,
    )
    assert params.output_format == output_format
