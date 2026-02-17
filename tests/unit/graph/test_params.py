"""Unit tests for graph generation parameter validation.

Tests for the GenerateGraphParams model which provides shared
validation for CLI and workflow action parameters.
"""

from pathlib import Path
from typing import Any, Dict

import pytest
from pydantic import ValidationError

from causaliq_knowledge.graph.params import GenerateGraphParams
from causaliq_knowledge.graph.view_filter import PromptDetail


# Test creating params with minimal required fields.
def test_params_minimal_required() -> None:
    """Test creating params with only required fields."""
    params = GenerateGraphParams(
        network_context=Path("model.json"),
        output="none",
        llm_cache="cache.db",
    )

    assert params.network_context == Path("model.json")
    assert params.prompt_detail == PromptDetail.STANDARD
    assert params.use_benchmark_names is False
    assert params.llm_model == "groq/llama-3.1-8b-instant"
    assert params.output == "none"
    assert params.llm_cache == "cache.db"
    assert params.llm_temperature == 0.1


# Test creating params with all fields specified.
def test_params_all_fields() -> None:
    """Test creating params with all fields explicitly set."""
    params = GenerateGraphParams(
        network_context=Path("test/model.json"),
        prompt_detail=PromptDetail.RICH,
        use_benchmark_names=False,
        llm_model="gemini/gemini-2.5-flash",
        output="output/workflow_cache.db",
        llm_cache="cache/test.db",
        llm_temperature=0.5,
    )

    assert params.network_context == Path("test/model.json")
    assert params.prompt_detail == PromptDetail.RICH
    assert params.llm_model == "gemini/gemini-2.5-flash"
    assert params.output == "output/workflow_cache.db"
    assert params.llm_cache == "cache/test.db"
    assert params.llm_temperature == 0.5


# Test LLM model validation with valid providers.
@pytest.mark.parametrize(
    "llm_model",
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
def test_params_valid_llm_providers(llm_model: str) -> None:
    """Test that all valid LLM provider prefixes are accepted."""
    params = GenerateGraphParams(
        network_context=Path("model.json"),
        llm_model=llm_model,
        output="none",
        llm_cache="cache.db",
    )
    assert params.llm_model == llm_model


# Test LLM model validation rejects invalid providers.
def test_params_invalid_llm_provider() -> None:
    """Test that invalid LLM provider prefixes are rejected."""
    with pytest.raises(ValidationError) as exc_info:
        GenerateGraphParams(
            network_context=Path("model.json"),
            llm_model="invalid/model-name",
            output="none",
            llm_cache="cache.db",
        )

    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert "provider prefix" in errors[0]["msg"].lower()


# Test LLM model validation rejects missing provider.
def test_params_llm_missing_provider() -> None:
    """Test that LLM model without provider prefix is rejected."""
    with pytest.raises(ValidationError) as exc_info:
        GenerateGraphParams(
            network_context=Path("model.json"),
            llm_model="llama-3.1-8b-instant",
            output="none",
            llm_cache="cache.db",
        )

    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert "provider prefix" in errors[0]["msg"].lower()


# Test llm_temperature validation bounds.
def test_params_llm_temperature_bounds() -> None:
    """Test llm_temperature validation accepts valid range."""
    # Valid minimum
    params_min = GenerateGraphParams(
        network_context=Path("model.json"),
        output="none",
        llm_cache="cache.db",
        llm_temperature=0.0,
    )
    assert params_min.llm_temperature == 0.0

    # Valid maximum
    params_max = GenerateGraphParams(
        network_context=Path("model.json"),
        output="none",
        llm_cache="cache.db",
        llm_temperature=2.0,
    )
    assert params_max.llm_temperature == 2.0

    # Valid middle
    params_mid = GenerateGraphParams(
        network_context=Path("model.json"),
        output="none",
        llm_cache="cache.db",
        llm_temperature=1.0,
    )
    assert params_mid.llm_temperature == 1.0


# Test llm_temperature below minimum is rejected.
def test_params_llm_temperature_below_minimum() -> None:
    """Test llm_temperature below 0.0 is rejected."""
    with pytest.raises(ValidationError) as exc_info:
        GenerateGraphParams(
            network_context=Path("model.json"),
            output="none",
            llm_cache="cache.db",
            llm_temperature=-0.1,
        )

    errors = exc_info.value.errors()
    assert len(errors) == 1


# Test llm_temperature above maximum is rejected.
def test_params_llm_temperature_above_maximum() -> None:
    """Test llm_temperature above 2.0 is rejected."""
    with pytest.raises(ValidationError) as exc_info:
        GenerateGraphParams(
            network_context=Path("model.json"),
            output="none",
            llm_cache="cache.db",
            llm_temperature=2.1,
        )

    errors = exc_info.value.errors()
    assert len(errors) == 1


# Test from_dict with string values.
def test_params_from_dict_strings() -> None:
    """Test from_dict converts string values to correct types."""
    data: Dict[str, Any] = {
        "network_context": "test/model.json",
        "prompt_detail": "rich",
        "output": "output/workflow_cache.db",
        "llm_cache": "cache/test.db",
        "llm_model": "groq/llama-3.1-8b-instant",
    }

    params = GenerateGraphParams.from_dict(data)

    assert params.network_context == Path("test/model.json")
    assert params.prompt_detail == PromptDetail.RICH
    assert params.output == "output/workflow_cache.db"
    assert params.llm_cache == "cache/test.db"


# Test from_dict with already-typed values.
def test_params_from_dict_typed() -> None:
    """Test from_dict accepts already-typed values."""
    data: Dict[str, Any] = {
        "network_context": Path("test/model.json"),
        "prompt_detail": PromptDetail.MINIMAL,
        "output": "none",
        "llm_cache": "cache.db",
    }

    params = GenerateGraphParams.from_dict(data)

    assert params.network_context == Path("test/model.json")
    assert params.prompt_detail == PromptDetail.MINIMAL
    assert params.output == "none"


# Test from_dict with case-insensitive enum values.
def test_params_from_dict_case_insensitive() -> None:
    """Test from_dict handles case-insensitive enum values."""
    data: Dict[str, Any] = {
        "network_context": "model.json",
        "prompt_detail": "STANDARD",
        "output": "none",
        "llm_cache": "cache.db",
    }

    params = GenerateGraphParams.from_dict(data)

    assert params.prompt_detail == PromptDetail.STANDARD


# Test from_dict validation error propagation.
def test_params_from_dict_validation_error() -> None:
    """Test from_dict propagates validation errors."""
    data: Dict[str, Any] = {
        "network_context": "model.json",
        "llm_model": "invalid/provider",
        "output": "none",
        "llm_cache": "cache.db",
    }

    with pytest.raises(ValidationError):
        GenerateGraphParams.from_dict(data)


# Test get_effective_cache_path with cache disabled via 'none'.
def test_params_cache_path_disabled() -> None:
    """Test get_effective_cache_path returns None when llm_cache is 'none'."""
    params = GenerateGraphParams(
        network_context=Path("test/model.json"),
        output="none",
        llm_cache="none",
    )

    assert params.get_effective_cache_path() is None


# Test get_effective_cache_path with explicit path.
def test_params_cache_path_explicit() -> None:
    """Test get_effective_cache_path returns explicit path."""
    params = GenerateGraphParams(
        network_context=Path("test/model.json"),
        output="none",
        llm_cache="custom/cache.db",
    )

    assert params.get_effective_cache_path() == Path("custom/cache.db")


# Test llm_cache accepts 'none' case-insensitively.
def test_params_llm_cache_none_case_insensitive() -> None:
    """Test llm_cache accepts 'NONE' and normalises to 'none'."""
    params = GenerateGraphParams(
        network_context=Path("test/model.json"),
        output="none",
        llm_cache="NONE",
    )

    assert params.llm_cache == "none"
    assert params.get_effective_cache_path() is None


# Test llm_cache rejects invalid suffix.
def test_params_llm_cache_invalid_suffix() -> None:
    """Test llm_cache rejects paths not ending with .db."""
    with pytest.raises(ValidationError) as exc_info:
        GenerateGraphParams(
            network_context=Path("test/model.json"),
            output="none",
            llm_cache="cache.txt",
        )

    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert ".db" in errors[0]["msg"]


# Test prompt_detail level enum values.
@pytest.mark.parametrize(
    "prompt_detail",
    [PromptDetail.MINIMAL, PromptDetail.STANDARD, PromptDetail.RICH],
)
def test_params_prompt_detail_levels(prompt_detail: PromptDetail) -> None:
    """Test all prompt_detail levels are accepted."""
    params = GenerateGraphParams(
        network_context=Path("model.json"),
        prompt_detail=prompt_detail,
        output="none",
        llm_cache="cache.db",
    )
    assert params.prompt_detail == prompt_detail


# Test output accepts 'none' case-insensitively.
def test_params_output_none_case_insensitive() -> None:
    """Test output accepts 'NONE' and normalises to 'none'."""
    params = GenerateGraphParams(
        network_context=Path("test/model.json"),
        output="NONE",
        llm_cache="cache.db",
    )

    assert params.output == "none"
    assert params.get_effective_output_path() is None


# Test output accepts .db file path for Workflow Cache.
def test_params_output_workflow_cache() -> None:
    """Test output accepts .db file path for Workflow Cache."""
    params = GenerateGraphParams(
        network_context=Path("test/model.json"),
        output="output/workflow_cache.db",
        llm_cache="cache.db",
    )

    assert params.output == "output/workflow_cache.db"
    assert params.get_effective_output_path() == Path(
        "output/workflow_cache.db"
    )


# Test output accepts .json file path for CLI usage.
def test_params_output_json_file() -> None:
    """Test output accepts .json file path for CLI usage."""
    params = GenerateGraphParams(
        network_context=Path("test/model.json"),
        output="output/graph.json",
        llm_cache="cache.db",
    )

    assert params.output == "output/graph.json"
    assert params.get_effective_output_path() == Path("output/graph.json")


# Test output accepts any path as directory output.
def test_params_output_accepts_any_path() -> None:
    """Test output accepts any path (interpreted as directory)."""
    params = GenerateGraphParams(
        network_context=Path("test/model.json"),
        output="output/results",
        llm_cache="cache.db",
    )

    assert params.output == "output/results"
    assert params.is_directory_output() is True
    assert params.is_workflow_cache_output() is False


# Test is_workflow_cache_output returns False for 'none'.
def test_params_is_workflow_cache_output_none() -> None:
    """Test is_workflow_cache_output returns False when output is 'none'."""
    params = GenerateGraphParams(
        network_context=Path("test/model.json"),
        output="none",
        llm_cache="cache.db",
    )

    assert params.is_workflow_cache_output() is False
    assert params.is_directory_output() is False
