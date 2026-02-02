"""Unit tests for ViewFilter class."""

from __future__ import annotations

from causaliq_knowledge.graph.models import (
    ModelSpec,
    PromptDetails,
    VariableRole,
    VariableSpec,
    VariableType,
    ViewDefinition,
)
from causaliq_knowledge.graph.view_filter import PromptDetail, ViewFilter


def _create_test_spec() -> ModelSpec:
    """Create a test model specification."""
    return ModelSpec(
        dataset_id="test",
        domain="epidemiology",
        variables=[
            VariableSpec(
                name="smoking",
                short_description="Daily cigarette consumption",
                type=VariableType.BINARY,
                role=VariableRole.EXOGENOUS,
                extended_description="Known risk factor for cancer",
                sensitivity_hints="Directly causes lung damage",
            ),
            VariableSpec(
                name="cancer",
                short_description="Lung cancer diagnosis",
                type=VariableType.BINARY,
                role=VariableRole.ENDOGENOUS,
            ),
        ],
        prompt_details=PromptDetails(
            minimal=ViewDefinition(include_fields=["name"]),
            standard=ViewDefinition(
                include_fields=["name", "short_description", "type"]
            ),
            rich=ViewDefinition(
                include_fields=[
                    "name",
                    "short_description",
                    "type",
                    "role",
                    "extended_description",
                    "sensitivity_hints",
                ]
            ),
        ),
    )


# Test PromptDetail enum values.
def test_view_level_minimal_value() -> None:
    assert PromptDetail.MINIMAL.value == "minimal"


# Test PromptDetail enum standard value.
def test_view_level_standard_value() -> None:
    assert PromptDetail.STANDARD.value == "standard"


# Test PromptDetail enum rich value.
def test_view_level_rich_value() -> None:
    assert PromptDetail.RICH.value == "rich"


# Test ViewFilter initialisation stores spec.
def test_view_filter_init_stores_spec() -> None:
    spec = _create_test_spec()
    view_filter = ViewFilter(spec)
    assert view_filter.spec is spec


# Test get_include_fields returns minimal fields.
def test_get_include_fields_minimal() -> None:
    spec = _create_test_spec()
    view_filter = ViewFilter(spec)
    fields = view_filter.get_include_fields(PromptDetail.MINIMAL)
    assert fields == ["name"]


# Test get_include_fields returns standard fields.
def test_get_include_fields_standard() -> None:
    spec = _create_test_spec()
    view_filter = ViewFilter(spec)
    fields = view_filter.get_include_fields(PromptDetail.STANDARD)
    assert fields == ["name", "short_description", "type"]


# Test get_include_fields returns rich fields.
def test_get_include_fields_rich() -> None:
    spec = _create_test_spec()
    view_filter = ViewFilter(spec)
    fields = view_filter.get_include_fields(PromptDetail.RICH)
    assert "extended_description" in fields
    assert "sensitivity_hints" in fields


# Test filter_variable with minimal level.
def test_filter_variable_minimal() -> None:
    spec = _create_test_spec()
    view_filter = ViewFilter(spec)
    result = view_filter.filter_variable(
        spec.variables[0], PromptDetail.MINIMAL
    )
    assert result == {"name": "smoking"}


# Test filter_variable with standard level.
def test_filter_variable_standard() -> None:
    spec = _create_test_spec()
    view_filter = ViewFilter(spec)
    result = view_filter.filter_variable(
        spec.variables[0], PromptDetail.STANDARD
    )
    assert result["name"] == "smoking"
    assert result["short_description"] == "Daily cigarette consumption"
    assert result["type"] == "binary"


# Test filter_variable with rich level.
def test_filter_variable_rich() -> None:
    spec = _create_test_spec()
    view_filter = ViewFilter(spec)
    result = view_filter.filter_variable(spec.variables[0], PromptDetail.RICH)
    assert result["role"] == "exogenous"
    assert result["extended_description"] == "Known risk factor for cancer"


# Test filter_variable excludes None values.
def test_filter_variable_excludes_none() -> None:
    spec = _create_test_spec()
    view_filter = ViewFilter(spec)
    # Second variable has no domain_context
    result = view_filter.filter_variable(spec.variables[1], PromptDetail.RICH)
    assert "domain_context" not in result


# Test filter_variables returns all variables.
def test_filter_variables_returns_all() -> None:
    spec = _create_test_spec()
    view_filter = ViewFilter(spec)
    results = view_filter.filter_variables(PromptDetail.MINIMAL)
    assert len(results) == 2
    assert results[0] == {"name": "smoking"}
    assert results[1] == {"name": "cancer"}


# Test filter_variables with standard level.
def test_filter_variables_standard() -> None:
    spec = _create_test_spec()
    view_filter = ViewFilter(spec)
    results = view_filter.filter_variables(PromptDetail.STANDARD)
    assert all("short_description" in r for r in results)


# Test get_variable_names returns all names.
def test_get_variable_names() -> None:
    spec = _create_test_spec()
    view_filter = ViewFilter(spec)
    names = view_filter.get_variable_names()
    assert names == ["smoking", "cancer"]


# Test get_domain returns domain string.
def test_get_domain() -> None:
    spec = _create_test_spec()
    view_filter = ViewFilter(spec)
    assert view_filter.get_domain() == "epidemiology"


# Test get_context_summary with minimal level.
def test_get_context_summary_minimal() -> None:
    spec = _create_test_spec()
    view_filter = ViewFilter(spec)
    summary = view_filter.get_context_summary(PromptDetail.MINIMAL)
    assert summary["domain"] == "epidemiology"
    assert summary["dataset_id"] == "test"
    assert len(summary["variables"]) == 2


# Test get_context_summary includes filtered variables.
def test_get_context_summary_variables_filtered() -> None:
    spec = _create_test_spec()
    view_filter = ViewFilter(spec)
    summary = view_filter.get_context_summary(PromptDetail.MINIMAL)
    for var in summary["variables"]:
        assert list(var.keys()) == ["name"]


# Test PromptDetail is string enum.
def test_view_level_is_string_enum() -> None:
    assert PromptDetail.MINIMAL.value == "minimal"
    assert PromptDetail.STANDARD.value == "standard"
    assert PromptDetail.RICH.value == "rich"


# Test get_variable_names returns benchmark names when use_llm_names=False.
def test_get_variable_names_with_use_llm_names_false() -> None:
    spec = ModelSpec(
        dataset_id="test",
        domain="test",
        variables=[
            VariableSpec(name="smoke", llm_name="tobacco_use", type="binary"),
            VariableSpec(name="lung", llm_name="cancer_status", type="binary"),
        ],
    )
    view_filter = ViewFilter(spec, use_llm_names=False)
    names = view_filter.get_variable_names()
    # Should return benchmark names, not llm_names
    assert names == ["smoke", "lung"]
