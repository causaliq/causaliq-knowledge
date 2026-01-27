"""Unit tests for ViewFilter class."""

from __future__ import annotations

from causaliq_knowledge.graph.models import (
    ModelSpec,
    VariableRole,
    VariableSpec,
    VariableType,
    ViewDefinition,
    Views,
)
from causaliq_knowledge.graph.view_filter import ViewFilter, ViewLevel


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
        views=Views(
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


# Test ViewLevel enum values.
def test_view_level_minimal_value() -> None:
    assert ViewLevel.MINIMAL.value == "minimal"


# Test ViewLevel enum standard value.
def test_view_level_standard_value() -> None:
    assert ViewLevel.STANDARD.value == "standard"


# Test ViewLevel enum rich value.
def test_view_level_rich_value() -> None:
    assert ViewLevel.RICH.value == "rich"


# Test ViewFilter initialisation stores spec.
def test_view_filter_init_stores_spec() -> None:
    spec = _create_test_spec()
    view_filter = ViewFilter(spec)
    assert view_filter.spec is spec


# Test get_include_fields returns minimal fields.
def test_get_include_fields_minimal() -> None:
    spec = _create_test_spec()
    view_filter = ViewFilter(spec)
    fields = view_filter.get_include_fields(ViewLevel.MINIMAL)
    assert fields == ["name"]


# Test get_include_fields returns standard fields.
def test_get_include_fields_standard() -> None:
    spec = _create_test_spec()
    view_filter = ViewFilter(spec)
    fields = view_filter.get_include_fields(ViewLevel.STANDARD)
    assert fields == ["name", "short_description", "type"]


# Test get_include_fields returns rich fields.
def test_get_include_fields_rich() -> None:
    spec = _create_test_spec()
    view_filter = ViewFilter(spec)
    fields = view_filter.get_include_fields(ViewLevel.RICH)
    assert "extended_description" in fields
    assert "sensitivity_hints" in fields


# Test filter_variable with minimal level.
def test_filter_variable_minimal() -> None:
    spec = _create_test_spec()
    view_filter = ViewFilter(spec)
    result = view_filter.filter_variable(spec.variables[0], ViewLevel.MINIMAL)
    assert result == {"name": "smoking"}


# Test filter_variable with standard level.
def test_filter_variable_standard() -> None:
    spec = _create_test_spec()
    view_filter = ViewFilter(spec)
    result = view_filter.filter_variable(spec.variables[0], ViewLevel.STANDARD)
    assert result["name"] == "smoking"
    assert result["short_description"] == "Daily cigarette consumption"
    assert result["type"] == "binary"


# Test filter_variable with rich level.
def test_filter_variable_rich() -> None:
    spec = _create_test_spec()
    view_filter = ViewFilter(spec)
    result = view_filter.filter_variable(spec.variables[0], ViewLevel.RICH)
    assert result["role"] == "exogenous"
    assert result["extended_description"] == "Known risk factor for cancer"


# Test filter_variable excludes None values.
def test_filter_variable_excludes_none() -> None:
    spec = _create_test_spec()
    view_filter = ViewFilter(spec)
    # Second variable has no domain_context
    result = view_filter.filter_variable(spec.variables[1], ViewLevel.RICH)
    assert "domain_context" not in result


# Test filter_variables returns all variables.
def test_filter_variables_returns_all() -> None:
    spec = _create_test_spec()
    view_filter = ViewFilter(spec)
    results = view_filter.filter_variables(ViewLevel.MINIMAL)
    assert len(results) == 2
    assert results[0] == {"name": "smoking"}
    assert results[1] == {"name": "cancer"}


# Test filter_variables with standard level.
def test_filter_variables_standard() -> None:
    spec = _create_test_spec()
    view_filter = ViewFilter(spec)
    results = view_filter.filter_variables(ViewLevel.STANDARD)
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
    summary = view_filter.get_context_summary(ViewLevel.MINIMAL)
    assert summary["domain"] == "epidemiology"
    assert summary["dataset_id"] == "test"
    assert len(summary["variables"]) == 2


# Test get_context_summary includes filtered variables.
def test_get_context_summary_variables_filtered() -> None:
    spec = _create_test_spec()
    view_filter = ViewFilter(spec)
    summary = view_filter.get_context_summary(ViewLevel.MINIMAL)
    for var in summary["variables"]:
        assert list(var.keys()) == ["name"]


# Test ViewLevel is string enum.
def test_view_level_is_string_enum() -> None:
    assert ViewLevel.MINIMAL.value == "minimal"
    assert ViewLevel.STANDARD.value == "standard"
    assert ViewLevel.RICH.value == "rich"
