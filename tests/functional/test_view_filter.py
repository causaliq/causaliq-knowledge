"""Functional tests for ViewFilter using tracked test files."""

from __future__ import annotations

from pathlib import Path

from causaliq_knowledge.graph import (
    NetworkContext,
    PromptDetail,
    ViewFilter,
)

# Path to test model files.
TEST_MODELS_DIR = (
    Path(__file__).parent.parent / "data" / "functional" / "models"
)


# Test ViewFilter with simple_chain model minimal view.
def test_view_filter_simple_chain_minimal() -> None:
    spec = NetworkContext.load(TEST_MODELS_DIR / "simple_chain.json")
    view_filter = ViewFilter(spec)
    results = view_filter.filter_variables(PromptDetail.MINIMAL)
    assert len(results) == 3
    # Minimal view should only have name field
    for var in results:
        assert "name" in var
        assert len(var) == 1


# Test ViewFilter with simple_chain model standard view.
def test_view_filter_simple_chain_standard() -> None:
    spec = NetworkContext.load(TEST_MODELS_DIR / "simple_chain.json")
    view_filter = ViewFilter(spec)
    results = view_filter.filter_variables(PromptDetail.STANDARD)
    assert len(results) == 3
    # Standard view includes name, type, short_description
    for var in results:
        assert "name" in var
        assert "type" in var
        assert "short_description" in var


# Test ViewFilter with collider model has custom views defined.
def test_view_filter_collider_custom_views() -> None:
    spec = NetworkContext.load(TEST_MODELS_DIR / "collider.json")
    view_filter = ViewFilter(spec)
    # Collider has explicit view definitions
    minimal_fields = view_filter.get_include_fields(PromptDetail.MINIMAL)
    assert minimal_fields == ["name"]
    standard_fields = view_filter.get_include_fields(PromptDetail.STANDARD)
    assert "states" in standard_fields


# Test ViewFilter collider model rich view includes extended description.
def test_view_filter_collider_rich_view() -> None:
    spec = NetworkContext.load(TEST_MODELS_DIR / "collider.json")
    view_filter = ViewFilter(spec)
    results = view_filter.filter_variables(PromptDetail.RICH)
    # genetic_factor has extended_description
    genetic = next(v for v in results if v["name"] == "genetic_factor")
    assert "extended_description" in genetic
    assert "risk" in genetic["extended_description"]


# Test ViewFilter get_context_summary returns complete structure.
def test_view_filter_context_summary_structure() -> None:
    spec = NetworkContext.load(TEST_MODELS_DIR / "collider.json")
    view_filter = ViewFilter(spec)
    summary = view_filter.get_context_summary(PromptDetail.STANDARD)
    assert summary["domain"] == "epidemiology"
    assert summary["dataset_id"] == "collider"
    assert len(summary["variables"]) == 3


# Test ViewFilter get_variable_names from loaded file.
def test_view_filter_get_variable_names() -> None:
    spec = NetworkContext.load(TEST_MODELS_DIR / "simple_chain.json")
    view_filter = ViewFilter(spec)
    names = view_filter.get_variable_names()
    assert names == ["cause", "mediator", "effect"]


# Test ViewFilter get_domain from loaded file.
def test_view_filter_get_domain() -> None:
    spec = NetworkContext.load(TEST_MODELS_DIR / "collider.json")
    view_filter = ViewFilter(spec)
    assert view_filter.get_domain() == "epidemiology"


# Test ViewFilter minimal view excludes None fields.
def test_view_filter_minimal_excludes_none_fields() -> None:
    spec = NetworkContext.load(TEST_MODELS_DIR / "minimal.json")
    view_filter = ViewFilter(spec)
    results = view_filter.filter_variables(PromptDetail.RICH)
    # Minimal model has variables without extended descriptions
    for var in results:
        # None values should not be present in filtered output
        assert all(v is not None for v in var.values())
