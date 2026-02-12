"""Unit tests for graph generation prompts."""

import pytest

from causaliq_knowledge.graph.prompts import (
    ADJACENCY_MATRIX_RESPONSE_SCHEMA,
    EDGE_LIST_RESPONSE_SCHEMA,
    GRAPH_SYSTEM_PROMPT_ADJACENCY,
    GRAPH_SYSTEM_PROMPT_EDGE_LIST,
    GraphQueryPrompt,
    OutputFormat,
    _format_variable_details,
)
from causaliq_knowledge.graph.view_filter import PromptDetail


# Test OutputFormat enum values.
def test_output_format_enum_values() -> None:
    assert OutputFormat.EDGE_LIST.value == "edge_list"
    assert OutputFormat.ADJACENCY_MATRIX.value == "adjacency_matrix"


# Test OutputFormat is string subclass.
def test_output_format_is_string() -> None:
    assert isinstance(OutputFormat.EDGE_LIST, str)
    assert OutputFormat.EDGE_LIST == "edge_list"


# Test system prompt for edge list format contains required elements.
def test_edge_list_system_prompt_structure() -> None:
    assert "edges" in GRAPH_SYSTEM_PROMPT_EDGE_LIST
    assert "source" in GRAPH_SYSTEM_PROMPT_EDGE_LIST
    assert "target" in GRAPH_SYSTEM_PROMPT_EDGE_LIST
    assert "confidence" in GRAPH_SYSTEM_PROMPT_EDGE_LIST
    assert "JSON" in GRAPH_SYSTEM_PROMPT_EDGE_LIST


# Test system prompt for adjacency matrix contains required elements.
def test_adjacency_matrix_system_prompt_structure() -> None:
    assert "adjacency_matrix" in GRAPH_SYSTEM_PROMPT_ADJACENCY
    assert "variables" in GRAPH_SYSTEM_PROMPT_ADJACENCY
    assert "JSON" in GRAPH_SYSTEM_PROMPT_ADJACENCY


# Test edge list response schema has required structure.
def test_edge_list_response_schema_structure() -> None:
    assert EDGE_LIST_RESPONSE_SCHEMA["type"] == "object"
    assert "edges" in EDGE_LIST_RESPONSE_SCHEMA["required"]
    edges_schema = EDGE_LIST_RESPONSE_SCHEMA["properties"]["edges"]
    assert edges_schema["type"] == "array"


# Test adjacency matrix response schema has required structure.
def test_adjacency_matrix_response_schema_structure() -> None:
    assert ADJACENCY_MATRIX_RESPONSE_SCHEMA["type"] == "object"
    assert "variables" in ADJACENCY_MATRIX_RESPONSE_SCHEMA["required"]
    assert "adjacency_matrix" in ADJACENCY_MATRIX_RESPONSE_SCHEMA["required"]


# Test format variable details for minimal view.
def test_format_variable_details_minimal() -> None:
    variables = [{"name": "A"}, {"name": "B"}, {"name": "C"}]
    result = _format_variable_details(variables, PromptDetail.MINIMAL)
    assert "- A" in result
    assert "- B" in result
    assert "- C" in result


# Test format variable details for standard view.
def test_format_variable_details_standard() -> None:
    variables = [
        {
            "name": "smoking",
            "type": "binary",
            "short_description": "Whether patient smokes",
            "states": ["no", "yes"],
        }
    ]
    result = _format_variable_details(variables, PromptDetail.STANDARD)
    assert "- smoking" in result
    assert "Type: binary" in result
    assert "Description: Whether patient smokes" in result
    assert "States: no, yes" in result


# Test format variable details for standard view with missing fields.
def test_format_variable_details_standard_partial() -> None:
    variables = [{"name": "X", "type": "continuous"}]
    result = _format_variable_details(variables, PromptDetail.STANDARD)
    assert "- X" in result
    assert "Type: continuous" in result
    assert "Description:" not in result  # No description provided


# Test format variable details for rich view.
def test_format_variable_details_rich() -> None:
    variables = [
        {
            "name": "tobacco_use",
            "type": "binary",
            "role": "exogenous",
            "category": "exposure",
            "short_description": "Smoking history",
            "extended_description": "Long description here",
            "states": ["no", "yes"],
            "sensitivity_hints": "Risk factor",
            "related_domain_knowledge": ["Causes inflammation"],
        }
    ]
    result = _format_variable_details(variables, PromptDetail.RICH)
    assert "- tobacco_use" in result
    assert "Type: binary" in result
    assert "Role: exogenous" in result
    assert "Category: exposure" in result
    assert "Description: Smoking history" in result
    assert "Extended: Long description here" in result
    assert "States: no, yes" in result
    assert "Causal hints: Risk factor" in result
    assert "Domain knowledge: Causes inflammation" in result


# Test format variable details handles missing name.
def test_format_variable_details_missing_name() -> None:
    variables = [{"type": "binary"}]
    result = _format_variable_details(variables, PromptDetail.MINIMAL)
    assert "- unknown" in result


# Test GraphQueryPrompt build returns tuple of strings.
def test_graph_query_prompt_build_returns_tuple() -> None:
    variables = [{"name": "A"}, {"name": "B"}]
    prompt = GraphQueryPrompt(variables=variables, level=PromptDetail.MINIMAL)
    system, user = prompt.build()
    assert isinstance(system, str)
    assert isinstance(user, str)


# Test GraphQueryPrompt with edge list format uses correct system prompt.
def test_graph_query_prompt_edge_list_format() -> None:
    variables = [{"name": "A"}]
    prompt = GraphQueryPrompt(
        variables=variables,
        level=PromptDetail.MINIMAL,
        output_format=OutputFormat.EDGE_LIST,
    )
    system, _ = prompt.build()
    assert "edges" in system
    assert "source" in system


# Test GraphQueryPrompt with adjacency matrix format uses correct prompt.
def test_graph_query_prompt_adjacency_format() -> None:
    variables = [{"name": "A"}]
    prompt = GraphQueryPrompt(
        variables=variables,
        level=PromptDetail.MINIMAL,
        output_format=OutputFormat.ADJACENCY_MATRIX,
    )
    system, _ = prompt.build()
    assert "adjacency_matrix" in system


# Test GraphQueryPrompt with custom system prompt overrides default.
def test_graph_query_prompt_custom_system_prompt() -> None:
    variables = [{"name": "A"}]
    custom = "You are a custom assistant."
    prompt = GraphQueryPrompt(
        variables=variables,
        level=PromptDetail.MINIMAL,
        system_prompt=custom,
    )
    system, _ = prompt.build()
    assert system == custom


# Test GraphQueryPrompt minimal level without domain.
def test_graph_query_prompt_minimal_no_domain() -> None:
    variables = [{"name": "X"}, {"name": "Y"}, {"name": "Z"}]
    prompt = GraphQueryPrompt(variables=variables, level=PromptDetail.MINIMAL)
    _, user = prompt.build()
    assert "Variables: X, Y, Z" in user
    # Should not have "In the domain of" phrase
    assert "In the domain of" not in user


# Test GraphQueryPrompt minimal level with domain.
def test_graph_query_prompt_minimal_with_domain() -> None:
    variables = [{"name": "A"}, {"name": "B"}]
    prompt = GraphQueryPrompt(
        variables=variables,
        level=PromptDetail.MINIMAL,
        domain="epidemiology",
    )
    _, user = prompt.build()
    assert "epidemiology" in user
    assert "A, B" in user


# Test GraphQueryPrompt standard level without domain.
def test_graph_query_prompt_standard_no_domain() -> None:
    variables = [
        {"name": "smoking", "type": "binary", "short_description": "Smoker"}
    ]
    prompt = GraphQueryPrompt(variables=variables, level=PromptDetail.STANDARD)
    _, user = prompt.build()
    assert "smoking" in user
    assert "Type: binary" in user
    assert "domain" not in user.lower()


# Test GraphQueryPrompt standard level with domain.
def test_graph_query_prompt_standard_with_domain() -> None:
    variables = [{"name": "X", "type": "continuous"}]
    prompt = GraphQueryPrompt(
        variables=variables,
        level=PromptDetail.STANDARD,
        domain="climate_science",
    )
    _, user = prompt.build()
    assert "climate_science" in user


# Test GraphQueryPrompt rich level without domain.
def test_graph_query_prompt_rich_no_domain() -> None:
    variables = [
        {
            "name": "temperature",
            "type": "continuous",
            "role": "endogenous",
            "short_description": "Surface temp",
        }
    ]
    prompt = GraphQueryPrompt(variables=variables, level=PromptDetail.RICH)
    _, user = prompt.build()
    assert "temperature" in user
    assert "Role: endogenous" in user
    assert "exogenous variables have no parents" in user


# Test GraphQueryPrompt rich level with domain.
def test_graph_query_prompt_rich_with_domain() -> None:
    variables = [{"name": "A", "type": "binary", "role": "exogenous"}]
    prompt = GraphQueryPrompt(
        variables=variables,
        level=PromptDetail.RICH,
        domain="genetics",
    )
    _, user = prompt.build()
    assert "genetics" in user
    assert "Role: exogenous" in user


# Test get_variable_names returns correct list.
def test_get_variable_names() -> None:
    variables = [{"name": "X"}, {"name": "Y"}, {"name": "Z"}]
    prompt = GraphQueryPrompt(variables=variables, level=PromptDetail.MINIMAL)
    names = prompt.get_variable_names()
    assert names == ["X", "Y", "Z"]


# Test get_variable_names handles missing names.
def test_get_variable_names_missing() -> None:
    variables = [{"name": "A"}, {"type": "binary"}, {"name": "C"}]
    prompt = GraphQueryPrompt(variables=variables, level=PromptDetail.MINIMAL)
    names = prompt.get_variable_names()
    assert names == ["A", "unknown", "C"]


# Test from_context class method creates valid prompt.
def test_from_context(sample_context) -> None:
    prompt = GraphQueryPrompt.from_context(sample_context)
    system, user = prompt.build()
    assert isinstance(system, str)
    assert isinstance(user, str)
    assert prompt.domain == sample_context.domain


# Test from_context with minimal level.
def test_from_context_minimal(sample_context) -> None:
    prompt = GraphQueryPrompt.from_context(
        sample_context,
        level=PromptDetail.MINIMAL,
    )
    _, user = prompt.build()
    # Should have variable names but minimal detail
    assert "Variables:" in user


# Test from_context with adjacency matrix format.
def test_from_context_adjacency(sample_context) -> None:
    prompt = GraphQueryPrompt.from_context(
        sample_context,
        output_format=OutputFormat.ADJACENCY_MATRIX,
    )
    system, _ = prompt.build()
    assert "adjacency_matrix" in system


# Test from_context with custom system prompt.
def test_from_context_custom_system(sample_context) -> None:
    custom = "Custom system prompt."
    prompt = GraphQueryPrompt.from_context(
        sample_context,
        system_prompt=custom,
    )
    system, _ = prompt.build()
    assert system == custom


# Test default values for GraphQueryPrompt.
def test_graph_query_prompt_defaults() -> None:
    variables = [{"name": "A"}]
    prompt = GraphQueryPrompt(variables=variables)
    assert prompt.level == PromptDetail.STANDARD
    assert prompt.domain is None
    assert prompt.output_format == OutputFormat.EDGE_LIST
    assert prompt.system_prompt is None


# Fixture for sample NetworkContext.
@pytest.fixture
def sample_context():
    """Create a sample NetworkContext for testing."""
    from causaliq_knowledge.graph.models import (
        NetworkContext,
        PromptDetails,
        VariableSpec,
        VariableType,
        ViewDefinition,
    )

    return NetworkContext(
        network="test_model",
        domain="test_domain",
        prompt_details=PromptDetails(
            minimal=ViewDefinition(
                description="Minimal view",
                include_fields=["name"],
            ),
            standard=ViewDefinition(
                description="Standard view",
                include_fields=["name", "type", "short_description"],
            ),
            rich=ViewDefinition(
                description="Rich view",
                include_fields=[
                    "name",
                    "type",
                    "role",
                    "short_description",
                    "extended_description",
                ],
            ),
        ),
        variables=[
            VariableSpec(
                name="var_a",
                type=VariableType.BINARY,
                short_description="First variable",
            ),
            VariableSpec(
                name="var_b",
                type=VariableType.CONTINUOUS,
                short_description="Second variable",
            ),
        ],
    )
