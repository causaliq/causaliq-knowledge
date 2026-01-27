"""Integration tests for GraphGenerator.

These tests make real API calls to LLM providers and are marked as slow.
They will be skipped if the required API key is not set.

Run with: pytest -m slow tests/integration/test_graph_generator.py -v
"""

import pytest
from conftest import skip_no_groq

pytestmark = [pytest.mark.slow, pytest.mark.integration, skip_no_groq]


# Test GraphGenerator generates edges from simple variables.
def test_generator_generates_edges():
    from causaliq_knowledge.graph import (
        GeneratedGraph,
        GraphGenerator,
        GraphGeneratorConfig,
        ViewLevel,
    )

    config = GraphGeneratorConfig(
        temperature=0.1,
        max_tokens=500,
        view_level=ViewLevel.MINIMAL,
    )
    generator = GraphGenerator(
        model="groq/llama-3.1-8b-instant", config=config
    )

    variables = [
        {"name": "smoking"},
        {"name": "lung_cancer"},
        {"name": "age"},
    ]

    graph = generator.generate_graph(variables, domain="epidemiology")

    assert isinstance(graph, GeneratedGraph)
    assert len(graph.variables) == 3
    assert graph.metadata is not None
    assert graph.metadata.model == "llama-3.1-8b-instant"
    assert graph.metadata.provider == "groq"
    assert generator.call_count == 1

    # Should have at least one edge (smoking -> lung_cancer is well-known)
    # But we don't assert specific edges as LLM output can vary


# Test GraphGenerator with standard view level.
def test_generator_with_standard_view():
    from causaliq_knowledge.graph import (
        GeneratedGraph,
        GraphGenerator,
        GraphGeneratorConfig,
        ViewLevel,
    )

    config = GraphGeneratorConfig(
        temperature=0.1,
        max_tokens=500,
        view_level=ViewLevel.STANDARD,
    )
    generator = GraphGenerator(
        model="groq/llama-3.1-8b-instant", config=config
    )

    variables = [
        {
            "name": "education",
            "type": "ordinal",
            "short_description": "Level of formal education completed",
        },
        {
            "name": "income",
            "type": "continuous",
            "short_description": "Annual household income in USD",
        },
        {
            "name": "age",
            "type": "continuous",
            "short_description": "Age of the individual in years",
        },
    ]

    graph = generator.generate_graph(variables, domain="economics")

    assert isinstance(graph, GeneratedGraph)
    assert len(graph.variables) == 3
    assert graph.metadata is not None


# Test GraphGenerator with adjacency matrix output format.
def test_generator_adjacency_matrix_format():
    from causaliq_knowledge.graph import (
        GeneratedGraph,
        GraphGenerator,
        GraphGeneratorConfig,
        OutputFormat,
        ViewLevel,
    )

    config = GraphGeneratorConfig(
        temperature=0.1,
        max_tokens=500,
        view_level=ViewLevel.MINIMAL,
        output_format=OutputFormat.ADJACENCY_MATRIX,
    )
    generator = GraphGenerator(
        model="groq/llama-3.1-8b-instant", config=config
    )

    variables = [
        {"name": "rain"},
        {"name": "wet_ground"},
        {"name": "sprinkler"},
    ]

    graph = generator.generate_graph(variables)

    assert isinstance(graph, GeneratedGraph)
    # Verify adjacency matrix can be generated
    matrix = graph.get_adjacency_matrix()
    assert len(matrix) == 3
    assert all(len(row) == 3 for row in matrix)


# Test GraphGenerator tracks call count correctly.
def test_generator_call_count():
    from causaliq_knowledge.graph import (
        GraphGenerator,
        GraphGeneratorConfig,
        ViewLevel,
    )

    config = GraphGeneratorConfig(
        temperature=0.1,
        max_tokens=200,
        view_level=ViewLevel.MINIMAL,
    )
    generator = GraphGenerator(
        model="groq/llama-3.1-8b-instant", config=config
    )

    variables = [{"name": "a"}, {"name": "b"}]

    generator.generate_graph(variables)
    assert generator.call_count == 1

    generator.generate_graph(variables)
    assert generator.call_count == 2


# Test GraphGenerator get_stats returns expected data.
def test_generator_get_stats():
    from causaliq_knowledge.graph import (
        GraphGenerator,
        GraphGeneratorConfig,
        ViewLevel,
    )

    config = GraphGeneratorConfig(
        temperature=0.1,
        max_tokens=200,
        view_level=ViewLevel.MINIMAL,
    )
    generator = GraphGenerator(
        model="groq/llama-3.1-8b-instant", config=config
    )

    variables = [{"name": "a"}, {"name": "b"}]
    generator.generate_graph(variables)

    stats = generator.get_stats()

    assert stats["model"] == "groq/llama-3.1-8b-instant"
    assert stats["call_count"] == 1
    assert "client_call_count" in stats
