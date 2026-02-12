"""Tests for graph response models and parsing."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from causaliq_knowledge.graph.response import (
    GeneratedGraph,
    GenerationMetadata,
    ProposedEdge,
    parse_adjacency_matrix_response,
    parse_edge_list_response,
    parse_graph_response,
)

# --- ProposedEdge tests ---


# Test ProposedEdge creation with basic attributes.
def test_proposed_edge_creation() -> None:
    edge = ProposedEdge(
        source="smoking",
        target="lung_cancer",
        confidence=0.95,
    )
    assert edge.source == "smoking"
    assert edge.target == "lung_cancer"
    assert edge.confidence == 0.95
    assert edge.reasoning is None


# Test ProposedEdge with reasoning.
def test_proposed_edge_with_reasoning() -> None:
    edge = ProposedEdge(
        source="age",
        target="income",
        confidence=0.7,
        reasoning="Age typically correlates with career progression",
    )
    assert edge.reasoning == "Age typically correlates with career progression"


# Test ProposedEdge default confidence.
def test_proposed_edge_default_confidence() -> None:
    edge = ProposedEdge(source="a", target="b")
    assert edge.confidence == 0.5


# Test ProposedEdge confidence clamping above 1.0.
def test_proposed_edge_confidence_clamped_high() -> None:
    edge = ProposedEdge(source="a", target="b", confidence=1.5)
    assert edge.confidence == 1.0


# Test ProposedEdge confidence clamping below 0.0.
def test_proposed_edge_confidence_clamped_low() -> None:
    edge = ProposedEdge(source="a", target="b", confidence=-0.5)
    assert edge.confidence == 0.0


# Test ProposedEdge confidence with None defaults to 0.5.
def test_proposed_edge_confidence_none() -> None:
    edge = ProposedEdge(
        source="a",
        target="b",
        confidence=None,  # type: ignore
    )
    assert edge.confidence == 0.5


# Test ProposedEdge confidence with invalid type defaults to 0.5.
def test_proposed_edge_confidence_invalid_type() -> None:
    edge = ProposedEdge(
        source="a",
        target="b",
        confidence="invalid",  # type: ignore
    )
    assert edge.confidence == 0.5


# --- GenerationMetadata tests ---


# Test GenerationMetadata creation with required fields.
def test_generation_metadata_creation() -> None:
    metadata = GenerationMetadata(model="llama-3.1-8b-instant")
    assert metadata.model == "llama-3.1-8b-instant"
    assert metadata.provider == ""
    assert metadata.llm_latency_ms == 0
    assert metadata.from_cache is False


# Test GenerationMetadata with all fields.
def test_generation_metadata_all_fields() -> None:
    ts = datetime(2026, 1, 27, 12, 0, 0, tzinfo=timezone.utc)
    llm_ts = datetime(2026, 1, 27, 11, 0, 0, tzinfo=timezone.utc)
    metadata = GenerationMetadata(
        model="gpt-4o",
        provider="openai",
        timestamp=ts,
        llm_timestamp=llm_ts,
        llm_latency_ms=1500,
        input_tokens=500,
        output_tokens=200,
        from_cache=True,
    )
    assert metadata.model == "gpt-4o"
    assert metadata.provider == "openai"
    assert metadata.timestamp == ts
    assert metadata.llm_timestamp == llm_ts
    assert metadata.llm_latency_ms == 1500
    assert metadata.input_tokens == 500
    assert metadata.output_tokens == 200
    assert metadata.from_cache is True


# Test GenerationMetadata to_dict returns all fields correctly.
def test_generation_metadata_to_dict() -> None:
    ts = datetime(2026, 1, 27, 12, 0, 0, tzinfo=timezone.utc)
    llm_ts = datetime(2026, 1, 27, 11, 0, 0, tzinfo=timezone.utc)
    messages = [{"role": "user", "content": "test prompt"}]
    metadata = GenerationMetadata(
        model="gpt-4o",
        provider="openai",
        timestamp=ts,
        llm_timestamp=llm_ts,
        llm_latency_ms=1500,
        input_tokens=500,
        output_tokens=200,
        from_cache=True,
        messages=messages,
        temperature=0.2,
        max_tokens=4000,
        finish_reason="stop",
        llm_cost_usd=0.015,
    )
    result = metadata.to_dict()
    assert result["llm_model"] == "gpt-4o"
    assert result["llm_provider"] == "openai"
    assert result["timestamp"] == "2026-01-27T12:00:00+00:00"
    assert result["llm_timestamp"] == "2026-01-27T11:00:00+00:00"
    assert result["llm_latency_ms"] == 1500
    assert result["llm_input_tokens"] == 500
    assert result["llm_output_tokens"] == 200
    assert result["from_cache"] is True
    assert result["llm_messages"] == messages
    assert result["llm_temperature"] == 0.2
    assert result["llm_max_tokens"] == 4000
    assert result["llm_finish_reason"] == "stop"
    assert result["llm_cost_usd"] == 0.015


# Test to_dict splits message content with newlines into arrays.
def test_generation_metadata_to_dict_splits_newlines() -> None:
    messages = [
        {"role": "system", "content": "Line 1\nLine 2\nLine 3"},
        {"role": "user", "content": "Simple content"},
    ]
    metadata = GenerationMetadata(model="test", messages=messages)
    result = metadata.to_dict()

    # Content with newlines should be split into array
    assert result["llm_messages"][0]["content"] == [
        "Line 1",
        "Line 2",
        "Line 3",
    ]
    # Content without newlines should remain as string
    assert result["llm_messages"][1]["content"] == "Simple content"


# Test backward compatibility alias initial_cost_usd.
def test_generation_metadata_initial_cost_usd_alias() -> None:
    metadata = GenerationMetadata(model="test", llm_cost_usd=0.025)
    assert metadata.initial_cost_usd == 0.025


# Test backward compatibility alias latency_ms.
def test_generation_metadata_latency_ms_alias() -> None:
    metadata = GenerationMetadata(model="test", llm_latency_ms=1500)
    assert metadata.latency_ms == 1500


# --- GeneratedGraph tests ---


# Test GeneratedGraph creation with edges and variables.
def test_generated_graph_creation() -> None:
    edges = [
        ProposedEdge(source="a", target="b", confidence=0.8),
        ProposedEdge(source="b", target="c", confidence=0.6),
    ]
    graph = GeneratedGraph(
        edges=edges,
        variables=["a", "b", "c"],
        reasoning="Test reasoning",
    )
    assert len(graph.edges) == 2
    assert graph.variables == ["a", "b", "c"]
    assert graph.reasoning == "Test reasoning"


# Test GeneratedGraph get_edge_list method.
def test_generated_graph_get_edge_list() -> None:
    edges = [
        ProposedEdge(source="a", target="b", confidence=0.8),
        ProposedEdge(source="b", target="c", confidence=0.6),
    ]
    graph = GeneratedGraph(edges=edges, variables=["a", "b", "c"])
    edge_list = graph.get_edge_list()
    assert edge_list == [("a", "b", 0.8), ("b", "c", 0.6)]


# Test GeneratedGraph get_adjacency_matrix method.
def test_generated_graph_get_adjacency_matrix() -> None:
    edges = [
        ProposedEdge(source="a", target="b", confidence=0.8),
        ProposedEdge(source="b", target="c", confidence=0.6),
    ]
    graph = GeneratedGraph(edges=edges, variables=["a", "b", "c"])
    matrix = graph.get_adjacency_matrix()
    # a=0, b=1, c=2
    # a->b: [0][1] = 0.8
    # b->c: [1][2] = 0.6
    expected = [
        [0.0, 0.8, 0.0],
        [0.0, 0.0, 0.6],
        [0.0, 0.0, 0.0],
    ]
    assert matrix == expected


# Test GeneratedGraph adjacency matrix with unknown variable in edge.
def test_generated_graph_adjacency_matrix_unknown_variable() -> None:
    edges = [
        ProposedEdge(source="a", target="b", confidence=0.8),
        ProposedEdge(source="unknown", target="c", confidence=0.5),
    ]
    graph = GeneratedGraph(edges=edges, variables=["a", "b", "c"])
    matrix = graph.get_adjacency_matrix()
    # unknown source is ignored
    expected = [
        [0.0, 0.8, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]
    assert matrix == expected


# Test GeneratedGraph filter_by_confidence method.
def test_generated_graph_filter_by_confidence() -> None:
    edges = [
        ProposedEdge(source="a", target="b", confidence=0.8),
        ProposedEdge(source="b", target="c", confidence=0.4),
        ProposedEdge(source="a", target="c", confidence=0.6),
    ]
    graph = GeneratedGraph(edges=edges, variables=["a", "b", "c"])
    filtered = graph.filter_by_confidence(threshold=0.5)
    assert len(filtered.edges) == 2
    assert filtered.edges[0].confidence == 0.8
    assert filtered.edges[1].confidence == 0.6


# Test GeneratedGraph filter preserves metadata.
def test_generated_graph_filter_preserves_metadata() -> None:
    metadata = GenerationMetadata(model="test-model")
    graph = GeneratedGraph(
        edges=[ProposedEdge(source="a", target="b", confidence=0.8)],
        variables=["a", "b"],
        reasoning="test",
        metadata=metadata,
    )
    filtered = graph.filter_by_confidence(0.5)
    assert filtered.metadata == metadata
    assert filtered.reasoning == "test"


# --- parse_edge_list_response tests ---


# Test parsing valid edge list response.
def test_parse_edge_list_response_valid() -> None:
    response = {
        "edges": [
            {"source": "a", "target": "b", "confidence": 0.8},
            {"source": "b", "target": "c", "confidence": 0.6},
        ],
        "reasoning": "Based on domain knowledge",
    }
    graph = parse_edge_list_response(response, ["a", "b", "c"])
    assert len(graph.edges) == 2
    assert graph.edges[0].source == "a"
    assert graph.edges[0].target == "b"
    assert graph.reasoning == "Based on domain knowledge"


# Test parsing edge list skips unknown source variable.
def test_parse_edge_list_skips_unknown_source() -> None:
    response = {
        "edges": [
            {"source": "unknown", "target": "b", "confidence": 0.8},
            {"source": "a", "target": "b", "confidence": 0.6},
        ],
    }
    graph = parse_edge_list_response(response, ["a", "b"])
    assert len(graph.edges) == 1
    assert graph.edges[0].source == "a"


# Test parsing edge list skips unknown target variable.
def test_parse_edge_list_skips_unknown_target() -> None:
    response = {
        "edges": [
            {"source": "a", "target": "unknown", "confidence": 0.8},
        ],
    }
    graph = parse_edge_list_response(response, ["a", "b"])
    assert len(graph.edges) == 0


# Test parsing edge list skips self-loops.
def test_parse_edge_list_skips_self_loops() -> None:
    response = {
        "edges": [
            {"source": "a", "target": "a", "confidence": 0.8},
            {"source": "a", "target": "b", "confidence": 0.6},
        ],
    }
    graph = parse_edge_list_response(response, ["a", "b"])
    assert len(graph.edges) == 1
    assert graph.edges[0].target == "b"


# Test parsing edge list skips invalid edge entries.
def test_parse_edge_list_skips_invalid_entries() -> None:
    response = {
        "edges": [
            "not a dict",
            {"source": "a", "target": "b", "confidence": 0.6},
        ],
    }
    graph = parse_edge_list_response(response, ["a", "b"])
    assert len(graph.edges) == 1


# Test parsing edge list with missing edges key.
def test_parse_edge_list_empty_edges() -> None:
    response = {"reasoning": "No edges found"}
    graph = parse_edge_list_response(response, ["a", "b"])
    assert len(graph.edges) == 0
    assert graph.reasoning == "No edges found"


# Test parsing edge list raises on non-dict response.
def test_parse_edge_list_raises_on_non_dict() -> None:
    with pytest.raises(ValueError, match="Expected dict response"):
        parse_edge_list_response("not a dict", ["a", "b"])  # type: ignore


# Test parsing edge list raises on non-list edges.
def test_parse_edge_list_raises_on_non_list_edges() -> None:
    with pytest.raises(ValueError, match="Expected 'edges' to be a list"):
        parse_edge_list_response({"edges": "not a list"}, ["a", "b"])


# --- parse_adjacency_matrix_response tests ---


# Test parsing valid adjacency matrix response.
def test_parse_adjacency_matrix_valid() -> None:
    response = {
        "variables": ["a", "b", "c"],
        "adjacency_matrix": [
            [0.0, 0.8, 0.0],
            [0.0, 0.0, 0.6],
            [0.0, 0.0, 0.0],
        ],
        "reasoning": "Matrix representation",
    }
    graph = parse_adjacency_matrix_response(response, ["a", "b", "c"])
    assert len(graph.edges) == 2
    assert graph.variables == ["a", "b", "c"]
    assert graph.reasoning == "Matrix representation"


# Test adjacency matrix extracts correct edges.
def test_parse_adjacency_matrix_extracts_edges() -> None:
    response = {
        "variables": ["x", "y"],
        "adjacency_matrix": [
            [0.0, 0.9],
            [0.3, 0.0],
        ],
    }
    graph = parse_adjacency_matrix_response(response, ["x", "y"])
    assert len(graph.edges) == 2
    # x->y with 0.9
    # y->x with 0.3
    edge_tuples = {(e.source, e.target, e.confidence) for e in graph.edges}
    assert ("x", "y", 0.9) in edge_tuples
    assert ("y", "x", 0.3) in edge_tuples


# Test adjacency matrix ignores zero entries.
def test_parse_adjacency_matrix_ignores_zeros() -> None:
    response = {
        "variables": ["a", "b"],
        "adjacency_matrix": [
            [0.0, 0.0],
            [0.0, 0.0],
        ],
    }
    graph = parse_adjacency_matrix_response(response, ["a", "b"])
    assert len(graph.edges) == 0


# Test adjacency matrix ignores diagonal (self-loops).
def test_parse_adjacency_matrix_ignores_diagonal() -> None:
    response = {
        "variables": ["a", "b"],
        "adjacency_matrix": [
            [0.9, 0.5],
            [0.0, 0.8],
        ],
    }
    graph = parse_adjacency_matrix_response(response, ["a", "b"])
    assert len(graph.edges) == 1
    assert graph.edges[0].source == "a"
    assert graph.edges[0].target == "b"


# Test adjacency matrix raises on non-dict response.
def test_parse_adjacency_matrix_raises_on_non_dict() -> None:
    with pytest.raises(ValueError, match="Expected dict response"):
        parse_adjacency_matrix_response([], ["a", "b"])  # type: ignore


# Test adjacency matrix raises on wrong row count.
def test_parse_adjacency_matrix_raises_on_wrong_row_count() -> None:
    response = {
        "variables": ["a", "b", "c"],
        "adjacency_matrix": [
            [0.0, 0.8],
            [0.0, 0.0],
        ],
    }
    with pytest.raises(ValueError, match="Matrix has 2 rows but 3 variables"):
        parse_adjacency_matrix_response(response, ["a", "b", "c"])


# Test adjacency matrix raises on wrong column count.
def test_parse_adjacency_matrix_raises_on_wrong_column_count() -> None:
    response = {
        "variables": ["a", "b"],
        "adjacency_matrix": [
            [0.0, 0.8, 0.5],
            [0.0, 0.0],
        ],
    }
    with pytest.raises(ValueError, match="row 0 has incorrect dimensions"):
        parse_adjacency_matrix_response(response, ["a", "b"])


# --- parse_graph_response tests ---


# Test parse_graph_response with edge list format.
def test_parse_graph_response_edge_list() -> None:
    response_text = '{"edges": [{"source": "a", "target": "b"}]}'
    graph = parse_graph_response(response_text, ["a", "b"], "edge_list")
    assert len(graph.edges) == 1


# Test parse_graph_response with adjacency matrix format.
def test_parse_graph_response_adjacency_matrix() -> None:
    response_text = """{
        "variables": ["a", "b"],
        "adjacency_matrix": [[0.0, 0.8], [0.0, 0.0]]
    }"""
    graph = parse_graph_response(response_text, ["a", "b"], "adjacency_matrix")
    assert len(graph.edges) == 1


# Test parse_graph_response strips markdown code blocks.
def test_parse_graph_response_strips_markdown_json() -> None:
    response_text = '```json\n{"edges": []}\n```'
    graph = parse_graph_response(response_text, ["a", "b"], "edge_list")
    assert len(graph.edges) == 0


# Test parse_graph_response strips plain markdown blocks.
def test_parse_graph_response_strips_markdown_plain() -> None:
    response_text = '```\n{"edges": []}\n```'
    graph = parse_graph_response(response_text, ["a", "b"], "edge_list")
    assert len(graph.edges) == 0


# Test parse_graph_response raises on invalid JSON.
def test_parse_graph_response_raises_on_invalid_json() -> None:
    with pytest.raises(ValueError, match="Failed to parse JSON"):
        parse_graph_response("not valid json", ["a", "b"], "edge_list")


# --- Additional coverage tests ---


# Test edge list parsing converts non-string reasoning to string.
def test_parse_edge_list_converts_non_string_reasoning() -> None:
    response = {
        "edges": [],
        "reasoning": 12345,  # Non-string reasoning
    }
    graph = parse_edge_list_response(response, ["a", "b"])
    assert graph.reasoning == "12345"


# Test adjacency matrix raises on non-list variables.
def test_parse_adjacency_matrix_raises_on_non_list_variables() -> None:
    response = {
        "variables": "not a list",
        "adjacency_matrix": [],
    }
    with pytest.raises(ValueError, match="Expected 'variables' to be a list"):
        parse_adjacency_matrix_response(response, ["a", "b"])


# Test adjacency matrix raises on non-list matrix.
def test_parse_adjacency_matrix_raises_on_non_list_matrix() -> None:
    response = {
        "variables": ["a", "b"],
        "adjacency_matrix": "not a list",
    }
    with pytest.raises(
        ValueError, match="Expected 'adjacency_matrix' to be a list"
    ):
        parse_adjacency_matrix_response(response, ["a", "b"])


# Test adjacency matrix converts non-string reasoning to string.
def test_parse_adjacency_matrix_converts_non_string_reasoning() -> None:
    response = {
        "variables": ["a", "b"],
        "adjacency_matrix": [[0.0, 0.0], [0.0, 0.0]],
        "reasoning": {"key": "value"},  # Non-string reasoning
    }
    graph = parse_adjacency_matrix_response(response, ["a", "b"])
    assert "key" in graph.reasoning


# Test adjacency matrix warns on unexpected variable.
def test_parse_adjacency_matrix_warns_on_unexpected_variable() -> None:
    response = {
        "variables": ["a", "unexpected"],
        "adjacency_matrix": [[0.0, 0.5], [0.0, 0.0]],
    }
    # Should not raise, just warn and continue
    graph = parse_adjacency_matrix_response(response, ["a", "b"])
    assert len(graph.variables) == 2


# Test adjacency matrix skips non-numeric confidence values.
def test_parse_adjacency_matrix_skips_non_numeric_confidence() -> None:
    response = {
        "variables": ["a", "b"],
        "adjacency_matrix": [[0.0, "invalid"], [0.0, 0.0]],
    }
    graph = parse_adjacency_matrix_response(response, ["a", "b"])
    # "invalid" should be skipped, so no edges
    assert len(graph.edges) == 0
