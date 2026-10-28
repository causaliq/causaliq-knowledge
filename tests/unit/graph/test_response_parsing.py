"""Tests for graph response parsing functions."""

from __future__ import annotations

import pytest

from causaliq_knowledge.graph.response import (
    parse_adjacency_matrix_response,
    parse_edge_list_response,
    parse_graph_response,
    parse_pdg_response,
)

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


# =============================================================================
# PDG Response Parsing Tests
# =============================================================================


# Test parse_pdg_response with valid response.
def test_parse_pdg_response_basic() -> None:
    from causaliq_core.graph.pdg import PDG

    response_text = """
    {
        "edges": [
            {
                "source": "smoking",
                "target": "cancer",
                "existence": 0.9,
                "orientation": 0.8
            }
        ],
        "reasoning": "Smoking causes cancer"
    }
    """
    pdg = parse_pdg_response(response_text, ["smoking", "cancer"])

    assert isinstance(pdg, PDG)
    assert len(pdg.nodes) == 2
    assert "smoking" in pdg.nodes
    assert "cancer" in pdg.nodes
    assert len(pdg.edges) == 1

    # Check probability conversion
    # p_forward = 0.9 * 0.8 = 0.72
    # p_backward = 0.9 * 0.2 = 0.18
    # p_none = 0.1
    key = ("cancer", "smoking")  # Canonical order (alphabetical)
    assert key in pdg.edges
    probs = pdg.edges[key]
    # Because "cancer" < "smoking", forward/backward are swapped
    assert abs(probs.backward - 0.72) < 0.001
    assert abs(probs.forward - 0.18) < 0.001
    assert probs.undirected == 0.0
    assert abs(probs.none - 0.1) < 0.001


# Test parse_pdg_response with markdown code blocks.
def test_parse_pdg_response_with_markdown() -> None:
    response_text = """```json
    {
        "edges": [
            {
                "source": "a", "target": "b",
                "existence": 0.5, "orientation": 0.5
            }
        ]
    }
    ```"""
    pdg = parse_pdg_response(response_text, ["a", "b"])
    assert len(pdg.edges) == 1


# Test parse_pdg_response handles invalid JSON.
def test_parse_pdg_response_invalid_json() -> None:
    with pytest.raises(ValueError) as exc_info:
        parse_pdg_response("not valid json", ["a", "b"])
    assert "failed to parse json" in str(exc_info.value).lower()


# Test parse_pdg_response rejects non-dict response.
def test_parse_pdg_response_non_dict() -> None:
    with pytest.raises(ValueError) as exc_info:
        parse_pdg_response("[]", ["a", "b"])
    assert "expected dict" in str(exc_info.value).lower()


# Test parse_pdg_response rejects non-list edges.
def test_parse_pdg_response_non_list_edges() -> None:
    with pytest.raises(ValueError) as exc_info:
        parse_pdg_response('{"edges": "not a list"}', ["a", "b"])
    assert "expected 'edges' to be a list" in str(exc_info.value).lower()


# Test parse_pdg_response skips invalid edge entries.
def test_parse_pdg_response_skips_invalid_edges() -> None:
    response_text = """
    {
        "edges": [
            "not a dict",
            {
                "source": "a", "target": "b",
                "existence": 0.5, "orientation": 0.5
            }
        ]
    }
    """
    pdg = parse_pdg_response(response_text, ["a", "b"])
    # Should have 1 valid edge, the invalid one is skipped
    assert len(pdg.edges) == 1


# Test parse_pdg_response skips unknown source variable.
def test_parse_pdg_response_skips_unknown_source() -> None:
    response_text = """
    {
        "edges": [
            {
                "source": "unknown", "target": "b",
                "existence": 0.9, "orientation": 0.5
            }
        ]
    }
    """
    pdg = parse_pdg_response(response_text, ["a", "b"])
    # Edge with unknown source should be skipped
    assert len(pdg.edges) == 0


# Test parse_pdg_response skips unknown target variable.
def test_parse_pdg_response_skips_unknown_target() -> None:
    response_text = """
    {
        "edges": [
            {
                "source": "a", "target": "unknown",
                "existence": 0.9, "orientation": 0.5
            }
        ]
    }
    """
    pdg = parse_pdg_response(response_text, ["a", "b"])
    # Edge with unknown target should be skipped
    assert len(pdg.edges) == 0


# Test parse_pdg_response skips self-loops.
def test_parse_pdg_response_skips_self_loops() -> None:
    response_text = """
    {
        "edges": [
            {
                "source": "a", "target": "a",
                "existence": 0.9, "orientation": 0.5
            }
        ]
    }
    """
    pdg = parse_pdg_response(response_text, ["a", "b"])
    assert len(pdg.edges) == 0


# Test parse_pdg_response handles invalid probability values.
def test_parse_pdg_response_clamps_invalid_probabilities() -> None:
    response_text = """
    {
        "edges": [
            {
                "source": "a", "target": "b",
                "existence": 1.5, "orientation": -0.5
            }
        ]
    }
    """
    pdg = parse_pdg_response(response_text, ["a", "b"])
    # Values should be clamped to [0, 1]
    probs = pdg.edges[("a", "b")]
    assert probs.p_exist == 1.0  # Clamped from 1.5
    # orientation clamped to 0.0 means p_forward = 0, p_backward = 1.0


# Test parse_pdg_response handles non-numeric probability values.
def test_parse_pdg_response_handles_non_numeric_probabilities() -> None:
    response_text = """
    {
        "edges": [
            {
                "source": "a", "target": "b",
                "existence": "high", "orientation": "left"
            }
        ]
    }
    """
    pdg = parse_pdg_response(response_text, ["a", "b"])
    # Should fall back to default 0.5 for both
    probs = pdg.edges[("a", "b")]
    assert probs.p_exist == 0.5


# Test parse_pdg_response keeps first edge when it has higher existence.
def test_parse_pdg_response_keeps_first_higher_existence() -> None:
    response_text = """
    {
        "edges": [
            {
                "source": "a", "target": "b",
                "existence": 0.9, "orientation": 0.5
            },
            {
                "source": "b", "target": "a",
                "existence": 0.5, "orientation": 0.7
            }
        ]
    }
    """
    pdg = parse_pdg_response(response_text, ["a", "b"])
    # Should keep first edge (higher existence 0.9 > 0.5)
    assert len(pdg.edges) == 1
    probs = pdg.edges[("a", "b")]
    # First edge: existence=0.9, orientation=0.5
    # p_forward = 0.9 * 0.5 = 0.45
    assert abs(probs.forward - 0.45) < 0.001


# Test parse_pdg_response preserves edge with higher existence.
def test_parse_pdg_response_keeps_higher_existence() -> None:
    response_text = """
    {
        "edges": [
            {
                "source": "a", "target": "b",
                "existence": 0.3, "orientation": 0.5
            },
            {
                "source": "b", "target": "a",
                "existence": 0.9, "orientation": 0.7
            }
        ]
    }
    """
    pdg = parse_pdg_response(response_text, ["a", "b"])
    # Should keep the edge with higher existence (0.9)
    assert len(pdg.edges) == 1
    probs = pdg.edges[("a", "b")]
    assert abs(probs.p_exist - 0.9) < 0.001


# Test parse_pdg_response with empty edges list.
def test_parse_pdg_response_empty_edges() -> None:
    response_text = '{"edges": []}'
    pdg = parse_pdg_response(response_text, ["a", "b"])
    assert len(pdg.edges) == 0
    assert len(pdg.nodes) == 2


# Test parse_pdg_response with default probabilities.
def test_parse_pdg_response_missing_probabilities() -> None:
    response_text = """
    {
        "edges": [
            {"source": "a", "target": "b"}
        ]
    }
    """
    pdg = parse_pdg_response(response_text, ["a", "b"])
    # Should use default 0.5 for both existence and orientation
    probs = pdg.edges[("a", "b")]
    assert probs.p_exist == 0.5
    assert abs(probs.forward - 0.25) < 0.001  # 0.5 * 0.5
    assert abs(probs.backward - 0.25) < 0.001  # 0.5 * 0.5


# Test parse_pdg_response with plain markdown code block.
def test_parse_pdg_response_plain_markdown() -> None:
    response_text = """```
    {
        "edges": [
            {"source": "a", "target": "b", "existence": 0.8}
        ]
    }
    ```"""
    pdg = parse_pdg_response(response_text, ["a", "b"])
    assert len(pdg.edges) == 1
    probs = pdg.edges[("a", "b")]
    assert abs(probs.p_exist - 0.8) < 0.001
