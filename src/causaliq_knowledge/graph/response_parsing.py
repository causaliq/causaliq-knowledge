"""Parsing of LLM graph generation responses.

This module converts parsed JSON payloads into `GeneratedGraph` and `PDG`
objects for the edge list, adjacency matrix, and PDG response formats.
The models themselves live in `causaliq_knowledge.graph.response_models`.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from causaliq_core.graph.pdg import PDG, EdgeProbabilities

from causaliq_knowledge.graph.json_repair import _try_parse_json
from causaliq_knowledge.graph.response_models import (
    GeneratedGraph,
    ProposedEdge,
)

logger = logging.getLogger(__name__)


def _require_dict_response(response: Any) -> None:
    """Raise when a parsed response is not a dictionary."""
    if not isinstance(response, dict):
        raise ValueError(f"Expected dict response, got {type(response)}")


def _read_edges_payload(response: Dict[str, Any]) -> List[Any]:
    """Return the edges list of a response, validating its type."""
    edges_data = response.get("edges", [])
    if not isinstance(edges_data, list):
        raise ValueError(
            f"Expected 'edges' to be a list, got {type(edges_data)}"
        )
    return edges_data


def _read_reasoning(response: Dict[str, Any]) -> str:
    """Return the reasoning text of a response as a string."""
    reasoning = response.get("reasoning", "")
    if not isinstance(reasoning, str):
        reasoning = str(reasoning)
    return reasoning


def _read_edge_endpoints(edge_data: Dict[str, Any]) -> Tuple[str, str]:
    """Return the source and target names of an edge payload."""
    return edge_data.get("source", ""), edge_data.get("target", "")


def _is_valid_edge_pair(
    source: str,
    target: str,
    valid_vars: Set[str],
) -> bool:
    """Return True when an edge joins two distinct known variables."""
    if source not in valid_vars:
        logger.warning(f"Unknown source variable: {source}")
        return False
    if target not in valid_vars:
        logger.warning(f"Unknown target variable: {target}")
        return False
    if source == target:
        logger.warning(f"Skipping self-loop: {source} -> {target}")
        return False
    return True


def _edge_from_payload(
    edge_data: Any,
    index: int,
    valid_vars: Set[str],
) -> Optional[ProposedEdge]:
    """Build a ProposedEdge from one edge list payload entry."""
    if not isinstance(edge_data, dict):
        logger.warning(f"Skipping invalid edge at index {index}: not a dict")
        return None
    source, target = _read_edge_endpoints(edge_data)
    if not _is_valid_edge_pair(source, target, valid_vars):
        return None
    return ProposedEdge(
        source=source,
        target=target,
        confidence=edge_data.get("confidence", 0.5),
        reasoning=edge_data.get("reasoning"),
    )


def parse_edge_list_response(
    response: Dict[str, Any],
    variables: List[str],
) -> GeneratedGraph:
    """Parse an edge list format response from the LLM.

    Expected JSON format:
    {
        "edges": [
            {"source": "var1", "target": "var2", "confidence": 0.8},
            ...
        ],
        "reasoning": "explanation"
    }

    Args:
        response: Parsed JSON response from the LLM.
        variables: List of valid variable names for validation.

    Returns:
        GeneratedGraph with parsed edges.

    Raises:
        ValueError: If the response format is invalid.
    """
    _require_dict_response(response)
    edges_data = _read_edges_payload(response)
    reasoning = _read_reasoning(response)
    valid_vars = set(variables)
    edges = []

    for i, edge_data in enumerate(edges_data):
        edge = _edge_from_payload(edge_data, i, valid_vars)
        if edge is not None:
            edges.append(edge)

    return GeneratedGraph(
        edges=edges,
        variables=variables,
        reasoning=reasoning,
        raw_response=response,
    )


def _read_matrix_payload(
    response: Dict[str, Any],
) -> Tuple[List[Any], List[Any]]:
    """Return the variables and adjacency matrix of a response."""
    variables = response.get("variables", [])
    if not isinstance(variables, list):
        raise ValueError(
            f"Expected 'variables' to be a list, got {type(variables)}"
        )

    matrix = response.get("adjacency_matrix", [])
    if not isinstance(matrix, list):
        raise ValueError(
            f"Expected 'adjacency_matrix' to be a list, got {type(matrix)}"
        )

    return variables, matrix


def _validate_matrix_shape(
    variables: List[Any],
    matrix: List[Any],
) -> None:
    """Raise when the matrix is not square or mismatches the variables."""
    n = len(variables)
    if len(matrix) != n:
        raise ValueError(
            f"Matrix has {len(matrix)} rows but {n} variables declared"
        )

    for i, row in enumerate(matrix):
        if not isinstance(row, list) or len(row) != n:
            raise ValueError(f"Matrix row {i} has incorrect dimensions")


def _warn_unexpected_variables(
    variables: List[Any],
    expected_variables: List[str],
) -> None:
    """Log a warning for response variables the caller did not supply."""
    valid_vars = set(expected_variables)
    for var in variables:
        if var not in valid_vars:
            logger.warning(f"Unexpected variable in response: {var}")


def _edge_from_cell(
    value: Any,
    source: Any,
    target: Any,
    is_diagonal: bool,
) -> Optional[ProposedEdge]:
    """Build a ProposedEdge from one adjacency matrix cell."""
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None

    if confidence > 0.0 and not is_diagonal:  # Skip zeros and self-loops
        return ProposedEdge(
            source=source,
            target=target,
            confidence=max(0.0, min(1.0, confidence)),
        )

    return None


def _edges_from_row(
    variables: List[Any],
    row: List[Any],
    source: Any,
    row_index: int,
) -> List[ProposedEdge]:
    """Extract the non-zero edges of one adjacency matrix row."""
    edges = []
    for j, target in enumerate(variables):
        edge = _edge_from_cell(row[j], source, target, row_index == j)
        if edge is not None:
            edges.append(edge)
    return edges


def _edges_from_matrix(
    variables: List[Any],
    matrix: List[Any],
) -> List[ProposedEdge]:
    """Extract the edges declared by an adjacency matrix."""
    edges: List[ProposedEdge] = []
    for i, source in enumerate(variables):
        edges.extend(_edges_from_row(variables, matrix[i], source, i))
    return edges


def parse_adjacency_matrix_response(
    response: Dict[str, Any],
    expected_variables: List[str],
) -> GeneratedGraph:
    """Parse an adjacency matrix format response from the LLM.

    Expected JSON format:
    {
        "variables": ["var1", "var2", "var3"],
        "adjacency_matrix": [
            [0.0, 0.8, 0.0],
            [0.0, 0.0, 0.6],
            [0.0, 0.0, 0.0]
        ],
        "reasoning": "explanation"
    }

    Args:
        response: Parsed JSON response from the LLM.
        expected_variables: List of expected variable names for validation.

    Returns:
        GeneratedGraph with edges extracted from the matrix.

    Raises:
        ValueError: If the response format is invalid.
    """
    _require_dict_response(response)
    variables, matrix = _read_matrix_payload(response)
    reasoning = _read_reasoning(response)

    # Validate matrix dimensions
    _validate_matrix_shape(variables, matrix)

    # Validate variables against expected
    _warn_unexpected_variables(variables, expected_variables)

    # Extract edges from matrix
    edges = _edges_from_matrix(variables, matrix)

    return GeneratedGraph(
        edges=edges,
        variables=variables,
        reasoning=reasoning,
        raw_response=response,
    )


def _strip_markdown_fence(text: str) -> str:
    """Strip a surrounding markdown code fence from response text."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _parse_json_document(text: str) -> Any:
    """Parse response text as JSON, raising ValueError on failure."""
    response, error = _try_parse_json(_strip_markdown_fence(text))
    if error:
        logger.error(error)
        raise ValueError(f"Failed to parse JSON response: {error}")
    return response


def parse_graph_response(
    response_text: str,
    variables: List[str],
    output_format: str = "edge_list",
) -> GeneratedGraph:
    """Parse an LLM response into a GeneratedGraph.

    Handles JSON extraction from markdown code blocks and parses
    according to the specified output format.

    Args:
        response_text: Raw text response from the LLM.
        variables: List of valid variable names.
        output_format: Expected format ("edge_list" or "adjacency_matrix").

    Returns:
        GeneratedGraph with parsed edges and metadata.

    Raises:
        ValueError: If JSON parsing fails or format is invalid.
    """
    response = _parse_json_document(response_text)

    if output_format == "adjacency_matrix":
        return parse_adjacency_matrix_response(response, variables)

    return parse_edge_list_response(response, variables)


def _read_edge_probabilities(
    edge_data: Dict[str, Any],
    index: int,
) -> EdgeProbabilities:
    """Convert existence and orientation scores to EdgeProbabilities."""
    # Get existence and orientation probabilities
    existence = edge_data.get("existence", 0.5)
    orientation = edge_data.get("orientation", 0.5)

    # Clamp to valid range
    try:
        existence = max(0.0, min(1.0, float(existence)))
        orientation = max(0.0, min(1.0, float(orientation)))
    except (TypeError, ValueError):
        logger.warning(f"Invalid probability values for edge {index}")
        existence = 0.5
        orientation = 0.5

    # Convert to EdgeProbabilities format
    # p_forward = P(source -> target) = P(exists) * P(orientation)
    # p_backward = P(target -> source) = P(exists) * (1 - P(orientation))
    p_forward = existence * orientation
    p_backward = existence * (1.0 - orientation)
    p_undirected = 0.0  # Causal LLM perspective: no undirected
    p_none = 1.0 - existence

    return EdgeProbabilities(
        forward=p_forward,
        backward=p_backward,
        undirected=p_undirected,
        none=p_none,
    )


def _canonicalise_edge(
    source: str,
    target: str,
    probs: EdgeProbabilities,
) -> Tuple[Tuple[str, str], EdgeProbabilities]:
    """Return an edge key and probabilities in canonical (alphabetic) order."""
    if source < target:
        return (source, target), probs

    # Swap forward/backward for canonical order
    return (target, source), EdgeProbabilities(
        forward=probs.backward,
        backward=probs.forward,
        undirected=probs.undirected,
        none=probs.none,
    )


def _pdg_edge_from_payload(
    edge_data: Any,
    index: int,
    valid_vars: Set[str],
) -> Optional[Tuple[Tuple[str, str], EdgeProbabilities]]:
    """Build a canonical PDG edge from one edge payload entry."""
    if not isinstance(edge_data, dict):
        logger.warning(f"Skipping invalid edge at index {index}: not a dict")
        return None

    source, target = _read_edge_endpoints(edge_data)

    # Validate variable names
    if not _is_valid_edge_pair(source, target, valid_vars):
        return None

    probs = _read_edge_probabilities(edge_data, index)
    return _canonicalise_edge(source, target, probs)


def _merge_pdg_edge(
    pdg_edges: Dict[Tuple[str, str], EdgeProbabilities],
    edge: Tuple[Tuple[str, str], EdgeProbabilities],
) -> None:
    """Store an edge, keeping the highest existence probability seen."""
    # If we already have an edge for this pair, keep the one with
    # higher existence probability (LLM may have proposed both
    # directions)
    key, probs = edge
    existing = pdg_edges.get(key)
    if existing is not None and existing.p_exist >= probs.p_exist:
        return
    pdg_edges[key] = probs


def parse_pdg_response(
    response_text: str,
    variables: List[str],
) -> PDG:
    """Parse an LLM response into a PDG (Probabilistic Dependency Graph).

    Expected JSON format:
    {
        "edges": [
            {
                "source": "var1",
                "target": "var2",
                "existence": 0.9,
                "orientation": 0.8
            },
            ...
        ],
        "reasoning": "explanation"
    }

    The existence and orientation probabilities are converted to
    EdgeProbabilities format:
    - p_forward = existence * orientation
    - p_backward = existence * (1 - orientation)
    - p_undirected = 0.0 (from causal perspective)
    - p_none = 1 - existence

    Args:
        response_text: Raw text response from the LLM.
        variables: List of valid variable names.

    Returns:
        PDG with edge probabilities.

    Raises:
        ValueError: If JSON parsing fails or format is invalid.
    """
    response = _parse_json_document(response_text)
    _require_dict_response(response)
    edges_data = _read_edges_payload(response)

    valid_vars = set(variables)
    pdg_edges: Dict[tuple[str, str], EdgeProbabilities] = {}

    for i, edge_data in enumerate(edges_data):
        edge = _pdg_edge_from_payload(edge_data, i, valid_vars)
        if edge is not None:
            _merge_pdg_edge(pdg_edges, edge)

    return PDG(list(variables), pdg_edges)
