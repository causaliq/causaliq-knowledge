"""Response models and parsing for LLM graph generation.

This module provides Pydantic models for representing LLM-generated
causal graphs and functions for parsing LLM responses in edge list,
adjacency matrix, and PDG (Probabilistic Dependency Graph) formats.

The implementation is split across three sibling modules, and every name
that was previously importable from here is re-exported unchanged:

- `response_models`: the data models.
- `response_parsing`: the parsing functions.
- `json_repair`: JSON extraction and repair helpers.
"""

from __future__ import annotations

from causaliq_knowledge.graph.json_repair import (  # noqa: F401
    _sanitise_json_text,
    _try_parse_json,
)
from causaliq_knowledge.graph.response_models import (
    GeneratedGraph,
    GenerationMetadata,
    PDGGenerationResult,
    ProposedEdge,
)
from causaliq_knowledge.graph.response_parsing import (
    parse_adjacency_matrix_response,
    parse_edge_list_response,
    parse_graph_response,
    parse_pdg_response,
)

__all__ = [
    "GeneratedGraph",
    "GenerationMetadata",
    "PDGGenerationResult",
    "ProposedEdge",
    "parse_adjacency_matrix_response",
    "parse_edge_list_response",
    "parse_graph_response",
    "parse_pdg_response",
]
