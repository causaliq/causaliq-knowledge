"""Graph generation module for causaliq-knowledge.

This module provides functionality for LLM-based causal graph generation
from variable specifications.
"""

from causaliq_knowledge.graph.disguiser import VariableDisguiser
from causaliq_knowledge.graph.generator import (
    GraphGenerator,
    GraphGeneratorConfig,
)
from causaliq_knowledge.graph.loader import ModelLoader, ModelLoadError
from causaliq_knowledge.graph.models import (
    CausalPrinciple,
    Constraints,
    GroundTruth,
    LLMGuidance,
    ModelSpec,
    Provenance,
    VariableRole,
    VariableSpec,
    VariableType,
    ViewDefinition,
    Views,
)
from causaliq_knowledge.graph.params import GenerateGraphParams
from causaliq_knowledge.graph.prompts import (
    ADJACENCY_MATRIX_RESPONSE_SCHEMA,
    EDGE_LIST_RESPONSE_SCHEMA,
    GraphQueryPrompt,
    OutputFormat,
)
from causaliq_knowledge.graph.response import (
    GeneratedGraph,
    GenerationMetadata,
    ProposedEdge,
    parse_adjacency_matrix_response,
    parse_edge_list_response,
    parse_graph_response,
)
from causaliq_knowledge.graph.view_filter import PromptDetail, ViewFilter

__all__ = [
    # Models
    "ModelSpec",
    "Provenance",
    "LLMGuidance",
    "ViewDefinition",
    "Views",
    "VariableSpec",
    "VariableRole",
    "VariableType",
    "Constraints",
    "CausalPrinciple",
    "GroundTruth",
    # Params (shared validation)
    "GenerateGraphParams",
    # Loader
    "ModelLoader",
    "ModelLoadError",
    # Filtering
    "ViewFilter",
    "PromptDetail",
    # Disguising
    "VariableDisguiser",
    # Prompts
    "GraphQueryPrompt",
    "OutputFormat",
    "EDGE_LIST_RESPONSE_SCHEMA",
    "ADJACENCY_MATRIX_RESPONSE_SCHEMA",
    # Response models
    "ProposedEdge",
    "GeneratedGraph",
    "GenerationMetadata",
    "parse_edge_list_response",
    "parse_adjacency_matrix_response",
    "parse_graph_response",
    # Generator
    "GraphGenerator",
    "GraphGeneratorConfig",
]
