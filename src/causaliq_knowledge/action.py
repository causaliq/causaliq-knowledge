"""CausalIQ workflow action for graph generation.

This module provides the workflow action integration for causaliq-knowledge,
allowing graph generation to be used as a step in CausalIQ workflows.

The action is auto-discovered by causaliq-workflow when this package is
imported, using the convention of exporting a class named 'ActionProvider'.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

from causaliq_core import (
    ActionExecutionError,
    ActionInput,
    ActionResult,
    ActionValidationError,
    CausalIQActionProvider,
)
from causaliq_workflow.logger import WorkflowLogger
from causaliq_workflow.registry import WorkflowContext
from pydantic import ValidationError

from causaliq_knowledge.graph.params import GenerateGraphParams

if TYPE_CHECKING:  # pragma: no cover
    from causaliq_knowledge.graph.response import GeneratedGraph

logger = logging.getLogger(__name__)

# Re-export for convenience (unused imports are intentional for API surface)
__all__ = [
    "ActionExecutionError",
    "ActionInput",
    "ActionValidationError",
    "CausalIQActionProvider",
    "WorkflowContext",
    "WorkflowLogger",
    "KnowledgeActionProvider",
    "ActionProvider",
    "SUPPORTED_ACTIONS",
]


# Supported actions within this package
SUPPORTED_ACTIONS = {"generate_graph"}


def _create_action_inputs() -> Dict[str, Any]:
    """Create action input specifications.

    Returns:
        Dictionary of ActionInput specifications.
    """
    return {
        "action": ActionInput(
            name="action",
            description="Action to perform (e.g., 'generate_graph')",
            required=True,
            type_hint="str",
        ),
        "context": ActionInput(
            name="context",
            description="Path to network context JSON file",
            required=True,
            type_hint="str",
        ),
        "prompt_detail": ActionInput(
            name="prompt_detail",
            description="Detail level for prompts: minimal, standard, or rich",
            required=False,
            default="standard",
            type_hint="str",
        ),
        "use_benchmark_names": ActionInput(
            name="use_benchmark_names",
            description="Use benchmark names instead of LLM names",
            required=False,
            default=False,
            type_hint="bool",
        ),
        "llm_model": ActionInput(
            name="llm_model",
            description="LLM model identifier (e.g., groq/llama-3.1-8b)",
            required=False,
            default="groq/llama-3.1-8b-instant",
            type_hint="str",
        ),
        "output": ActionInput(
            name="output",
            description="Workflow Cache .db path or 'none' for no persistence",
            required=False,
            default="none",
            type_hint="str",
        ),
        "llm_cache": ActionInput(
            name="llm_cache",
            description="Path to LLM cache database (.db) or 'none'",
            required=False,
            default="none",
            type_hint="str",
        ),
        "llm_temperature": ActionInput(
            name="llm_temperature",
            description="LLM sampling temperature (0.0-2.0)",
            required=False,
            default=0.1,
            type_hint="float",
        ),
    }


class KnowledgeActionProvider(CausalIQActionProvider):
    """Workflow action provider for causaliq-knowledge integration.

    This action integrates causaliq-knowledge graph generation into
    CausalIQ workflows, allowing LLM-based graph generation to be used
    as workflow steps.

    The provider supports the 'generate_graph' action, which:
    - Loads a model specification from a JSON file
    - Queries an LLM to propose causal relationships
    - Returns the generated graph structure

    Attributes:
        name: Provider identifier for workflow 'uses' field.
        version: Provider version.
        description: Human-readable description.
        supported_actions: Set of actions this provider supports.
        inputs: Input parameter specifications.

    Example workflow step:
        ```yaml
        steps:
          - name: Generate causal graph
            uses: causaliq-knowledge
            with:
              action: generate_graph
              context: "{{data_dir}}/cancer.json"
              output: "{{data_dir}}/results.db"
              llm_cache: "{{data_dir}}/llm_cache.db"
              prompt_detail: standard
              llm_model: groq/llama-3.1-8b-instant
        ```
    """

    name: str = "causaliq-knowledge"
    version: str = "0.4.0"
    description: str = "Generate causal graphs using LLM knowledge"
    author: str = "CausalIQ"

    supported_actions: Set[str] = SUPPORTED_ACTIONS
    supported_types: Set[str] = {"graphml", "json"}

    inputs: Dict[str, Any] = _create_action_inputs()
    outputs: Dict[str, str] = {
        "graph": "Generated graph structure as JSON",
        "edge_count": "Number of edges in generated graph",
        "variable_count": "Number of variables in the model",
        "model_used": "LLM model used for generation",
        "cached": "Whether the result was retrieved from cache",
    }

    def __init__(self) -> None:
        """Initialise action with empty execution metadata."""
        super().__init__()

    def _validate_parameters(
        self, action: str, parameters: Dict[str, Any]
    ) -> None:
        """Validate action and parameters.

        Args:
            action: Action to perform (e.g., 'generate_graph').
            parameters: Dictionary of parameter values.

        Raises:
            ActionValidationError: If validation fails.
        """
        # Check action is supported
        if action not in SUPPORTED_ACTIONS:
            raise ActionValidationError(
                f"Unknown action: '{action}'. "
                f"Supported actions: {SUPPORTED_ACTIONS}"
            )

        # For generate_graph, validate using GenerateGraphParams
        if action == "generate_graph":
            # Check required context
            if "context" not in parameters:
                raise ActionValidationError(
                    "Missing required input: 'context' for generate_graph"
                )

            try:
                # Validate using Pydantic model
                GenerateGraphParams.from_dict(parameters)
            except (ValidationError, ValueError) as e:
                raise ActionValidationError(
                    f"Invalid parameters for generate_graph: {e}"
                )

    def run(
        self,
        action: str,
        parameters: Dict[str, Any],
        mode: str = "dry-run",
        context: Optional[Any] = None,
        logger: Optional[Any] = None,
    ) -> ActionResult:
        """Execute the action with validated parameters.

        Args:
            action: Action to perform (e.g., 'generate_graph').
            parameters: Dictionary of parameter values.
            mode: Execution mode ('dry-run', 'run', 'compare').
            context: Workflow context for optimisation.
            logger: Optional logger for task execution reporting.

        Returns:
            Tuple of (status, metadata, objects) where:
            - status: 'success' or 'skipped' (for dry-run)
            - metadata: Dict with edge_count, variable_count, model_used, etc.
            - objects: List of serialised objects (graphml, json)

        Raises:
            ActionExecutionError: If action execution fails.
        """
        # Validate parameters first
        self._validate_parameters(action, parameters)

        if action == "generate_graph":
            return self._run_generate_graph(parameters, mode, context, logger)
        else:  # pragma: no cover
            # This shouldn't happen after validate_parameters
            raise ActionExecutionError(f"Unknown action: {action}")

    def _run_generate_graph(
        self,
        parameters: Dict[str, Any],
        mode: str,
        context: Optional[Any],
        logger: Optional[Any],
    ) -> ActionResult:
        """Execute the generate_graph action.

        Args:
            parameters: Validated parameter values.
            mode: Execution mode.
            context: Workflow context.
            logger: Optional workflow logger.

        Returns:
            ActionResult tuple (status, metadata, objects).
        """
        try:
            params = GenerateGraphParams.from_dict(parameters)
        except (ValidationError, ValueError) as e:
            raise ActionExecutionError(f"Parameter validation failed: {e}")

        # Check context exists
        if not params.context.exists():
            raise ActionExecutionError(
                f"Network context not found: {params.context}"
            )

        # Dry-run mode: validate only, don't execute
        if mode == "dry-run":
            return self._dry_run_result(params)

        # Run mode: execute graph generation
        return self._execute_generate_graph(params, context)

    def _dry_run_result(self, params: GenerateGraphParams) -> ActionResult:
        """Return dry-run result without executing.

        Args:
            params: Validated parameters.

        Returns:
            ActionResult tuple for dry-run (skipped status, no objects).
        """
        metadata = {
            "message": "Dry-run mode: would generate graph",
            "context": str(params.context),
            "llm_model": params.llm_model,
            "llm_prompt_detail": params.prompt_detail.value,
            "output": params.output,
        }
        return ("skipped", metadata, [])

    def _build_execution_metadata(
        self,
        graph: "GeneratedGraph",
        params: GenerateGraphParams,
        stats: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build execution metadata from graph generation results.

        Extracts relevant metadata from the GeneratedGraph for inclusion
        in the ActionResult metadata dictionary.

        Args:
            graph: The generated graph with metadata.
            params: Generation parameters used.
            stats: Generator statistics.

        Returns:
            Dictionary of execution metadata.
        """
        # Start with generation parameters
        metadata: Dict[str, Any] = {
            "context": str(params.context),
            "llm_model": params.llm_model,
            "llm_prompt_detail": params.prompt_detail.value,
            "use_benchmark_names": params.use_benchmark_names,
            "edge_count": len(graph.edges),
            "variable_count": len(graph.variables),
            "cache_hits": stats.get("cache_hits", 0),
            "cache_misses": stats.get("cache_misses", 0),
        }

        # Add generation metadata if present
        if graph.metadata:
            meta = graph.metadata
            metadata.update(
                {
                    "llm_provider": meta.provider,
                    "timestamp": meta.timestamp.isoformat(),
                    "llm_timestamp": meta.llm_timestamp.isoformat(),
                    "llm_latency_ms": meta.llm_latency_ms,
                    "llm_input_tokens": meta.input_tokens,
                    "llm_output_tokens": meta.output_tokens,
                    "from_cache": meta.from_cache,
                    "llm_temperature": meta.temperature,
                    "llm_max_tokens": meta.max_tokens,
                    "llm_finish_reason": meta.finish_reason,
                    "llm_cost_usd": meta.llm_cost_usd,
                }
            )
            # Add messages (can be large, but important for reproducibility)
            if meta.messages:
                metadata["llm_messages"] = meta.messages

        return metadata

    def _execute_generate_graph(
        self,
        params: GenerateGraphParams,
        context: Optional[Any] = None,
    ) -> ActionResult:
        """Execute graph generation.

        Args:
            params: Validated parameters.
            context: Workflow context for cache key generation.

        Returns:
            ActionResult tuple (status, metadata, objects).
        """
        # Import here to avoid slow startup and circular imports
        from causaliq_core.cache import TokenCache

        from causaliq_knowledge.graph import NetworkContext
        from causaliq_knowledge.graph.generator import (
            GraphGenerator,
            GraphGeneratorConfig,
        )

        try:
            # Load network context
            network_ctx = NetworkContext.load(params.context)
            logger.info(
                f"Loaded network context: {network_ctx.network} "
                f"({len(network_ctx.variables)} variables)"
            )
        except Exception as e:
            raise ActionExecutionError(f"Failed to load network context: {e}")

        # Track mapping for name conversion
        llm_to_benchmark_mapping: Dict[str, str] = {}

        # Determine naming mode
        use_llm_names = not params.use_benchmark_names
        if use_llm_names and network_ctx.uses_distinct_llm_names():
            llm_to_benchmark_mapping = network_ctx.get_llm_to_name_mapping()

        # Set up LLM response cache (not workflow cache)
        llm_cache: Optional[TokenCache] = None
        cache_path = params.get_effective_cache_path()
        if cache_path is not None:
            try:
                llm_cache = TokenCache(str(cache_path))
                llm_cache.open()
            except Exception as e:
                raise ActionExecutionError(f"Failed to open LLM cache: {e}")

        try:
            # Import OutputFormat for generator config
            from causaliq_knowledge.graph.prompts import OutputFormat

            # Create generator - always use edge_list format
            # Derive request_id from output filename stem
            if params.output.lower() == "none":
                request_id = "none"
            else:
                request_id = Path(params.output).stem

            config = GraphGeneratorConfig(
                temperature=params.llm_temperature,
                output_format=OutputFormat.EDGE_LIST,
                prompt_detail=params.prompt_detail,
                use_llm_names=use_llm_names,
                request_id=request_id,
            )
            generator = GraphGenerator(
                model=params.llm_model, config=config, cache=llm_cache
            )

            # Generate graph
            graph = generator.generate_from_context(
                network_ctx, level=params.prompt_detail
            )

            # Map LLM names back to benchmark names
            if llm_to_benchmark_mapping:
                graph = self._map_graph_names(graph, llm_to_benchmark_mapping)

            # Get stats
            stats = generator.get_stats()

            # Build execution metadata
            metadata = self._build_execution_metadata(graph, params, stats)

            # Add cached flag for convenience
            metadata["cached"] = stats.get("cache_hits", 0) > 0
            metadata["model_used"] = params.llm_model

            # Serialise graph to open-standard formats
            graphml_content = self._serialise_graphml(graph)
            json_content = self._serialise_json(graph)

            # Build objects list
            objects: List[Dict[str, Any]] = [
                {
                    "type": "graphml",
                    "name": "graph",
                    "content": graphml_content,
                },
                {
                    "type": "json",
                    "name": "confidences",
                    "content": json_content,
                },
            ]

            return ("success", metadata, objects)

        except Exception as e:
            raise ActionExecutionError(f"Graph generation failed: {e}")
        finally:
            if llm_cache:
                llm_cache.close()

    def _map_graph_names(
        self, graph: "GeneratedGraph", mapping: Dict[str, str]
    ) -> "GeneratedGraph":
        """Map variable names in a graph using a mapping dictionary.

        Args:
            graph: The generated graph with edges to map.
            mapping: Dictionary mapping old names to new names.

        Returns:
            New GeneratedGraph with mapped variable names.
        """
        from causaliq_knowledge.graph.response import (
            GeneratedGraph,
            ProposedEdge,
        )

        new_edges = []
        for edge in graph.edges:
            new_edge = ProposedEdge(
                source=mapping.get(edge.source, edge.source),
                target=mapping.get(edge.target, edge.target),
                confidence=edge.confidence,
            )
            new_edges.append(new_edge)

        new_variables = [mapping.get(v, v) for v in graph.variables]

        return GeneratedGraph(
            edges=new_edges,
            variables=new_variables,
            reasoning=graph.reasoning,
            metadata=graph.metadata,
        )

    def serialise(
        self,
        data_type: str,
        data: Any,
    ) -> str:
        """Serialise data to GraphML or JSON format string.

        Converts a GeneratedGraph to the specified format.

        Args:
            data_type: Type of data. Supported values:
                - 'graphml': GraphML format
                - 'json': JSON format with full metadata
            data: GeneratedGraph instance, or tuple (graph, extra_blobs)
                as returned by GraphEntryEncoder.decode().

        Returns:
            String representation of the graph in the specified format.

        Raises:
            NotImplementedError: If the data type is not supported.
            ValueError: If data is not a GeneratedGraph.
        """
        from causaliq_knowledge.graph.response import GeneratedGraph

        # Validate against supported_types first
        if data_type not in self.supported_types:
            raise NotImplementedError(
                f"Provider '{self.name}' does not support serialising "
                f"data_type '{data_type}'. Supported: {self.supported_types}"
            )

        # Handle tuple from decode() which returns (graph, extra_blobs)
        if isinstance(data, tuple) and len(data) == 2:
            data = data[0]

        # Validate data is a GeneratedGraph
        if not isinstance(data, GeneratedGraph):
            raise ValueError(
                f"Expected GeneratedGraph, got {type(data).__name__}"
            )

        if data_type == "graphml":
            return self._serialise_graphml(data)
        else:  # json
            return self._serialise_json(data)

    def _serialise_graphml(self, graph: "GeneratedGraph") -> str:
        """Serialise graph to GraphML format."""
        from io import StringIO

        from causaliq_core.graph import SDG
        from causaliq_core.graph.io import graphml

        edges = [(e.source, "->", e.target) for e in graph.edges]
        sdg = SDG(list(graph.variables), edges)

        buffer = StringIO()
        graphml.write(sdg, buffer)
        buffer.seek(0)
        return buffer.getvalue()

    def _serialise_json(self, graph: "GeneratedGraph") -> str:
        """Serialise graph to JSON format with full metadata."""
        import json

        result: Dict[str, Any] = {
            "variables": list(graph.variables),
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "confidence": e.confidence,
                    "reasoning": e.reasoning,
                }
                for e in graph.edges
            ],
            "reasoning": graph.reasoning,
        }

        if graph.metadata is not None:
            meta = graph.metadata
            result["metadata"] = {
                "llm_model": meta.model,
                "llm_provider": meta.provider,
                "llm_timestamp": meta.llm_timestamp.isoformat(),
                "llm_latency_ms": meta.llm_latency_ms,
                "llm_input_tokens": meta.input_tokens,
                "llm_output_tokens": meta.output_tokens,
                "llm_from_cache": meta.from_cache,
                "llm_temperature": meta.temperature,
                "llm_max_tokens": meta.max_tokens,
                "llm_finish_reason": meta.finish_reason,
                "llm_cost_usd": meta.llm_cost_usd,
            }

        return json.dumps(result, indent=2)

    def deserialise(
        self,
        data_type: str,
        content: str,
    ) -> "GeneratedGraph":
        """Deserialise data from GraphML format string.

        Converts GraphML format to a GeneratedGraph.

        Args:
            data_type: Type of data (must be 'graphml').
            content: GraphML string representation of the data.

        Returns:
            The deserialised GeneratedGraph object.

        Raises:
            NotImplementedError: If the data type is not supported.
        """
        from io import StringIO

        from causaliq_core.graph.io import graphml

        from causaliq_knowledge.graph.response import (
            GeneratedGraph,
            ProposedEdge,
        )

        # Validate against supported_types
        if data_type not in self.supported_types:
            raise NotImplementedError(
                f"Provider '{self.name}' does not support deserialising "
                f"data_type '{data_type}'. Supported: {self.supported_types}"
            )

        # Only graphml can be deserialised to GeneratedGraph
        if data_type != "graphml":
            raise NotImplementedError(
                f"Provider '{self.name}' cannot deserialise '{data_type}' "
                f"to GeneratedGraph. Use 'graphml'."
            )

        # Read graph from StringIO
        sdg = graphml.read(StringIO(content))

        # Convert SDG to GeneratedGraph
        edges = [
            ProposedEdge(source=src, target=tgt, confidence=0.5)
            for (src, tgt) in sdg.edges.keys()
        ]
        graph = GeneratedGraph(
            edges=edges,
            variables=list(sdg.nodes),
            reasoning="Deserialised from GraphML",
        )

        return graph


# Export as ActionProvider for auto-discovery by causaliq-workflow
# This name is required by the auto-discovery convention
ActionProvider = KnowledgeActionProvider
