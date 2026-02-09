"""CausalIQ workflow action for graph generation.

This module provides the workflow action integration for causaliq-knowledge,
allowing graph generation to be used as a step in CausalIQ workflows.

The action is auto-discovered by causaliq-workflow when this package is
imported, using the convention of exporting a class named 'CausalIQAction'.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from causaliq_workflow.action import (
    ActionExecutionError,
    ActionInput,
    ActionValidationError,
)
from causaliq_workflow.action import CausalIQAction as BaseCausalIQAction
from causaliq_workflow.logger import WorkflowLogger
from causaliq_workflow.registry import WorkflowContext
from pydantic import ValidationError

from causaliq_knowledge.graph.params import GenerateGraphParams

if TYPE_CHECKING:  # pragma: no cover
    from causaliq_knowledge.graph.models import ModelSpec
    from causaliq_knowledge.graph.response import GeneratedGraph

logger = logging.getLogger(__name__)

# Re-export for convenience (unused imports are intentional for API surface)
__all__ = [
    "ActionExecutionError",
    "ActionInput",
    "ActionValidationError",
    "BaseCausalIQAction",
    "WorkflowContext",
    "WorkflowLogger",
    "GenerateGraphAction",
    "CausalIQAction",
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
        "model_spec": ActionInput(
            name="model_spec",
            description="Path to model specification JSON file",
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
            required=True,
            type_hint="str",
        ),
        "llm_cache": ActionInput(
            name="llm_cache",
            description="Path to LLM cache database (.db) or 'none'",
            required=True,
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


class GenerateGraphAction(BaseCausalIQAction):
    """Workflow action for generating causal graphs from model specifications.

    This action integrates causaliq-knowledge graph generation into
    CausalIQ workflows, allowing LLM-based graph generation to be used
    as workflow steps.

    The action supports the 'generate_graph' operation, which:
    - Loads a model specification from a JSON file
    - Queries an LLM to propose causal relationships
    - Returns the generated graph structure

    Attributes:
        name: Action identifier for workflow 'uses' field.
        version: Action version.
        description: Human-readable description.
        inputs: Input parameter specifications.

    Example workflow step:
        ```yaml
        steps:
          - name: Generate causal graph
            uses: causaliq-knowledge
            with:
              action: generate_graph
              model_spec: "{{data_dir}}/cancer.json"
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

    def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        """Validate input values against specifications.

        Args:
            inputs: Dictionary of input values to validate.

        Returns:
            True if all inputs are valid.

        Raises:
            ActionValidationError: If validation fails.
        """
        # Check required 'action' parameter
        if "action" not in inputs:
            raise ActionValidationError(
                "Missing required input: 'action'. "
                f"Supported actions: {SUPPORTED_ACTIONS}"
            )

        action = inputs["action"]
        if action not in SUPPORTED_ACTIONS:
            raise ActionValidationError(
                f"Unknown action: '{action}'. "
                f"Supported actions: {SUPPORTED_ACTIONS}"
            )

        # For generate_graph, validate using GenerateGraphParams
        if action == "generate_graph":
            # Check required model_spec
            if "model_spec" not in inputs:
                raise ActionValidationError(
                    "Missing required input: 'model_spec' for generate_graph"
                )

            # Build params dict (excluding 'action' which isn't a param)
            params_dict = {k: v for k, v in inputs.items() if k != "action"}

            try:
                # Validate using Pydantic model
                GenerateGraphParams.from_dict(params_dict)
            except (ValidationError, ValueError) as e:
                raise ActionValidationError(
                    f"Invalid parameters for generate_graph: {e}"
                )

        return True

    def run(
        self,
        inputs: Dict[str, Any],
        mode: str = "dry-run",
        context: Optional[Any] = None,
        logger: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Execute the action with validated inputs.

        Args:
            inputs: Dictionary of input values keyed by input name.
            mode: Execution mode ('dry-run', 'run', 'compare').
            context: Workflow context for optimisation.
            logger: Optional logger for task execution reporting.

        Returns:
            Dictionary containing:
            - status: 'success' or 'skipped' (for dry-run)
            - graph: Generated graph as JSON (if run mode)
            - edge_count: Number of edges
            - variable_count: Number of variables
            - model_used: LLM model identifier
            - cached: Whether result was from cache

        Raises:
            ActionExecutionError: If action execution fails.
        """
        # Validate inputs first
        self.validate_inputs(inputs)

        action = inputs["action"]

        if action == "generate_graph":
            return self._run_generate_graph(inputs, mode, context, logger)
        else:  # pragma: no cover
            # This shouldn't happen after validate_inputs
            raise ActionExecutionError(f"Unknown action: {action}")

    def _run_generate_graph(
        self,
        inputs: Dict[str, Any],
        mode: str,
        context: Optional[Any],
        logger: Optional[Any],
    ) -> Dict[str, Any]:
        """Execute the generate_graph action.

        Args:
            inputs: Validated input parameters.
            mode: Execution mode.
            context: Workflow context.
            logger: Optional workflow logger.

        Returns:
            Action result dictionary.
        """
        # Build params (excluding 'action')
        params_dict = {k: v for k, v in inputs.items() if k != "action"}

        try:
            params = GenerateGraphParams.from_dict(params_dict)
        except (ValidationError, ValueError) as e:
            raise ActionExecutionError(f"Parameter validation failed: {e}")

        # Check model_spec exists
        if not params.model_spec.exists():
            raise ActionExecutionError(
                f"Model specification not found: {params.model_spec}"
            )

        # Dry-run mode: validate only, don't execute
        if mode == "dry-run":
            return self._dry_run_result(params)

        # Run mode: execute graph generation
        return self._execute_generate_graph(params, context)

    def _dry_run_result(self, params: GenerateGraphParams) -> Dict[str, Any]:
        """Return dry-run result without executing.

        Args:
            params: Validated parameters.

        Returns:
            Dry-run result dictionary.
        """
        return {
            "status": "skipped",
            "message": "Dry-run mode: would generate graph",
            "model_spec": str(params.model_spec),
            "llm_model": params.llm_model,
            "prompt_detail": params.prompt_detail.value,
            "output": params.output,
        }

    def _populate_execution_metadata(
        self,
        graph: "GeneratedGraph",
        params: GenerateGraphParams,
        stats: Dict[str, Any],
    ) -> None:
        """Populate execution metadata from graph generation results.

        Extracts relevant metadata from the GeneratedGraph and stores it
        in _execution_metadata for later retrieval via get_action_metadata().

        Args:
            graph: The generated graph with metadata.
            params: Generation parameters used.
            stats: Generator statistics.
        """
        # Start with generation parameters
        self._execution_metadata = {
            "model_spec": str(params.model_spec),
            "llm_model": params.llm_model,
            "prompt_detail": params.prompt_detail.value,
            "use_benchmark_names": params.use_benchmark_names,
            "edge_count": len(graph.edges),
            "variable_count": len(graph.variables),
            "cache_hits": stats.get("cache_hits", 0),
            "cache_misses": stats.get("cache_misses", 0),
        }

        # Add generation metadata if present
        if graph.metadata:
            meta = graph.metadata
            self._execution_metadata.update(
                {
                    "provider": meta.provider,
                    "timestamp": meta.timestamp.isoformat(),
                    "latency_ms": meta.latency_ms,
                    "input_tokens": meta.input_tokens,
                    "output_tokens": meta.output_tokens,
                    "cost_usd": meta.cost_usd,
                    "from_cache": meta.from_cache,
                    "temperature": meta.temperature,
                    "max_tokens": meta.max_tokens,
                    "finish_reason": meta.finish_reason,
                    "initial_cost_usd": meta.initial_cost_usd,
                }
            )
            # Add timestamps if present
            if meta.request_timestamp:
                self._execution_metadata["request_timestamp"] = (
                    meta.request_timestamp.isoformat()
                )
            if meta.completion_timestamp:
                self._execution_metadata["completion_timestamp"] = (
                    meta.completion_timestamp.isoformat()
                )
            # Add messages (can be large, but important for reproducibility)
            if meta.messages:
                self._execution_metadata["messages"] = meta.messages

    def _execute_generate_graph(
        self,
        params: GenerateGraphParams,
        context: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Execute graph generation.

        Args:
            params: Validated parameters.
            context: Workflow context for cache key generation.

        Returns:
            Result dictionary with generated graph.
        """
        # Import here to avoid slow startup and circular imports
        from causaliq_core.cache import TokenCache
        from causaliq_workflow.cache import WorkflowCache

        from causaliq_knowledge.graph import GraphEntryEncoder, ModelLoader
        from causaliq_knowledge.graph.generator import (
            GraphGenerator,
            GraphGeneratorConfig,
        )

        try:
            # Load model specification
            spec = ModelLoader.load(params.model_spec)
            logger.info(
                f"Loaded model specification: {spec.dataset_id} "
                f"({len(spec.variables)} variables)"
            )
        except Exception as e:
            raise ActionExecutionError(
                f"Failed to load model specification: {e}"
            )

        # Track mapping for name conversion
        llm_to_benchmark_mapping: Dict[str, str] = {}

        # Determine naming mode
        use_llm_names = not params.use_benchmark_names
        if use_llm_names and spec.uses_distinct_llm_names():
            llm_to_benchmark_mapping = spec.get_llm_to_name_mapping()

        # Set up cache
        cache: Optional[TokenCache] = None
        cache_path = params.get_effective_cache_path()
        if cache_path is not None:
            try:
                cache = TokenCache(str(cache_path))
                cache.open()
            except Exception as e:
                raise ActionExecutionError(f"Failed to open cache: {e}")

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
                model=params.llm_model, config=config, cache=cache
            )

            # Generate graph
            graph = generator.generate_from_spec(
                spec, level=params.prompt_detail
            )

            # Map LLM names back to benchmark names
            if llm_to_benchmark_mapping:
                graph = self._map_graph_names(graph, llm_to_benchmark_mapping)

            # Get stats
            stats = generator.get_stats()

            # Populate execution metadata for get_action_metadata()
            self._populate_execution_metadata(graph, params, stats)

            # Build result
            result = {
                "status": "success",
                "graph": self._graph_to_dict(graph),
                "edge_count": len(graph.edges),
                "variable_count": len(graph.variables),
                "model_used": params.llm_model,
                "cached": stats.get("cache_hits", 0) > 0,
                "outputs": {
                    "graph": self._graph_to_dict(graph),
                    "edge_count": len(graph.edges),
                    "variable_count": len(graph.variables),
                    "model_used": params.llm_model,
                    "cached": stats.get("cache_hits", 0) > 0,
                },
            }

            # Write output if specified
            output_path = params.get_effective_output_path()
            if output_path:
                if params.is_workflow_cache_output():
                    # Write to Workflow Cache (.db file)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    self._write_to_workflow_cache(
                        output_path,
                        graph,
                        context,
                        WorkflowCache,
                        GraphEntryEncoder,
                    )
                    result["output_cache"] = str(output_path)
                else:
                    # Directory output - check for matrix context conflict
                    # Only block if context has non-empty matrix
                    has_matrix = (
                        context is not None
                        and hasattr(context, "matrix")
                        and context.matrix
                    )
                    if has_matrix:
                        raise ActionExecutionError(
                            "Workflow with matrix variables requires .db "
                            "output (Workflow Cache), not directory output. "
                            "Each matrix combination would overwrite files."
                        )
                    # Write GraphML + JSON files to directory
                    self._write_to_directory(output_path, graph, spec)
                    result["output_dir"] = str(output_path)

            return result

        except Exception as e:
            raise ActionExecutionError(f"Graph generation failed: {e}")
        finally:
            if cache:
                cache.close()

    def _write_to_workflow_cache(
        self,
        output_path: Path,
        graph: "GeneratedGraph",
        context: Optional[Any],
        workflow_cache_cls: type,
        encoder_cls: type,
    ) -> None:
        """Write generated graph to Workflow Cache.

        Args:
            output_path: Path to Workflow Cache .db file.
            graph: Generated graph to store.
            context: Workflow context for cache key generation.
            workflow_cache_cls: WorkflowCache class (passed to avoid import).
            encoder_cls: GraphEntryEncoder class (passed to avoid import).
        """
        # Build cache key from context matrix values (if available)
        if context is not None and hasattr(context, "matrix"):
            key_data = dict(context.matrix)
        else:
            # Fallback: use model spec filename as key
            key_data = {"source": "generate_graph"}

        with workflow_cache_cls(str(output_path)) as wf_cache:
            encoder = encoder_cls()
            wf_cache.register_encoder("graph", encoder)
            wf_cache.put(key_data, "graph", graph)

    def _graph_to_dict(self, graph: "GeneratedGraph") -> Dict[str, Any]:
        """Convert GeneratedGraph to dictionary.

        Args:
            graph: Generated graph object.

        Returns:
            Dictionary representation of the graph.
        """
        return {
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "confidence": edge.confidence,
                }
                for edge in graph.edges
            ],
            "variables": graph.variables,
            "reasoning": graph.reasoning,
        }

    def _write_to_directory(
        self,
        output_dir: Path,
        graph: "GeneratedGraph",
        spec: "ModelSpec",
    ) -> None:
        """Write graph output to directory as GraphML + JSON files.

        Creates three files in the output directory:
        - graph.graphml: Graph structure in GraphML format
        - metadata.json: Variables, reasoning, generation info
        - confidences.json: Edge confidence scores

        Args:
            output_dir: Directory path to write files to.
            graph: Generated graph to write.
            spec: Model specification (for additional metadata).
        """
        import json

        from causaliq_core.graph.io import graphml
        from causaliq_core.graph.sdg import SDG

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build SDG from edges - SDG requires 3-tuple (source, "->", target)
        edges = [(edge.source, "->", edge.target) for edge in graph.edges]
        sdg = SDG(list(graph.variables), edges)

        # Write GraphML
        graphml_path = output_dir / "graph.graphml"
        graphml.write(sdg, str(graphml_path))

        # Build and write metadata
        metadata: Dict[str, Any] = {
            "dataset_id": spec.dataset_id,
            "domain": spec.domain,
            "variables": graph.variables,
            "edge_count": len(graph.edges),
            "reasoning": graph.reasoning,
        }

        # Add edge-level reasoning if present
        edge_reasoning = {}
        for edge in graph.edges:
            if edge.reasoning:
                key = f"{edge.source}->{edge.target}"
                edge_reasoning[key] = edge.reasoning
        if edge_reasoning:
            metadata["edge_reasoning"] = edge_reasoning

        # Add generation metadata if present
        if graph.metadata:
            metadata["generation"] = {
                "model": graph.metadata.model,
                "provider": graph.metadata.provider,
                "timestamp": graph.metadata.timestamp.isoformat(),
                "latency_ms": graph.metadata.latency_ms,
                "input_tokens": graph.metadata.input_tokens,
                "output_tokens": graph.metadata.output_tokens,
            }

        metadata_path = output_dir / "metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

        # Build and write confidences
        confidences = {
            f"{edge.source}->{edge.target}": edge.confidence
            for edge in graph.edges
        }

        confidences_path = output_dir / "confidences.json"
        confidences_path.write_text(
            json.dumps(confidences, indent=2), encoding="utf-8"
        )

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


# Export as CausalIQAction for auto-discovery by causaliq-workflow
# This name is required by the auto-discovery convention
CausalIQAction = GenerateGraphAction
