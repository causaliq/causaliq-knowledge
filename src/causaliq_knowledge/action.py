"""CausalIQ workflow action for graph generation.

This module provides the workflow action integration for causaliq-knowledge,
allowing graph generation to be used as a step in CausalIQ workflows.

The action is auto-discovered by causaliq-workflow when this package is
imported, using the convention of exporting a class named 'CausalIQAction'.
"""

from __future__ import annotations

import json
import logging
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
        "disguise": ActionInput(
            name="disguise",
            description="Enable variable name disguising",
            required=False,
            default=False,
            type_hint="bool",
        ),
        "use_benchmark_names": ActionInput(
            name="use_benchmark_names",
            description="Use benchmark names instead of LLM names",
            required=False,
            default=False,
            type_hint="bool",
        ),
        "seed": ActionInput(
            name="seed",
            description="Random seed for reproducible disguising",
            required=False,
            default=None,
            type_hint="Optional[int]",
        ),
        "llm": ActionInput(
            name="llm",
            description="LLM model identifier (e.g., groq/llama-3.1-8b)",
            required=False,
            default="groq/llama-3.1-8b-instant",
            type_hint="str",
        ),
        "output": ActionInput(
            name="output",
            description="Output file path for results",
            required=False,
            default=None,
            type_hint="Optional[str]",
        ),
        "output_format": ActionInput(
            name="output_format",
            description="Output format: edge_list or adjacency_matrix",
            required=False,
            default="edge_list",
            type_hint="str",
        ),
        "cache": ActionInput(
            name="cache",
            description="Enable LLM response caching",
            required=False,
            default=True,
            type_hint="bool",
        ),
        "cache_path": ActionInput(
            name="cache_path",
            description="Path to cache database file",
            required=False,
            default=None,
            type_hint="Optional[str]",
        ),
        "temperature": ActionInput(
            name="temperature",
            description="LLM sampling temperature (0.0-2.0)",
            required=False,
            default=0.1,
            type_hint="float",
        ),
        "request_id": ActionInput(
            name="request_id",
            description="Identifier for requests (stored in metadata)",
            required=False,
            default="workflow",
            type_hint="str",
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
              prompt_detail: standard
              llm: groq/llama-3.1-8b-instant
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

        # Set default request_id if not provided
        if "request_id" not in params_dict or not params_dict["request_id"]:
            params_dict["request_id"] = "workflow"

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
        return self._execute_generate_graph(params)

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
            "llm": params.llm,
            "prompt_detail": params.prompt_detail.value,
            "output_format": params.output_format.value,
        }

    def _execute_generate_graph(
        self, params: GenerateGraphParams
    ) -> Dict[str, Any]:
        """Execute graph generation.

        Args:
            params: Validated parameters.

        Returns:
            Result dictionary with generated graph.
        """
        # Import here to avoid slow startup and circular imports
        from causaliq_knowledge.cache import TokenCache
        from causaliq_knowledge.graph import ModelLoader
        from causaliq_knowledge.graph.disguiser import VariableDisguiser
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

        # Apply disguising if requested
        disguiser: Optional[VariableDisguiser] = None
        if params.disguise:
            disguiser = VariableDisguiser(spec, seed=params.seed)
            spec = disguiser.disguise_spec()

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
            # Create generator
            config = GraphGeneratorConfig(
                temperature=params.temperature,
                output_format=params.output_format,
                prompt_detail=params.prompt_detail,
                use_llm_names=use_llm_names,
                request_id=params.request_id,
            )
            generator = GraphGenerator(
                model=params.llm, config=config, cache=cache
            )

            # Generate graph
            graph = generator.generate_from_spec(
                spec, level=params.prompt_detail
            )

            # Reverse disguising if applied
            if disguiser:
                graph = disguiser.undisguise_graph(graph)

            # Map LLM names back to benchmark names
            if llm_to_benchmark_mapping:
                graph = self._map_graph_names(graph, llm_to_benchmark_mapping)

            # Get stats
            stats = generator.get_stats()

            # Build result
            result = {
                "status": "success",
                "graph": self._graph_to_dict(graph),
                "edge_count": len(graph.edges),
                "variable_count": len(graph.variables),
                "model_used": params.llm,
                "cached": stats.get("cache_hits", 0) > 0,
                "outputs": {
                    "graph": self._graph_to_dict(graph),
                    "edge_count": len(graph.edges),
                    "variable_count": len(graph.variables),
                    "model_used": params.llm,
                    "cached": stats.get("cache_hits", 0) > 0,
                },
            }

            # Write output file if specified
            if params.output:
                params.output.parent.mkdir(parents=True, exist_ok=True)
                params.output.write_text(
                    json.dumps(result["graph"], indent=2),
                    encoding="utf-8",
                )
                result["output_file"] = str(params.output)

            return result

        except Exception as e:
            raise ActionExecutionError(f"Graph generation failed: {e}")
        finally:
            if cache:
                cache.close()

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
