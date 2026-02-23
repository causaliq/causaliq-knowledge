"""CausalIQ workflow action for graph generation.

This module provides the workflow action integration for causaliq-knowledge,
allowing graph generation to be used as a step in CausalIQ workflows.

The action is auto-discovered by causaliq-workflow when this package is
imported, using the convention of exporting a class named 'ActionProvider'.
"""

from __future__ import annotations

import logging
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

from causaliq_core import (
    ActionExecutionError,
    ActionInput,
    ActionResult,
    ActionValidationError,
    CausalIQActionProvider,
)
from causaliq_core.graph.io import graphml
from causaliq_core.graph.pdg import PDG
from causaliq_workflow.logger import WorkflowLogger
from causaliq_workflow.registry import WorkflowContext
from pydantic import ValidationError

from causaliq_knowledge.graph.params import GenerateGraphParams

if TYPE_CHECKING:  # pragma: no cover
    pass

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
        "network_context": ActionInput(
            name="network_context",
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
              network_context: "{{data_dir}}/cancer.json"
              output: "{{data_dir}}/results.db"
              llm_cache: "{{data_dir}}/llm_cache.db"
              prompt_detail: standard
              llm_model: groq/llama-3.1-8b-instant
        ```
    """

    name: str = "causaliq-knowledge"
    version: str = "0.5.0"
    description: str = "Generate causal graphs using LLM knowledge"
    author: str = "CausalIQ"

    supported_actions: Set[str] = SUPPORTED_ACTIONS
    supported_types: Set[str] = set()  # PDG compression handled by core

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
            # Check required network_context
            if "network_context" not in parameters:
                raise ActionValidationError(
                    "Missing required input: 'network_context' for "
                    "generate_graph"
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

        # Check network_context exists
        if not params.network_context.exists():
            raise ActionExecutionError(
                f"Network context not found: {params.network_context}"
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
            "network_context": str(params.network_context),
            "llm_model": params.llm_model,
            "llm_prompt_detail": params.prompt_detail.value,
            "output": params.output,
        }
        return ("skipped", metadata, [])

    def _build_execution_metadata(
        self,
        pdg: PDG,
        params: GenerateGraphParams,
        generation_metadata: Any,
    ) -> Dict[str, Any]:
        """Build execution metadata from PDG generation results.

        Combines PDG statistics with comprehensive LLM generation metadata
        for full provenance tracking.

        Args:
            pdg: The generated PDG with edge probabilities.
            params: Generation parameters used.
            generation_metadata: GenerationMetadata from the generator.

        Returns:
            Dictionary of execution metadata.
        """
        # Count edges with non-zero existence probability
        edge_count = sum(
            1
            for probs in pdg.edges.values()
            if probs.p_exist > 0.01  # Threshold for "exists"
        )

        # Get comprehensive LLM metadata
        llm_metadata = generation_metadata.to_dict()

        # Build metadata combining parameters and LLM generation info
        metadata: Dict[str, Any] = {
            "network_context": str(params.network_context),
            "use_benchmark_names": params.use_benchmark_names,
            "edge_count": edge_count,
            "variable_count": len(pdg.nodes),
            # Include all LLM generation metadata
            **llm_metadata,
        }

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
            network_ctx = NetworkContext.load(params.network_context)
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
            # Derive request_id from output filename stem
            if params.output.lower() == "none":
                request_id = "none"
            else:
                request_id = Path(params.output).stem

            config = GraphGeneratorConfig(
                temperature=params.llm_temperature,
                prompt_detail=params.prompt_detail,
                use_llm_names=use_llm_names,
                request_id=request_id,
            )
            generator = GraphGenerator(
                model=params.llm_model, config=config, cache=llm_cache
            )

            # Generate PDG with edge probabilities
            result = generator.generate_pdg_from_context(
                network_ctx, level=params.prompt_detail
            )
            pdg = result.pdg
            generation_metadata = result.metadata

            # Map LLM names back to benchmark names if needed
            if llm_to_benchmark_mapping:
                pdg = self._map_pdg_names(pdg, llm_to_benchmark_mapping)

            # Build execution metadata with comprehensive LLM info
            metadata = self._build_execution_metadata(
                pdg, params, generation_metadata
            )

            # Add cached flag for convenience
            metadata["cached"] = generation_metadata.from_cache
            metadata["model_used"] = params.llm_model

            # Convert PDG to GraphML string for interchange
            graphml_buffer = StringIO()
            graphml.write_pdg(pdg, graphml_buffer)
            graphml_content = graphml_buffer.getvalue()

            # Return GraphML string for workflow cache compression
            objects: List[Dict[str, Any]] = [
                {
                    "type": "pdg",
                    "name": "graph",
                    "content": graphml_content,
                },
            ]

            return ("success", metadata, objects)

        except Exception as e:
            raise ActionExecutionError(f"Graph generation failed: {e}")
        finally:
            if llm_cache:
                llm_cache.close()

    def _map_pdg_names(self, pdg: PDG, mapping: Dict[str, str]) -> PDG:
        """Map variable names in a PDG using a mapping dictionary.

        Args:
            pdg: The generated PDG with edges to map.
            mapping: Dictionary mapping old names to new names.

        Returns:
            New PDG with mapped variable names.
        """
        from causaliq_core.graph.pdg import EdgeProbabilities

        # Map node names
        new_nodes = [mapping.get(n, n) for n in pdg.nodes]

        # Map edge keys and maintain probability values
        new_edges: Dict[tuple[str, str], EdgeProbabilities] = {}
        for (src, tgt), probs in pdg.edges.items():
            new_src = mapping.get(src, src)
            new_tgt = mapping.get(tgt, tgt)

            # Maintain canonical order (alphabetical)
            if new_src < new_tgt:
                key = (new_src, new_tgt)
                new_probs = probs
            else:
                key = (new_tgt, new_src)
                # Swap forward/backward for canonical order
                new_probs = EdgeProbabilities(
                    forward=probs.backward,
                    backward=probs.forward,
                    undirected=probs.undirected,
                    none=probs.none,
                )
            new_edges[key] = new_probs

        return PDG(new_nodes, new_edges)

    def serialise(
        self,
        data_type: str,
        data: Any,
    ) -> str:
        """Serialise data to string format.

        This provider does not support direct serialisation. PDG objects
        are compressed by causaliq-core's CoreActionProvider.

        Args:
            data_type: Type of data.
            data: Data to serialise.

        Raises:
            NotImplementedError: Always, as this provider doesn't
                handle serialisation.
        """
        raise NotImplementedError(
            f"Provider '{self.name}' does not support serialisation. "
            f"PDG compression is handled by causaliq-core."
        )

    def deserialise(
        self,
        data_type: str,
        content: str,
    ) -> Any:
        """Deserialise data from string format.

        This provider does not support direct deserialisation. PDG objects
        are decompressed by causaliq-core's CoreActionProvider.

        Args:
            data_type: Type of data.
            content: String content to deserialise.

        Raises:
            NotImplementedError: Always, as this provider doesn't
                handle deserialisation.
        """
        raise NotImplementedError(
            f"Provider '{self.name}' does not support deserialisation. "
            f"PDG decompression is handled by causaliq-core."
        )


# Export as ActionProvider for auto-discovery by causaliq-workflow
# This name is required by the auto-discovery convention
ActionProvider = KnowledgeActionProvider
