"""Unit tests for the CausalIQ workflow action provider.

Tests for KnowledgeActionProvider which integrates causaliq-knowledge
graph generation into CausalIQ workflows.
"""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from causaliq_knowledge.action import (
    SUPPORTED_ACTIONS,
    ActionExecutionError,
    ActionProvider,
    KnowledgeActionProvider,
)


# Test ActionProvider is aliased to KnowledgeActionProvider.
def test_action_provider_alias() -> None:
    """Test ActionProvider is alias to KnowledgeActionProvider."""
    assert ActionProvider is KnowledgeActionProvider


# Test supported actions constant.
def test_supported_actions() -> None:
    """Test SUPPORTED_ACTIONS contains expected actions."""
    assert "generate_graph" in SUPPORTED_ACTIONS
    assert len(SUPPORTED_ACTIONS) == 1


# Test action class attributes.
def test_action_class_attributes() -> None:
    """Test action class has required metadata attributes."""
    assert KnowledgeActionProvider.name == "causaliq-knowledge"
    assert KnowledgeActionProvider.version == "0.4.0"
    assert KnowledgeActionProvider.description != ""
    assert KnowledgeActionProvider.author == "CausalIQ"


# Test action has inputs specification.
def test_action_inputs_specification() -> None:
    """Test action has input specifications defined."""
    action = KnowledgeActionProvider()

    assert "action" in action.inputs
    assert "context" in action.inputs
    assert "prompt_detail" in action.inputs
    assert "llm_model" in action.inputs
    assert "output" in action.inputs
    assert "llm_cache" in action.inputs
    assert "llm_temperature" in action.inputs


# Test action has outputs specification.
def test_action_outputs_specification() -> None:
    """Test action has output specifications defined."""
    action = KnowledgeActionProvider()

    assert "graph" in action.outputs
    assert "edge_count" in action.outputs
    assert "variable_count" in action.outputs
    assert "model_used" in action.outputs
    assert "cached" in action.outputs


# Test validate_parameters rejects unknown action.
def test_validate_parameters_unknown_action() -> None:
    """Test validation fails for unknown action type."""
    from causaliq_workflow.action import ActionValidationError

    action = KnowledgeActionProvider()

    with pytest.raises(ActionValidationError) as exc_info:
        action.validate_parameters(
            "unknown_action",
            {"context": "model.json"},
        )

    assert "unknown action" in str(exc_info.value).lower()


# Test validate_parameters rejects missing context for generate_graph.
def test_validate_parameters_missing_context() -> None:
    """Test validation fails when context missing for generate_graph."""
    from causaliq_workflow.action import ActionValidationError

    action = KnowledgeActionProvider()

    with pytest.raises(ActionValidationError) as exc_info:
        action.validate_parameters("generate_graph", {})

    assert "context" in str(exc_info.value).lower()


# Test validate_parameters accepts valid generate_graph parameters.
def test_validate_parameters_valid_generate_graph() -> None:
    """Test validation passes for valid generate_graph parameters."""
    action = KnowledgeActionProvider()

    # Should not raise
    result = action.validate_parameters(
        "generate_graph",
        {
            "context": "model.json",
            "output": "none",
            "llm_cache": "cache.db",
        },
    )

    assert result is True


# Test validate_parameters validates LLM provider.
def test_validate_parameters_invalid_llm() -> None:
    """Test validation fails for invalid LLM provider."""
    from causaliq_workflow.action import ActionValidationError

    action = KnowledgeActionProvider()

    with pytest.raises(ActionValidationError) as exc_info:
        action.validate_parameters(
            "generate_graph",
            {
                "context": "model.json",
                "llm_model": "invalid/model",
                "output": "none",
                "llm_cache": "cache.db",
            },
        )

    assert "provider" in str(exc_info.value).lower()


# Test dry-run mode returns skipped status.
def test_run_dry_run_mode(tmp_path: Path) -> None:
    """Test dry-run mode returns skipped status without executing."""
    # Create a minimal context file
    context_file = tmp_path / "model.json"
    context_file.write_text(
        '{"schema_version": "2.0", "dataset_id": "test", '
        '"domain": "test", "variables": [{"name": "x", "type": "binary"}]}'
    )

    action = KnowledgeActionProvider()

    result = action.run(
        "generate_graph",
        {
            "context": str(context_file),
            "output": "none",
            "llm_cache": "cache.db",
        },
        mode="dry-run",
    )

    assert result["status"] == "skipped"
    assert "dry-run" in result.get("message", "").lower()
    assert result["llm_model"] == "groq/llama-3.1-8b-instant"
    assert result["prompt_detail"] == "standard"


# Test run fails for non-existent context.
def test_run_context_not_found() -> None:
    """Test run fails when context file doesn't exist."""
    from causaliq_workflow.action import ActionExecutionError

    action = KnowledgeActionProvider()

    with pytest.raises(ActionExecutionError) as exc_info:
        action.run(
            "generate_graph",
            {
                "context": "/nonexistent/model.json",
                "output": "none",
                "llm_cache": "cache.db",
            },
            mode="run",
        )

    assert "not found" in str(exc_info.value).lower()


# Test run mode executes graph generation.
def test_run_execute_mode(tmp_path: Path) -> None:
    """Test run mode executes graph generation with mocked LLM."""
    from causaliq_knowledge.graph.response import (
        GeneratedGraph,
        ProposedEdge,
    )

    # Create a minimal context file
    context_file = tmp_path / "model.json"
    context_file.write_text(
        '{"schema_version": "2.0", "dataset_id": "test", '
        '"domain": "test", "variables": ['
        '{"name": "x", "type": "binary"}, '
        '{"name": "y", "type": "binary"}]}'
    )

    # Mock the graph generator
    mock_graph = GeneratedGraph(
        edges=[ProposedEdge(source="x", target="y", confidence=0.8)],
        variables=["x", "y"],
        reasoning="Test reasoning",
    )

    with patch(
        "causaliq_knowledge.graph.generator.GraphGenerator"
    ) as mock_generator_class:
        mock_generator = MagicMock()
        mock_generator.generate_from_context.return_value = mock_graph
        mock_generator.get_stats.return_value = {"cache_hits": 0}
        mock_generator_class.return_value = mock_generator

        action = KnowledgeActionProvider()

        result = action.run(
            "generate_graph",
            {
                "context": str(context_file),
                "output": "none",
                "llm_cache": "none",  # Disable cache for simpler test
            },
            mode="run",
        )

    assert result["status"] == "success"
    assert result["edge_count"] == 1
    assert result["variable_count"] == 2
    assert result["model_used"] == "groq/llama-3.1-8b-instant"
    assert "graph" in result
    assert len(result["graph"]["edges"]) == 1


# Test run with output file.
def test_run_with_output_file(tmp_path: Path) -> None:
    """Test run writes output to file when specified."""
    from causaliq_knowledge.graph.response import (
        GeneratedGraph,
        ProposedEdge,
    )

    # Create a minimal context file
    context_file = tmp_path / "model.json"
    context_file.write_text(
        '{"schema_version": "2.0", "dataset_id": "test", '
        '"domain": "test", "variables": ['
        '{"name": "x", "type": "binary"}, '
        '{"name": "y", "type": "binary"}]}'
    )

    output_file = tmp_path / "output" / "workflow_cache.db"

    # Mock the graph generator
    mock_graph = GeneratedGraph(
        edges=[ProposedEdge(source="x", target="y", confidence=0.8)],
        variables=["x", "y"],
        reasoning="Test reasoning",
    )

    with patch(
        "causaliq_knowledge.graph.generator.GraphGenerator"
    ) as mock_generator_class:
        mock_generator = MagicMock()
        mock_generator.generate_from_context.return_value = mock_graph
        mock_generator.get_stats.return_value = {"cache_hits": 0}
        mock_generator_class.return_value = mock_generator

        action = KnowledgeActionProvider()

        result = action.run(
            "generate_graph",
            {
                "context": str(context_file),
                "output": str(output_file),
                "llm_cache": "none",
            },
            mode="run",
        )

    assert result["status"] == "success"
    assert "output_cache" in result
    assert output_file.exists()


# Test run writes to directory when directory output specified.
def test_run_with_directory_output(tmp_path: Path) -> None:
    """Test run writes GraphML and JSON files to directory."""
    import json

    from causaliq_knowledge.graph.response import (
        GeneratedGraph,
        ProposedEdge,
    )

    # Create a minimal context file
    context_file = tmp_path / "model.json"
    context_file.write_text(
        '{"schema_version": "2.0", "dataset_id": "test", '
        '"domain": "test", "variables": ['
        '{"name": "x", "type": "binary"}, '
        '{"name": "y", "type": "binary"}]}'
    )

    output_dir = tmp_path / "output"

    # Mock the graph generator
    mock_graph = GeneratedGraph(
        edges=[ProposedEdge(source="x", target="y", confidence=0.8)],
        variables=["x", "y"],
        reasoning="Test reasoning",
    )

    with patch(
        "causaliq_knowledge.graph.generator.GraphGenerator"
    ) as mock_generator_class:
        mock_generator = MagicMock()
        mock_generator.generate_from_context.return_value = mock_graph
        mock_generator.get_stats.return_value = {"cache_hits": 0}
        mock_generator_class.return_value = mock_generator

        action = KnowledgeActionProvider()

        result = action.run(
            "generate_graph",
            {
                "context": str(context_file),
                "output": str(output_dir),
                "llm_cache": "none",
            },
            mode="run",
        )

    assert result["status"] == "success"
    assert "output_dir" in result
    assert output_dir.exists()

    # Verify files created
    assert (output_dir / "graph.graphml").exists()
    assert (output_dir / "metadata.json").exists()
    assert (output_dir / "confidences.json").exists()

    # Verify confidences content
    confidences = json.loads((output_dir / "confidences.json").read_text())
    assert "x->y" in confidences
    assert confidences["x->y"] == 0.8


# Test directory output rejects matrix context.
def test_run_directory_output_rejects_matrix_context(tmp_path: Path) -> None:
    """Test directory output raises error when matrix context is present."""
    from causaliq_knowledge.graph.response import (
        GeneratedGraph,
        ProposedEdge,
    )

    # Create a minimal context file
    context_file = tmp_path / "model.json"
    context_file.write_text(
        '{"schema_version": "2.0", "dataset_id": "test", '
        '"domain": "test", "variables": ['
        '{"name": "x", "type": "binary"}, '
        '{"name": "y", "type": "binary"}]}'
    )

    output_dir = tmp_path / "output"

    # Mock the graph generator
    mock_graph = GeneratedGraph(
        edges=[ProposedEdge(source="x", target="y", confidence=0.8)],
        variables=["x", "y"],
    )

    # Create a mock context with matrix
    mock_context = MagicMock()
    mock_context.matrix = {"model": "test-model"}

    with patch(
        "causaliq_knowledge.graph.generator.GraphGenerator"
    ) as mock_generator_class:
        mock_generator = MagicMock()
        mock_generator.generate_from_context.return_value = mock_graph
        mock_generator.get_stats.return_value = {"cache_hits": 0}
        mock_generator_class.return_value = mock_generator

        action = KnowledgeActionProvider()

        with pytest.raises(ActionExecutionError) as exc_info:
            action.run(
                "generate_graph",
                {
                    "context": str(context_file),
                    "output": str(output_dir),
                    "llm_cache": "none",
                },
                mode="run",
                context=mock_context,
            )

        assert "matrix variables requires .db output" in str(exc_info.value)


# Test directory output includes edge reasoning.
def test_run_directory_output_with_edge_reasoning(tmp_path: Path) -> None:
    """Test directory output includes edge-level reasoning in metadata."""
    import json

    from causaliq_knowledge.graph.response import (
        GeneratedGraph,
        ProposedEdge,
    )

    # Create a minimal context file
    context_file = tmp_path / "model.json"
    context_file.write_text(
        '{"schema_version": "2.0", "dataset_id": "test", '
        '"domain": "test", "variables": ['
        '{"name": "x", "type": "binary"}, '
        '{"name": "y", "type": "binary"}]}'
    )

    output_dir = tmp_path / "output"

    # Mock the graph generator with edge reasoning
    mock_graph = GeneratedGraph(
        edges=[
            ProposedEdge(
                source="x",
                target="y",
                confidence=0.8,
                reasoning="X causes Y due to mechanism",
            )
        ],
        variables=["x", "y"],
    )

    with patch(
        "causaliq_knowledge.graph.generator.GraphGenerator"
    ) as mock_generator_class:
        mock_generator = MagicMock()
        mock_generator.generate_from_context.return_value = mock_graph
        mock_generator.get_stats.return_value = {"cache_hits": 0}
        mock_generator_class.return_value = mock_generator

        action = KnowledgeActionProvider()

        result = action.run(
            "generate_graph",
            {
                "context": str(context_file),
                "output": str(output_dir),
                "llm_cache": "none",
            },
            mode="run",
        )

    assert result["status"] == "success"

    # Verify edge reasoning in metadata
    metadata = json.loads((output_dir / "metadata.json").read_text())
    assert "edge_reasoning" in metadata
    assert metadata["edge_reasoning"]["x->y"] == "X causes Y due to mechanism"


# Test directory output includes generation metadata.
def test_run_directory_output_with_generation_metadata(tmp_path: Path) -> None:
    """Test directory output includes generation metadata."""
    import json
    from datetime import datetime, timezone

    from causaliq_knowledge.graph.response import (
        GeneratedGraph,
        GenerationMetadata,
        ProposedEdge,
    )

    # Create a minimal context file
    context_file = tmp_path / "model.json"
    context_file.write_text(
        '{"schema_version": "2.0", "dataset_id": "test", '
        '"domain": "test", "variables": ['
        '{"name": "x", "type": "binary"}, '
        '{"name": "y", "type": "binary"}]}'
    )

    output_dir = tmp_path / "output"

    # Mock the graph generator with generation metadata
    mock_graph = GeneratedGraph(
        edges=[ProposedEdge(source="x", target="y", confidence=0.8)],
        variables=["x", "y"],
        metadata=GenerationMetadata(
            model="test-model",
            provider="test-provider",
            timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            latency_ms=150.5,
            input_tokens=100,
            output_tokens=50,
            from_cache=False,
        ),
    )

    with patch(
        "causaliq_knowledge.graph.generator.GraphGenerator"
    ) as mock_generator_class:
        mock_generator = MagicMock()
        mock_generator.generate_from_context.return_value = mock_graph
        mock_generator.get_stats.return_value = {"cache_hits": 0}
        mock_generator_class.return_value = mock_generator

        action = KnowledgeActionProvider()

        result = action.run(
            "generate_graph",
            {
                "context": str(context_file),
                "output": str(output_dir),
                "llm_cache": "none",
            },
            mode="run",
        )

    assert result["status"] == "success"

    # Verify generation metadata
    metadata = json.loads((output_dir / "metadata.json").read_text())
    assert "generation" in metadata
    assert metadata["generation"]["model"] == "test-model"
    assert metadata["generation"]["provider"] == "test-provider"
    assert metadata["generation"]["input_tokens"] == 100
    assert metadata["generation"]["output_tokens"] == 50


# Test graph_to_dict conversion.
def test_graph_to_dict() -> None:
    """Test _graph_to_dict converts graph correctly."""
    from causaliq_knowledge.graph.response import (
        GeneratedGraph,
        ProposedEdge,
    )

    graph = GeneratedGraph(
        edges=[
            ProposedEdge(source="a", target="b", confidence=0.9),
            ProposedEdge(source="b", target="c", confidence=0.7),
        ],
        variables=["a", "b", "c"],
        reasoning="Test reasoning",
    )

    action = KnowledgeActionProvider()
    result = action._graph_to_dict(graph)

    assert result["variables"] == ["a", "b", "c"]
    assert result["reasoning"] == "Test reasoning"
    assert len(result["edges"]) == 2
    assert result["edges"][0]["source"] == "a"
    assert result["edges"][0]["target"] == "b"
    assert result["edges"][0]["confidence"] == 0.9


# Test map_graph_names conversion.
def test_map_graph_names() -> None:
    """Test _map_graph_names maps variable names correctly."""
    from causaliq_knowledge.graph.response import (
        GeneratedGraph,
        ProposedEdge,
    )

    graph = GeneratedGraph(
        edges=[
            ProposedEdge(source="old_a", target="old_b", confidence=0.9),
        ],
        variables=["old_a", "old_b"],
        reasoning="Test",
    )

    mapping = {"old_a": "new_a", "old_b": "new_b"}

    action = KnowledgeActionProvider()
    result = action._map_graph_names(graph, mapping)

    assert result.variables == ["new_a", "new_b"]
    assert result.edges[0].source == "new_a"
    assert result.edges[0].target == "new_b"


# Test map_graph_names with partial mapping.
def test_map_graph_names_partial() -> None:
    """Test _map_graph_names handles partial mapping."""
    from causaliq_knowledge.graph.response import (
        GeneratedGraph,
        ProposedEdge,
    )

    graph = GeneratedGraph(
        edges=[
            ProposedEdge(source="old_a", target="keep_b", confidence=0.9),
        ],
        variables=["old_a", "keep_b"],
        reasoning="Test",
    )

    # Only map old_a
    mapping = {"old_a": "new_a"}

    action = KnowledgeActionProvider()
    result = action._map_graph_names(graph, mapping)

    assert result.variables == ["new_a", "keep_b"]
    assert result.edges[0].source == "new_a"
    assert result.edges[0].target == "keep_b"


# Test _write_to_workflow_cache with no context uses fallback key.
def test_write_to_workflow_cache_no_context(tmp_path: Path) -> None:
    """Test _write_to_workflow_cache uses fallback key when context is None."""
    from causaliq_knowledge.graph.response import GeneratedGraph

    graph = GeneratedGraph(
        edges=[],
        variables=["x"],
        reasoning="Test",
    )

    mock_cache_instance = MagicMock()
    mock_cache_instance.__enter__ = MagicMock(return_value=mock_cache_instance)
    mock_cache_instance.__exit__ = MagicMock(return_value=False)
    mock_cache_cls = MagicMock(return_value=mock_cache_instance)
    mock_encoder_cls = MagicMock()

    output_path = tmp_path / "workflow_cache.db"

    action = KnowledgeActionProvider()
    action._write_to_workflow_cache(
        output_path=output_path,
        graph=graph,
        context=None,  # No context - should hit fallback
        workflow_cache_cls=mock_cache_cls,
        encoder_cls=mock_encoder_cls,
    )

    # Verify put was called with fallback key
    mock_cache_instance.put.assert_called_once()
    call_args = mock_cache_instance.put.call_args
    assert call_args[0][0] == {"source": "generate_graph"}


# Test _write_to_workflow_cache with context uses matrix key.
def test_write_to_workflow_cache_with_context(tmp_path: Path) -> None:
    """Test _write_to_workflow_cache uses matrix key when context provided."""
    from causaliq_knowledge.graph.response import GeneratedGraph

    graph = GeneratedGraph(
        edges=[],
        variables=["x"],
        reasoning="Test",
    )

    mock_cache_instance = MagicMock()
    mock_cache_instance.__enter__ = MagicMock(return_value=mock_cache_instance)
    mock_cache_instance.__exit__ = MagicMock(return_value=False)
    mock_cache_cls = MagicMock(return_value=mock_cache_instance)
    mock_encoder_cls = MagicMock()

    # Create mock context with matrix attribute
    mock_context = MagicMock()
    mock_context.matrix = {"model": "test.json", "seed": 42}

    output_path = tmp_path / "workflow_cache.db"

    action = KnowledgeActionProvider()
    action._write_to_workflow_cache(
        output_path=output_path,
        graph=graph,
        context=mock_context,  # Context with matrix
        workflow_cache_cls=mock_cache_cls,
        encoder_cls=mock_encoder_cls,
    )

    # Verify put was called with matrix key
    mock_cache_instance.put.assert_called_once()
    call_args = mock_cache_instance.put.call_args
    assert call_args[0][0] == {"model": "test.json", "seed": 42}


# Test request_id is derived from output filename.
def test_run_request_id_from_output(tmp_path: Path) -> None:
    """Test request_id is derived from output filename stem."""
    from causaliq_knowledge.graph.response import GeneratedGraph

    # Create a minimal context file
    context_file = tmp_path / "model.json"
    context_file.write_text(
        '{"schema_version": "2.0", "dataset_id": "test", '
        '"domain": "test", "variables": [{"name": "x", "type": "binary"}]}'
    )

    mock_graph = GeneratedGraph(
        edges=[],
        variables=["x"],
        reasoning="Test",
    )

    captured_config = {}

    def capture_config(*args: Any, **kwargs: Any) -> MagicMock:
        captured_config.update(kwargs)
        mock = MagicMock()
        mock.generate_from_context.return_value = mock_graph
        mock.get_stats.return_value = {"cache_hits": 0}
        return mock

    output_file = tmp_path / "results" / "expt01.db"

    with patch(
        "causaliq_knowledge.graph.generator.GraphGenerator",
        side_effect=capture_config,
    ):
        action = KnowledgeActionProvider()
        action.run(
            "generate_graph",
            {
                "context": str(context_file),
                "output": str(output_file),
                "llm_cache": "none",
            },
            mode="run",
        )

    # Check request_id was derived from output filename
    assert captured_config.get("config") is not None
    assert captured_config["config"].request_id == "expt01"

    # Cleanup is automatic via tmp_path fixture


# Test run fails for parameter validation error.
def test_run_generate_graph_validation_error(tmp_path: Path) -> None:
    """Test _run_generate_graph fails when param validation fails."""
    from causaliq_workflow.action import ActionExecutionError

    action = KnowledgeActionProvider()

    # Call _run_generate_graph directly with invalid temperature
    # This bypasses validate_parameters and hits lines 268-269
    with pytest.raises(ActionExecutionError) as exc_info:
        action._run_generate_graph(
            parameters={
                "context": str(tmp_path / "model.json"),
                "output": "none",
                "llm_cache": "none",
                "llm_temperature": 5.0,  # Invalid: must be 0.0-2.0
            },
            mode="run",
            context=None,
            logger=None,
        )

    assert "validation failed" in str(exc_info.value).lower()


# Test run fails when context loading fails.
def test_run_context_load_error(tmp_path: Path) -> None:
    """Test run fails when context fails to load."""
    from causaliq_workflow.action import ActionExecutionError

    # Create a context file with invalid JSON structure
    context_file = tmp_path / "model.json"
    context_file.write_text('{"invalid": "not a valid model spec"}')

    action = KnowledgeActionProvider()

    with pytest.raises(ActionExecutionError) as exc_info:
        action.run(
            "generate_graph",
            {
                "context": str(context_file),
                "output": "none",
                "llm_cache": "none",
            },
            mode="run",
        )

    assert "failed to load network context" in str(exc_info.value).lower()


# Test run uses LLM name mapping when context has distinct names.
def test_run_with_llm_name_mapping(tmp_path: Path) -> None:
    """Test run maps LLM names back to benchmark names when distinct."""
    from causaliq_knowledge.graph.response import (
        GeneratedGraph,
        ProposedEdge,
    )

    # Create context with distinct llm_names
    context_file = tmp_path / "model.json"
    context_file.write_text(
        '{"schema_version": "2.0", "dataset_id": "test", '
        '"domain": "test", "variables": ['
        '{"name": "X1", "llm_name": "Variable One", "type": "binary"}, '
        '{"name": "X2", "llm_name": "Variable Two", "type": "binary"}]}'
    )

    # Mock graph returns LLM names
    mock_graph = GeneratedGraph(
        edges=[
            ProposedEdge(
                source="Variable One", target="Variable Two", confidence=0.8
            )
        ],
        variables=["Variable One", "Variable Two"],
        reasoning="Test reasoning",
    )

    with patch(
        "causaliq_knowledge.graph.generator.GraphGenerator"
    ) as mock_generator_class:
        mock_generator = MagicMock()
        mock_generator.generate_from_context.return_value = mock_graph
        mock_generator.get_stats.return_value = {"cache_hits": 0}
        mock_generator_class.return_value = mock_generator

        action = KnowledgeActionProvider()

        result = action.run(
            "generate_graph",
            {
                "context": str(context_file),
                "output": "none",
                "llm_cache": "none",
                "use_benchmark_names": False,  # Explicitly use LLM names
            },
            mode="run",
        )

    # Graph should have benchmark names (X1, X2), not LLM names
    assert result["status"] == "success"
    assert result["graph"]["variables"] == ["X1", "X2"]
    assert result["graph"]["edges"][0]["source"] == "X1"
    assert result["graph"]["edges"][0]["target"] == "X2"


# Test run fails when cache fails to open.
def test_run_cache_open_error(tmp_path: Path) -> None:
    """Test run fails when cache database fails to open."""
    from causaliq_workflow.action import ActionExecutionError

    # Create a minimal context file
    context_file = tmp_path / "model.json"
    context_file.write_text(
        '{"schema_version": "2.0", "dataset_id": "test", '
        '"domain": "test", "variables": [{"name": "x", "type": "binary"}]}'
    )

    action = KnowledgeActionProvider()

    with patch("causaliq_core.cache.TokenCache") as mock_cache_class:
        mock_cache = MagicMock()
        mock_cache.open.side_effect = Exception("Database locked")
        mock_cache_class.return_value = mock_cache

        with pytest.raises(ActionExecutionError) as exc_info:
            action.run(
                "generate_graph",
                {
                    "context": str(context_file),
                    "output": "none",
                    "llm_cache": str(tmp_path / "cache.db"),  # Use actual path
                },
                mode="run",
            )

    assert "failed to open cache" in str(exc_info.value).lower()


# Test run fails when graph generation fails.
def test_run_graph_generation_error(tmp_path: Path) -> None:
    """Test run fails when graph generation raises an error."""
    from causaliq_workflow.action import ActionExecutionError

    # Create a minimal context file
    context_file = tmp_path / "model.json"
    context_file.write_text(
        '{"schema_version": "2.0", "dataset_id": "test", '
        '"domain": "test", "variables": ['
        '{"name": "x", "type": "binary"}, '
        '{"name": "y", "type": "binary"}]}'
    )

    with patch(
        "causaliq_knowledge.graph.generator.GraphGenerator"
    ) as mock_generator_class:
        mock_generator = MagicMock()
        mock_generator.generate_from_context.side_effect = RuntimeError(
            "LLM API unavailable"
        )
        mock_generator_class.return_value = mock_generator

        action = KnowledgeActionProvider()

        with pytest.raises(ActionExecutionError) as exc_info:
            action.run(
                "generate_graph",
                {
                    "context": str(context_file),
                    "output": "none",
                    "llm_cache": "none",
                },
                mode="run",
            )

    assert "graph generation failed" in str(exc_info.value).lower()


# Test cache is closed even when generation fails.
def test_run_cache_closed_on_error(tmp_path: Path) -> None:
    """Test cache is closed in finally block even when generation fails."""
    # Create a minimal context file
    context_file = tmp_path / "model.json"
    context_file.write_text(
        '{"schema_version": "2.0", "dataset_id": "test", '
        '"domain": "test", "variables": ['
        '{"name": "x", "type": "binary"}, '
        '{"name": "y", "type": "binary"}]}'
    )

    mock_cache = MagicMock()

    with (
        patch("causaliq_core.cache.TokenCache") as mock_cache_class,
        patch(
            "causaliq_knowledge.graph.generator.GraphGenerator"
        ) as mock_generator_class,
    ):
        mock_cache_class.return_value = mock_cache

        mock_generator = MagicMock()
        mock_generator.generate_from_context.side_effect = RuntimeError(
            "Generation error"
        )
        mock_generator_class.return_value = mock_generator

        action = KnowledgeActionProvider()

        with pytest.raises(Exception):
            action.run(
                "generate_graph",
                {
                    "context": str(context_file),
                    "output": "none",
                    "llm_cache": str(tmp_path / "cache.db"),
                },
                mode="run",
            )

    # Verify cache was opened and closed
    mock_cache.open.assert_called_once()
    mock_cache.close.assert_called_once()


# Test _populate_execution_metadata with timestamps and messages.
def test_populate_execution_metadata_with_full_metadata(
    tmp_path: Path,
) -> None:
    """Test metadata population includes timestamps and messages."""
    from datetime import datetime, timezone

    from causaliq_knowledge.graph.response import (
        GeneratedGraph,
        GenerationMetadata,
        ProposedEdge,
    )

    # Create a minimal context file
    context_file = tmp_path / "model.json"
    context_file.write_text(
        '{"schema_version": "2.0", "dataset_id": "test", '
        '"domain": "test", "variables": ['
        '{"name": "x", "type": "binary"}, '
        '{"name": "y", "type": "binary"}]}'
    )

    # Create metadata with all optional fields populated
    llm_time = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    messages = [
        {"role": "system", "content": "You are a causal expert."},
        {"role": "user", "content": "Identify causal relationships."},
    ]

    mock_graph = GeneratedGraph(
        edges=[ProposedEdge(source="x", target="y", confidence=0.8)],
        variables=["x", "y"],
        metadata=GenerationMetadata(
            model="test-model",
            provider="test-provider",
            timestamp=llm_time,
            latency_ms=1000,
            input_tokens=100,
            output_tokens=50,
            from_cache=False,
            messages=messages,
            temperature=0.5,
            max_tokens=2000,
            finish_reason="stop",
            llm_cost_usd=0.001,
        ),
    )

    with patch(
        "causaliq_knowledge.graph.generator.GraphGenerator"
    ) as mock_generator_class:
        mock_generator = MagicMock()
        mock_generator.generate_from_context.return_value = mock_graph
        mock_generator.get_stats.return_value = {"cache_hits": 0}
        mock_generator_class.return_value = mock_generator

        action = KnowledgeActionProvider()

        action.run(
            "generate_graph",
            {
                "context": str(context_file),
                "output": "none",
                "llm_cache": "none",
            },
            mode="run",
        )

    # Verify execution metadata includes messages and costs
    metadata = action.get_action_metadata()

    # Check messages are present
    assert "messages" in metadata
    assert len(metadata["messages"]) == 2
    assert metadata["messages"][0]["role"] == "system"
    assert metadata["messages"][1]["role"] == "user"

    # Check other metadata fields
    assert metadata["temperature"] == 0.5
    assert metadata["finish_reason"] == "stop"
    assert metadata["llm_cost_usd"] == 0.001


# =============================================================================
# serialise method tests
# =============================================================================


# Test serialise returns graphml string for graph data.
def test_serialise_returns_graphml() -> None:
    """Test serialise returns GraphML string."""
    from causaliq_knowledge.graph.response import GeneratedGraph, ProposedEdge

    graph = GeneratedGraph(
        edges=[ProposedEdge(source="A", target="B", confidence=0.8)],
        variables=["A", "B"],
        reasoning="Test",
    )

    action = KnowledgeActionProvider()

    result = action.serialise("graph", graph)

    assert isinstance(result, str)
    assert "<graphml" in result
    assert 'id="A"' in result
    assert 'id="B"' in result


# Test serialise raises NotImplementedError for unsupported data_type.
def test_serialise_unsupported_type() -> None:
    """Test serialise raises NotImplementedError for unknown type."""
    action = KnowledgeActionProvider()

    with pytest.raises(NotImplementedError) as exc_info:
        action.serialise("unknown", "data")

    assert "does not support serialising" in str(exc_info.value)


# Test serialise raises ValueError for wrong data type.
def test_serialise_wrong_data_type() -> None:
    """Test serialise raises ValueError when data is not GeneratedGraph."""
    action = KnowledgeActionProvider()

    with pytest.raises(ValueError) as exc_info:
        action.serialise("graph", "not a graph")

    assert "expected generatedgraph" in str(exc_info.value).lower()


# =============================================================================
# deserialise method tests
# =============================================================================


# Test deserialise returns GeneratedGraph from graphml string.
def test_deserialise_returns_graph() -> None:
    """Test deserialise returns GeneratedGraph from GraphML."""
    from causaliq_knowledge.graph.response import GeneratedGraph

    graphml_content = """<?xml version="1.0" encoding="utf-8"?>
    <graphml xmlns="http://graphml.graphdrawing.org/xmlns">
        <graph id="G" edgedefault="directed">
            <node id="A"/>
            <node id="B"/>
            <edge source="A" target="B"/>
        </graph>
    </graphml>"""

    action = KnowledgeActionProvider()

    result = action.deserialise("graph", graphml_content)

    assert isinstance(result, GeneratedGraph)
    assert set(result.variables) == {"A", "B"}
    assert len(result.edges) == 1


# Test deserialise raises NotImplementedError for unsupported data_type.
def test_deserialise_unsupported_type() -> None:
    """Test deserialise raises NotImplementedError for unknown type."""
    action = KnowledgeActionProvider()

    with pytest.raises(NotImplementedError) as exc_info:
        action.deserialise("unknown", "content")

    assert "does not support deserialising" in str(exc_info.value)


# Test serialise then deserialise roundtrip preserves graph structure.
def test_serialise_deserialise_roundtrip() -> None:
    """Test serialise then deserialise preserves graph structure."""
    from causaliq_knowledge.graph.response import GeneratedGraph, ProposedEdge

    # Create original graph
    original_graph = GeneratedGraph(
        edges=[
            ProposedEdge(source="A", target="B", confidence=0.9),
            ProposedEdge(source="B", target="C", confidence=0.8),
        ],
        variables=["A", "B", "C"],
        reasoning="Original graph",
    )

    action = KnowledgeActionProvider()

    # Serialise to string
    exported = action.serialise("graph", original_graph)

    # Deserialise from string
    restored_graph = action.deserialise("graph", exported)

    # Verify structure preserved
    assert set(restored_graph.variables) == set(original_graph.variables)
    assert len(restored_graph.edges) == len(original_graph.edges)

    original_edge_pairs = {(e.source, e.target) for e in original_graph.edges}
    restored_edge_pairs = {(e.source, e.target) for e in restored_graph.edges}
    assert restored_edge_pairs == original_edge_pairs
