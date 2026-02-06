"""Unit tests for the CausalIQ workflow action.

Tests for GenerateGraphAction which integrates causaliq-knowledge
graph generation into CausalIQ workflows.
"""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from causaliq_knowledge.action import (
    SUPPORTED_ACTIONS,
    ActionExecutionError,
    CausalIQAction,
    GenerateGraphAction,
)


# Test CausalIQAction is aliased to GenerateGraphAction.
def test_causaliq_action_alias() -> None:
    """Test CausalIQAction is exported as alias to GenerateGraphAction."""
    assert CausalIQAction is GenerateGraphAction


# Test supported actions constant.
def test_supported_actions() -> None:
    """Test SUPPORTED_ACTIONS contains expected actions."""
    assert "generate_graph" in SUPPORTED_ACTIONS


# Test action class attributes.
def test_action_class_attributes() -> None:
    """Test action class has required metadata attributes."""
    assert GenerateGraphAction.name == "causaliq-knowledge"
    assert GenerateGraphAction.version == "0.4.0"
    assert GenerateGraphAction.description != ""
    assert GenerateGraphAction.author == "CausalIQ"


# Test action has inputs specification.
def test_action_inputs_specification() -> None:
    """Test action has input specifications defined."""
    action = GenerateGraphAction()

    assert "action" in action.inputs
    assert "model_spec" in action.inputs
    assert "prompt_detail" in action.inputs
    assert "llm_model" in action.inputs
    assert "output" in action.inputs
    assert "llm_cache" in action.inputs
    assert "llm_temperature" in action.inputs


# Test action has outputs specification.
def test_action_outputs_specification() -> None:
    """Test action has output specifications defined."""
    action = GenerateGraphAction()

    assert "graph" in action.outputs
    assert "edge_count" in action.outputs
    assert "variable_count" in action.outputs
    assert "model_used" in action.outputs
    assert "cached" in action.outputs


# Test validate_inputs rejects missing action parameter.
def test_validate_inputs_missing_action() -> None:
    """Test validation fails when action parameter is missing."""
    from causaliq_workflow.action import ActionValidationError

    action = GenerateGraphAction()

    with pytest.raises(ActionValidationError) as exc_info:
        action.validate_inputs({"model_spec": "model.json"})

    assert "action" in str(exc_info.value).lower()


# Test validate_inputs rejects unknown action.
def test_validate_inputs_unknown_action() -> None:
    """Test validation fails for unknown action type."""
    from causaliq_workflow.action import ActionValidationError

    action = GenerateGraphAction()

    with pytest.raises(ActionValidationError) as exc_info:
        action.validate_inputs(
            {
                "action": "unknown_action",
                "model_spec": "model.json",
            }
        )

    assert "unknown action" in str(exc_info.value).lower()


# Test validate_inputs rejects missing model_spec for generate_graph.
def test_validate_inputs_missing_model_spec() -> None:
    """Test validation fails when model_spec missing for generate_graph."""
    from causaliq_workflow.action import ActionValidationError

    action = GenerateGraphAction()

    with pytest.raises(ActionValidationError) as exc_info:
        action.validate_inputs({"action": "generate_graph"})

    assert "model_spec" in str(exc_info.value).lower()


# Test validate_inputs accepts valid generate_graph inputs.
def test_validate_inputs_valid_generate_graph() -> None:
    """Test validation passes for valid generate_graph inputs."""
    action = GenerateGraphAction()

    # Should not raise
    result = action.validate_inputs(
        {
            "action": "generate_graph",
            "model_spec": "model.json",
            "output": "none",
            "llm_cache": "cache.db",
        }
    )

    assert result is True


# Test validate_inputs validates LLM provider.
def test_validate_inputs_invalid_llm() -> None:
    """Test validation fails for invalid LLM provider."""
    from causaliq_workflow.action import ActionValidationError

    action = GenerateGraphAction()

    with pytest.raises(ActionValidationError) as exc_info:
        action.validate_inputs(
            {
                "action": "generate_graph",
                "model_spec": "model.json",
                "llm_model": "invalid/model",
                "output": "none",
                "llm_cache": "cache.db",
            }
        )

    assert "provider" in str(exc_info.value).lower()


# Test dry-run mode returns skipped status.
def test_run_dry_run_mode(tmp_path: Path) -> None:
    """Test dry-run mode returns skipped status without executing."""
    # Create a minimal model spec file
    model_spec = tmp_path / "model.json"
    model_spec.write_text(
        '{"schema_version": "2.0", "dataset_id": "test", '
        '"domain": "test", "variables": [{"name": "x", "type": "binary"}]}'
    )

    action = GenerateGraphAction()

    result = action.run(
        inputs={
            "action": "generate_graph",
            "model_spec": str(model_spec),
            "output": "none",
            "llm_cache": "cache.db",
        },
        mode="dry-run",
    )

    assert result["status"] == "skipped"
    assert "dry-run" in result.get("message", "").lower()
    assert result["llm_model"] == "groq/llama-3.1-8b-instant"
    assert result["prompt_detail"] == "standard"


# Test run fails for non-existent model_spec.
def test_run_model_spec_not_found() -> None:
    """Test run fails when model_spec file doesn't exist."""
    from causaliq_workflow.action import ActionExecutionError

    action = GenerateGraphAction()

    with pytest.raises(ActionExecutionError) as exc_info:
        action.run(
            inputs={
                "action": "generate_graph",
                "model_spec": "/nonexistent/model.json",
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

    # Create a minimal model spec file
    model_spec = tmp_path / "model.json"
    model_spec.write_text(
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
        mock_generator.generate_from_spec.return_value = mock_graph
        mock_generator.get_stats.return_value = {"cache_hits": 0}
        mock_generator_class.return_value = mock_generator

        action = GenerateGraphAction()

        result = action.run(
            inputs={
                "action": "generate_graph",
                "model_spec": str(model_spec),
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

    # Create a minimal model spec file
    model_spec = tmp_path / "model.json"
    model_spec.write_text(
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
        mock_generator.generate_from_spec.return_value = mock_graph
        mock_generator.get_stats.return_value = {"cache_hits": 0}
        mock_generator_class.return_value = mock_generator

        action = GenerateGraphAction()

        result = action.run(
            inputs={
                "action": "generate_graph",
                "model_spec": str(model_spec),
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

    # Create a minimal model spec file
    model_spec = tmp_path / "model.json"
    model_spec.write_text(
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
        mock_generator.generate_from_spec.return_value = mock_graph
        mock_generator.get_stats.return_value = {"cache_hits": 0}
        mock_generator_class.return_value = mock_generator

        action = GenerateGraphAction()

        result = action.run(
            inputs={
                "action": "generate_graph",
                "model_spec": str(model_spec),
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

    # Create a minimal model spec file
    model_spec = tmp_path / "model.json"
    model_spec.write_text(
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
        mock_generator.generate_from_spec.return_value = mock_graph
        mock_generator.get_stats.return_value = {"cache_hits": 0}
        mock_generator_class.return_value = mock_generator

        action = GenerateGraphAction()

        with pytest.raises(ActionExecutionError) as exc_info:
            action.run(
                inputs={
                    "action": "generate_graph",
                    "model_spec": str(model_spec),
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

    # Create a minimal model spec file
    model_spec = tmp_path / "model.json"
    model_spec.write_text(
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
        mock_generator.generate_from_spec.return_value = mock_graph
        mock_generator.get_stats.return_value = {"cache_hits": 0}
        mock_generator_class.return_value = mock_generator

        action = GenerateGraphAction()

        result = action.run(
            inputs={
                "action": "generate_graph",
                "model_spec": str(model_spec),
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

    # Create a minimal model spec file
    model_spec = tmp_path / "model.json"
    model_spec.write_text(
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
        mock_generator.generate_from_spec.return_value = mock_graph
        mock_generator.get_stats.return_value = {"cache_hits": 0}
        mock_generator_class.return_value = mock_generator

        action = GenerateGraphAction()

        result = action.run(
            inputs={
                "action": "generate_graph",
                "model_spec": str(model_spec),
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

    action = GenerateGraphAction()
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

    action = GenerateGraphAction()
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

    action = GenerateGraphAction()
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

    action = GenerateGraphAction()
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

    action = GenerateGraphAction()
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

    # Create a minimal model spec file
    model_spec = tmp_path / "model.json"
    model_spec.write_text(
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
        mock.generate_from_spec.return_value = mock_graph
        mock.get_stats.return_value = {"cache_hits": 0}
        return mock

    output_file = tmp_path / "results" / "expt01.db"

    with patch(
        "causaliq_knowledge.graph.generator.GraphGenerator",
        side_effect=capture_config,
    ):
        action = GenerateGraphAction()
        action.run(
            inputs={
                "action": "generate_graph",
                "model_spec": str(model_spec),
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

    action = GenerateGraphAction()

    # Call _run_generate_graph directly with invalid temperature
    # This bypasses validate_inputs and hits lines 268-269
    with pytest.raises(ActionExecutionError) as exc_info:
        action._run_generate_graph(
            inputs={
                "model_spec": str(tmp_path / "model.json"),
                "output": "none",
                "llm_cache": "none",
                "llm_temperature": 5.0,  # Invalid: must be 0.0-2.0
            },
            mode="run",
            context=None,
            logger=None,
        )

    assert "validation failed" in str(exc_info.value).lower()


# Test run fails when model spec loading fails.
def test_run_model_spec_load_error(tmp_path: Path) -> None:
    """Test run fails when model specification fails to load."""
    from causaliq_workflow.action import ActionExecutionError

    # Create a model spec file with invalid JSON structure
    model_spec = tmp_path / "model.json"
    model_spec.write_text('{"invalid": "not a valid model spec"}')

    action = GenerateGraphAction()

    with pytest.raises(ActionExecutionError) as exc_info:
        action.run(
            inputs={
                "action": "generate_graph",
                "model_spec": str(model_spec),
                "output": "none",
                "llm_cache": "none",
            },
            mode="run",
        )

    assert "failed to load model specification" in str(exc_info.value).lower()


# Test run uses LLM name mapping when spec has distinct names.
def test_run_with_llm_name_mapping(tmp_path: Path) -> None:
    """Test run maps LLM names back to benchmark names when distinct."""
    from causaliq_knowledge.graph.response import (
        GeneratedGraph,
        ProposedEdge,
    )

    # Create model spec with distinct llm_names
    model_spec = tmp_path / "model.json"
    model_spec.write_text(
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
        mock_generator.generate_from_spec.return_value = mock_graph
        mock_generator.get_stats.return_value = {"cache_hits": 0}
        mock_generator_class.return_value = mock_generator

        action = GenerateGraphAction()

        result = action.run(
            inputs={
                "action": "generate_graph",
                "model_spec": str(model_spec),
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

    # Create a minimal model spec file
    model_spec = tmp_path / "model.json"
    model_spec.write_text(
        '{"schema_version": "2.0", "dataset_id": "test", '
        '"domain": "test", "variables": [{"name": "x", "type": "binary"}]}'
    )

    action = GenerateGraphAction()

    with patch("causaliq_core.cache.TokenCache") as mock_cache_class:
        mock_cache = MagicMock()
        mock_cache.open.side_effect = Exception("Database locked")
        mock_cache_class.return_value = mock_cache

        with pytest.raises(ActionExecutionError) as exc_info:
            action.run(
                inputs={
                    "action": "generate_graph",
                    "model_spec": str(model_spec),
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

    # Create a minimal model spec file
    model_spec = tmp_path / "model.json"
    model_spec.write_text(
        '{"schema_version": "2.0", "dataset_id": "test", '
        '"domain": "test", "variables": ['
        '{"name": "x", "type": "binary"}, '
        '{"name": "y", "type": "binary"}]}'
    )

    with patch(
        "causaliq_knowledge.graph.generator.GraphGenerator"
    ) as mock_generator_class:
        mock_generator = MagicMock()
        mock_generator.generate_from_spec.side_effect = RuntimeError(
            "LLM API unavailable"
        )
        mock_generator_class.return_value = mock_generator

        action = GenerateGraphAction()

        with pytest.raises(ActionExecutionError) as exc_info:
            action.run(
                inputs={
                    "action": "generate_graph",
                    "model_spec": str(model_spec),
                    "output": "none",
                    "llm_cache": "none",
                },
                mode="run",
            )

    assert "graph generation failed" in str(exc_info.value).lower()


# Test cache is closed even when generation fails.
def test_run_cache_closed_on_error(tmp_path: Path) -> None:
    """Test cache is closed in finally block even when generation fails."""
    # Create a minimal model spec file
    model_spec = tmp_path / "model.json"
    model_spec.write_text(
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
        mock_generator.generate_from_spec.side_effect = RuntimeError(
            "Generation error"
        )
        mock_generator_class.return_value = mock_generator

        action = GenerateGraphAction()

        with pytest.raises(Exception):
            action.run(
                inputs={
                    "action": "generate_graph",
                    "model_spec": str(model_spec),
                    "output": "none",
                    "llm_cache": str(tmp_path / "cache.db"),
                },
                mode="run",
            )

    # Verify cache was opened and closed
    mock_cache.open.assert_called_once()
    mock_cache.close.assert_called_once()
