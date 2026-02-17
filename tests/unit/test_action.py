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


# Test run rejects unknown action.
def test_run_rejects_unknown_action() -> None:
    """Test run fails for unknown action type."""
    from causaliq_core import ActionValidationError

    action = KnowledgeActionProvider()

    with pytest.raises(ActionValidationError) as exc_info:
        action.run(
            "unknown_action",
            {"context": "model.json", "output": "none", "llm_cache": "none"},
            mode="dry-run",
        )

    assert "unknown action" in str(exc_info.value).lower()


# Test run rejects missing context for generate_graph.
def test_run_rejects_missing_context() -> None:
    """Test run fails when context missing for generate_graph."""
    from causaliq_core import ActionValidationError

    action = KnowledgeActionProvider()

    with pytest.raises(ActionValidationError) as exc_info:
        action.run("generate_graph", {}, mode="dry-run")

    assert "context" in str(exc_info.value).lower()


# Test run rejects invalid LLM provider.
def test_run_rejects_invalid_llm_provider() -> None:
    """Test run fails for invalid LLM provider."""
    from causaliq_core import ActionValidationError

    action = KnowledgeActionProvider()

    with pytest.raises(ActionValidationError) as exc_info:
        action.run(
            "generate_graph",
            {
                "context": "model.json",
                "llm_model": "invalid/model",
                "output": "none",
                "llm_cache": "cache.db",
            },
            mode="dry-run",
        )

    assert "provider" in str(exc_info.value).lower()


# Test dry-run mode returns skipped status.
def test_run_dry_run_mode(tmp_path: Path) -> None:
    """Test dry-run mode returns skipped status without executing."""
    # Create a minimal context file
    context_file = tmp_path / "model.json"
    context_file.write_text(
        '{"schema_version": "2.0", "network": "test", '
        '"domain": "test", "variables": [{"name": "x", "type": "binary"}]}'
    )

    action = KnowledgeActionProvider()

    status, metadata, objects = action.run(
        "generate_graph",
        {
            "context": str(context_file),
            "output": "none",
            "llm_cache": "cache.db",
        },
        mode="dry-run",
    )

    assert status == "skipped"
    assert "dry-run" in metadata.get("message", "").lower()
    assert metadata["llm_model"] == "groq/llama-3.1-8b-instant"
    assert metadata["llm_prompt_detail"] == "standard"
    assert objects == []


# Test run fails for non-existent context.
def test_run_context_not_found() -> None:
    """Test run fails when context file doesn't exist."""
    from causaliq_core import ActionExecutionError

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
        '{"schema_version": "2.0", "network": "test", '
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

        status, metadata, objects = action.run(
            "generate_graph",
            {
                "context": str(context_file),
                "output": "none",
                "llm_cache": "none",  # Disable cache for simpler test
            },
            mode="run",
        )

    assert status == "success"
    assert metadata["edge_count"] == 1
    assert metadata["variable_count"] == 2
    assert metadata["model_used"] == "groq/llama-3.1-8b-instant"
    assert len(objects) == 2
    # Check serialised GraphML content
    graphml_obj = next(o for o in objects if o["type"] == "graphml")
    assert "<graphml" in graphml_obj["content"]


# =============================================================================
# Output file tests removed - workflow now handles cache storage
# =============================================================================


# Directory output tests removed - workflow handles export.


# Graph-to-dict test removed - workflow uses serialised objects.


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


# Tests removed - _write_to_workflow_cache removed (workflow handles caching)


# Test request_id is derived from output filename.
def test_run_request_id_from_output(tmp_path: Path) -> None:
    """Test request_id is derived from output filename stem."""
    from causaliq_knowledge.graph.response import GeneratedGraph

    # Create a minimal context file
    context_file = tmp_path / "model.json"
    context_file.write_text(
        '{"schema_version": "2.0", "network": "test", '
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
    from causaliq_core import ActionExecutionError

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
    from causaliq_core import ActionExecutionError

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
        '{"schema_version": "2.0", "network": "test", '
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

        status, metadata, objects = action.run(
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
    assert status == "success"
    # Check serialised JSON contains mapped names
    import json

    json_obj = next(o for o in objects if o["type"] == "json")
    graph_data = json.loads(json_obj["content"])
    assert graph_data["variables"] == ["X1", "X2"]
    assert graph_data["edges"][0]["source"] == "X1"
    assert graph_data["edges"][0]["target"] == "X2"


# Test run fails when cache fails to open.
def test_run_cache_open_error(tmp_path: Path) -> None:
    """Test run fails when cache database fails to open."""
    from causaliq_core import ActionExecutionError

    # Create a minimal context file
    context_file = tmp_path / "model.json"
    context_file.write_text(
        '{"schema_version": "2.0", "network": "test", '
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

    assert "failed to open llm cache" in str(exc_info.value).lower()


# Test run fails when graph generation fails.
def test_run_graph_generation_error(tmp_path: Path) -> None:
    """Test run fails when graph generation raises an error."""
    from causaliq_core import ActionExecutionError

    # Create a minimal context file
    context_file = tmp_path / "model.json"
    context_file.write_text(
        '{"schema_version": "2.0", "network": "test", '
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
        '{"schema_version": "2.0", "network": "test", '
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


# Test _build_execution_metadata returns full metadata with timestamps.
def test_build_execution_metadata_with_full_metadata(
    tmp_path: Path,
) -> None:
    """Test metadata includes timestamps, messages and costs."""
    from datetime import datetime, timezone

    from causaliq_knowledge.graph.response import (
        GeneratedGraph,
        GenerationMetadata,
        ProposedEdge,
    )

    # Create a minimal context file
    context_file = tmp_path / "model.json"
    context_file.write_text(
        '{"schema_version": "2.0", "network": "test", '
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
            llm_latency_ms=1000,
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

        status, metadata, objects = action.run(
            "generate_graph",
            {
                "context": str(context_file),
                "output": "none",
                "llm_cache": "none",
            },
            mode="run",
        )

    # Verify metadata includes messages and costs
    assert "llm_messages" in metadata
    assert len(metadata["llm_messages"]) == 2
    assert metadata["llm_messages"][0]["role"] == "system"
    assert metadata["llm_messages"][1]["role"] == "user"

    # Check other metadata fields
    assert metadata["llm_temperature"] == 0.5
    assert metadata["llm_finish_reason"] == "stop"
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

    result = action.serialise("graphml", graph)

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
        action.serialise("graphml", "not a graph")

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

    result = action.deserialise("graphml", graphml_content)

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
    exported = action.serialise("graphml", original_graph)

    # Deserialise from string
    restored_graph = action.deserialise("graphml", exported)

    # Verify structure preserved
    assert set(restored_graph.variables) == set(original_graph.variables)
    assert len(restored_graph.edges) == len(original_graph.edges)

    original_edge_pairs = {(e.source, e.target) for e in original_graph.edges}
    restored_edge_pairs = {(e.source, e.target) for e in restored_graph.edges}
    assert restored_edge_pairs == original_edge_pairs


# Test serialise returns JSON string for json data type.
def test_serialise_returns_json() -> None:
    """Test serialise returns JSON string for json type."""
    import json
    from datetime import datetime, timezone

    from causaliq_knowledge.graph.response import (
        GeneratedGraph,
        GenerationMetadata,
        ProposedEdge,
    )

    graph = GeneratedGraph(
        edges=[ProposedEdge(source="A", target="B", confidence=0.8)],
        variables=["A", "B"],
        reasoning="Test reasoning",
        metadata=GenerationMetadata(
            model="test-model",
            provider="test-provider",
            llm_timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            llm_latency_ms=100,
            input_tokens=50,
            output_tokens=25,
            llm_cost_usd=0.001,
        ),
    )

    action = KnowledgeActionProvider()

    result = action.serialise("json", graph)

    assert isinstance(result, str)
    data = json.loads(result)
    assert data["variables"] == ["A", "B"]
    assert len(data["edges"]) == 1
    assert data["edges"][0]["source"] == "A"
    assert data["edges"][0]["target"] == "B"
    assert data["reasoning"] == "Test reasoning"
    assert data["metadata"]["llm_model"] == "test-model"
    assert data["metadata"]["llm_provider"] == "test-provider"
    assert data["metadata"]["llm_cost_usd"] == 0.001


# Test supported_types attribute is defined correctly.
def test_supported_types_attribute() -> None:
    """Test supported_types contains expected types."""
    action = KnowledgeActionProvider()

    assert action.supported_types == {"graphml", "json"}


# Test serialise JSON without metadata still works.
def test_serialise_json_without_metadata() -> None:
    """Test serialise JSON works when graph has no metadata."""
    import json

    from causaliq_knowledge.graph.response import GeneratedGraph, ProposedEdge

    graph = GeneratedGraph(
        edges=[ProposedEdge(source="A", target="B", confidence=0.9)],
        variables=["A", "B"],
        reasoning="No metadata",
    )

    action = KnowledgeActionProvider()

    result = action.serialise("json", graph)

    data = json.loads(result)
    assert "metadata" not in data
    assert data["variables"] == ["A", "B"]
    assert data["reasoning"] == "No metadata"
