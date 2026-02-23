"""Unit tests for the CausalIQ workflow action provider.

Tests for KnowledgeActionProvider which integrates causaliq-knowledge
graph generation into CausalIQ workflows.
"""

from io import StringIO
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from causaliq_core.graph.io import graphml

from causaliq_knowledge.action import (
    SUPPORTED_ACTIONS,
    ActionProvider,
    KnowledgeActionProvider,
)


def _make_mock_result(pdg: Any) -> MagicMock:
    """Create a mock PDGGenerationResult for testing.

    Args:
        pdg: The PDG to wrap.

    Returns:
        A mock result with PDG and metadata.
    """
    from datetime import datetime, timezone

    mock_metadata = MagicMock()
    mock_metadata.model = "test-model"
    mock_metadata.provider = "test"
    mock_metadata.timestamp = datetime.now(timezone.utc)
    mock_metadata.llm_timestamp = datetime.now(timezone.utc)
    mock_metadata.llm_latency_ms = 100
    mock_metadata.input_tokens = 500
    mock_metadata.output_tokens = 200
    mock_metadata.from_cache = False
    mock_metadata.messages = [{"role": "user", "content": "test"}]
    mock_metadata.temperature = 0.1
    mock_metadata.max_tokens = 2000
    mock_metadata.finish_reason = "stop"
    mock_metadata.llm_cost_usd = 0.001
    mock_metadata.to_dict.return_value = {
        "llm_model": "test-model",
        "llm_provider": "test",
        "timestamp": mock_metadata.timestamp.isoformat(),
        "llm_timestamp": mock_metadata.llm_timestamp.isoformat(),
        "llm_latency_ms": 100,
        "llm_input_tokens": 500,
        "llm_output_tokens": 200,
        "from_cache": False,
        "llm_messages": [{"role": "user", "content": "test"}],
        "llm_temperature": 0.1,
        "llm_max_tokens": 2000,
        "llm_finish_reason": "stop",
        "llm_cost_usd": 0.001,
    }

    mock_result = MagicMock()
    mock_result.pdg = pdg
    mock_result.metadata = mock_metadata
    mock_result.raw_response = '{"edges": []}'

    return mock_result


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
    assert KnowledgeActionProvider.version == "0.5.0"
    assert KnowledgeActionProvider.description != ""
    assert KnowledgeActionProvider.author == "CausalIQ"


# Test action has inputs specification.
def test_action_inputs_specification() -> None:
    """Test action has input specifications defined."""
    action = KnowledgeActionProvider()

    assert "action" in action.inputs
    assert "network_context" in action.inputs
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
            {
                "network_context": "model.json",
                "output": "none",
                "llm_cache": "none",
            },
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

    assert "network_context" in str(exc_info.value).lower()


# Test run rejects invalid LLM provider.
def test_run_rejects_invalid_llm_provider() -> None:
    """Test run fails for invalid LLM provider."""
    from causaliq_core import ActionValidationError

    action = KnowledgeActionProvider()

    with pytest.raises(ActionValidationError) as exc_info:
        action.run(
            "generate_graph",
            {
                "network_context": "model.json",
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
            "network_context": str(context_file),
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
                "network_context": "/nonexistent/model.json",
                "output": "none",
                "llm_cache": "cache.db",
            },
            mode="run",
        )

    assert "not found" in str(exc_info.value).lower()


# Test run mode executes graph generation.
def test_run_execute_mode(tmp_path: Path) -> None:
    """Test run mode executes graph generation with mocked LLM."""
    from causaliq_core.graph.pdg import PDG, EdgeProbabilities

    # Create a minimal context file
    context_file = tmp_path / "model.json"
    context_file.write_text(
        '{"schema_version": "2.0", "network": "test", '
        '"domain": "test", "variables": ['
        '{"name": "x", "type": "binary"}, '
        '{"name": "y", "type": "binary"}]}'
    )

    # Mock the graph generator to return a PDG
    mock_pdg = PDG(
        ["x", "y"],
        {
            ("x", "y"): EdgeProbabilities(
                forward=0.8, backward=0.1, undirected=0.0, none=0.1
            )
        },
    )

    with patch(
        "causaliq_knowledge.graph.generator.GraphGenerator"
    ) as mock_generator_class:
        mock_generator = MagicMock()
        mock_generator.generate_pdg_from_context.return_value = (
            _make_mock_result(mock_pdg)
        )
        mock_generator.get_stats.return_value = {"cache_hits": 0}
        mock_generator_class.return_value = mock_generator

        action = KnowledgeActionProvider()

        status, metadata, objects = action.run(
            "generate_graph",
            {
                "network_context": str(context_file),
                "output": "none",
                "llm_cache": "none",  # Disable cache for simpler test
            },
            mode="run",
        )

    assert status == "success"
    assert metadata["edge_count"] == 1
    assert metadata["variable_count"] == 2
    assert metadata["model_used"] == "groq/llama-3.1-8b-instant"
    assert len(objects) == 1
    # Check PDG object returned as GraphML string
    pdg_obj = next(o for o in objects if o["type"] == "pdg")
    assert isinstance(pdg_obj["content"], str)
    assert "graphml" in pdg_obj["content"]


# =============================================================================
# Output file tests removed - workflow now handles cache storage
# =============================================================================


# Directory output tests removed - workflow handles export.


# Graph-to-dict test removed - workflow uses serialised objects.


# Test map_pdg_names conversion.
def test_map_pdg_names() -> None:
    """Test _map_pdg_names maps variable names correctly."""
    from causaliq_core.graph.pdg import PDG, EdgeProbabilities

    pdg = PDG(
        ["old_a", "old_b"],
        {
            ("old_a", "old_b"): EdgeProbabilities(
                forward=0.8, backward=0.1, undirected=0.0, none=0.1
            )
        },
    )

    mapping = {"old_a": "new_a", "old_b": "new_b"}

    action = KnowledgeActionProvider()
    result = action._map_pdg_names(pdg, mapping)

    assert set(result.nodes) == {"new_a", "new_b"}
    assert ("new_a", "new_b") in result.edges


# Test map_pdg_names with partial mapping.
def test_map_pdg_names_partial() -> None:
    """Test _map_pdg_names handles partial mapping."""
    from causaliq_core.graph.pdg import PDG, EdgeProbabilities

    pdg = PDG(
        ["old_a", "keep_b"],
        {
            ("keep_b", "old_a"): EdgeProbabilities(
                forward=0.8, backward=0.1, undirected=0.0, none=0.1
            )
        },
    )

    # Only map old_a
    mapping = {"old_a": "new_a"}

    action = KnowledgeActionProvider()
    result = action._map_pdg_names(pdg, mapping)

    assert set(result.nodes) == {"new_a", "keep_b"}
    assert ("keep_b", "new_a") in result.edges


# Test map_pdg_names swaps probabilities when canonical order changes.
def test_map_pdg_names_swaps_probs_for_canonical_order() -> None:
    """Test _map_pdg_names swaps forward/backward when order reverses."""
    from causaliq_core.graph.pdg import PDG, EdgeProbabilities

    # Edge (a, b) with forward=0.8, backward=0.1
    pdg = PDG(
        ["a", "b"],
        {
            ("a", "b"): EdgeProbabilities(
                forward=0.8, backward=0.1, undirected=0.05, none=0.05
            )
        },
    )

    # Map so resulting names are NOT alphabetical: a->z, b->a
    # After mapping: (z, a) which needs canonical reorder to (a, z)
    mapping = {"a": "z", "b": "a"}

    action = KnowledgeActionProvider()
    result = action._map_pdg_names(pdg, mapping)

    assert set(result.nodes) == {"a", "z"}
    # Key should be canonical (a, z)
    assert ("a", "z") in result.edges
    # forward/backward should be swapped
    probs = result.edges[("a", "z")]
    assert abs(probs.forward - 0.1) < 0.001  # Was backward
    assert abs(probs.backward - 0.8) < 0.001  # Was forward
    assert abs(probs.undirected - 0.05) < 0.001
    assert abs(probs.none - 0.05) < 0.001


# Tests removed - _write_to_workflow_cache removed (workflow handles caching)


# Test request_id is derived from output filename.
def test_run_request_id_from_output(tmp_path: Path) -> None:
    """Test request_id is derived from output filename stem."""
    from causaliq_core.graph.pdg import PDG

    # Create a minimal context file
    context_file = tmp_path / "model.json"
    context_file.write_text(
        '{"schema_version": "2.0", "network": "test", '
        '"domain": "test", "variables": [{"name": "x", "type": "binary"}]}'
    )

    mock_pdg = PDG(["x"], {})

    captured_config = {}

    def capture_config(*args: Any, **kwargs: Any) -> MagicMock:
        captured_config.update(kwargs)
        mock = MagicMock()
        mock.generate_pdg_from_context.return_value = _make_mock_result(
            mock_pdg
        )
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
                "network_context": str(context_file),
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
                "network_context": str(tmp_path / "model.json"),
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
                "network_context": str(context_file),
                "output": "none",
                "llm_cache": "none",
            },
            mode="run",
        )

    assert "failed to load network context" in str(exc_info.value).lower()


# Test run uses LLM name mapping when context has distinct names.
def test_run_with_llm_name_mapping(tmp_path: Path) -> None:
    """Test run maps LLM names back to benchmark names when distinct."""
    from causaliq_core.graph.pdg import PDG, EdgeProbabilities

    # Create context with distinct llm_names
    context_file = tmp_path / "model.json"
    context_file.write_text(
        '{"schema_version": "2.0", "network": "test", '
        '"domain": "test", "variables": ['
        '{"name": "X1", "llm_name": "Variable One", "type": "binary"}, '
        '{"name": "X2", "llm_name": "Variable Two", "type": "binary"}]}'
    )

    # Mock PDG returns LLM names (note: canonical order is alphabetical)
    mock_pdg = PDG(
        ["Variable One", "Variable Two"],
        {
            ("Variable One", "Variable Two"): EdgeProbabilities(
                forward=0.8, backward=0.1, undirected=0.0, none=0.1
            )
        },
    )

    with patch(
        "causaliq_knowledge.graph.generator.GraphGenerator"
    ) as mock_generator_class:
        mock_generator = MagicMock()
        mock_generator.generate_pdg_from_context.return_value = (
            _make_mock_result(mock_pdg)
        )
        mock_generator.get_stats.return_value = {"cache_hits": 0}
        mock_generator_class.return_value = mock_generator

        action = KnowledgeActionProvider()

        status, metadata, objects = action.run(
            "generate_graph",
            {
                "network_context": str(context_file),
                "output": "none",
                "llm_cache": "none",
                "use_benchmark_names": False,  # Explicitly use LLM names
            },
            mode="run",
        )

    # Graph should have benchmark names (X1, X2), not LLM names
    assert status == "success"
    # Check PDG contains mapped names (parse GraphML string)
    pdg_obj = next(o for o in objects if o["type"] == "pdg")
    result_pdg = graphml.read_pdg(StringIO(pdg_obj["content"]))
    assert set(result_pdg.nodes) == {"X1", "X2"}
    assert ("X1", "X2") in result_pdg.edges


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
                    "network_context": str(context_file),
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
                    "network_context": str(context_file),
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
                    "network_context": str(context_file),
                    "output": "none",
                    "llm_cache": str(tmp_path / "cache.db"),
                },
                mode="run",
            )

    # Verify cache was opened and closed
    mock_cache.open.assert_called_once()
    mock_cache.close.assert_called_once()


# Test _build_execution_metadata returns full metadata with LLM info.
def test_build_execution_metadata_with_full_metadata(
    tmp_path: Path,
) -> None:
    """Test metadata includes expected fields for PDG output."""
    from causaliq_core.graph.pdg import PDG, EdgeProbabilities

    # Create a minimal context file
    context_file = tmp_path / "model.json"
    context_file.write_text(
        '{"schema_version": "2.0", "network": "test", '
        '"domain": "test", "variables": ['
        '{"name": "x", "type": "binary"}, '
        '{"name": "y", "type": "binary"}]}'
    )

    # Create PDG with edge probabilities
    mock_pdg = PDG(
        ["x", "y"],
        {
            ("x", "y"): EdgeProbabilities(
                forward=0.8, backward=0.1, undirected=0.0, none=0.1
            )
        },
    )

    with patch(
        "causaliq_knowledge.graph.generator.GraphGenerator"
    ) as mock_generator_class:
        mock_generator = MagicMock()
        mock_generator.generate_pdg_from_context.return_value = (
            _make_mock_result(mock_pdg)
        )
        mock_generator.get_stats.return_value = {
            "call_count": 1,
            "client_call_count": 1,
        }
        mock_generator_class.return_value = mock_generator

        action = KnowledgeActionProvider()

        status, metadata, objects = action.run(
            "generate_graph",
            {
                "network_context": str(context_file),
                "output": "none",
                "llm_cache": "none",
            },
            mode="run",
        )

    # Verify metadata includes expected fields
    assert metadata["edge_count"] == 1
    assert metadata["variable_count"] == 2
    # Comprehensive LLM metadata fields from GenerationMetadata
    assert "llm_model" in metadata
    assert "llm_provider" in metadata
    assert "llm_input_tokens" in metadata
    assert "llm_output_tokens" in metadata
    assert "llm_cost_usd" in metadata
    assert "llm_messages" in metadata
    assert "from_cache" in metadata


# =============================================================================
# serialise method tests
# =============================================================================


# Test serialise raises NotImplementedError (PDG handled by causaliq-core).
def test_serialise_raises_not_implemented() -> None:
    """Test serialise raises NotImplementedError for any type."""
    action = KnowledgeActionProvider()

    with pytest.raises(NotImplementedError) as exc_info:
        action.serialise("pdg", "data")

    assert "does not support serialisation" in str(exc_info.value)


# Test deserialise raises NotImplementedError (PDG handled by causaliq-core).
def test_deserialise_raises_not_implemented() -> None:
    """Test deserialise raises NotImplementedError for any type."""
    action = KnowledgeActionProvider()

    with pytest.raises(NotImplementedError) as exc_info:
        action.deserialise("pdg", "content")

    assert "does not support deserialisation" in str(exc_info.value)


# Test supported_types attribute is empty (PDG handled by causaliq-core).
def test_supported_types_attribute() -> None:
    """Test supported_types is empty set."""
    action = KnowledgeActionProvider()

    # PDG compression is handled by causaliq-core, not this provider
    assert action.supported_types == set()
