"""Integration tests for causaliq-knowledge workflow integration.

These tests verify that causaliq-knowledge works correctly when executed
through causaliq-workflow. This tests the entry point discovery mechanism
and the complete workflow execution path.

These tests are marked as 'slow' since they involve workflow parsing
and action discovery.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_mock_result(pdg):
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


# Test entry point discovery finds causaliq-knowledge action.
def test_workflow_discovers_causaliq_knowledge_action() -> None:
    """Test that causaliq-workflow discovers causaliq-knowledge via entry."""
    from causaliq_workflow.registry import ActionRegistry

    registry = ActionRegistry()

    # Should find causaliq-knowledge via entry point
    assert registry.has_action("causaliq-knowledge"), (
        "causaliq-knowledge action not discovered. "
        f"Available: {registry.get_available_action_names()}"
    )


# Test entry point loads correct action class.
def test_workflow_loads_correct_action_class() -> None:
    """Test that loaded action class is KnowledgeActionProvider."""
    from causaliq_workflow.registry import ActionRegistry

    from causaliq_knowledge.action import KnowledgeActionProvider

    registry = ActionRegistry()
    action_class = registry.get_action_class("causaliq-knowledge")

    assert action_class is KnowledgeActionProvider


# Test workflow validates causaliq-knowledge step.
def test_workflow_validates_causaliq_knowledge_step(tmp_path: Path) -> None:
    """Test workflow validation passes for valid causaliq-knowledge step."""
    from causaliq_workflow.workflow import WorkflowExecutor

    # Create a minimal model spec
    context_file = tmp_path / "model.json"
    context_file.write_text(
        '{"schema_version": "2.0", "network": "test", '
        '"domain": "test", "variables": [{"name": "x", "type": "binary"}]}'
    )

    # Create workflow file
    workflow_yaml = tmp_path / "workflow.yaml"
    workflow_yaml.write_text(
        f"""
description: "Test workflow"
id: "test-workflow"

steps:
  - name: "Generate graph"
    uses: "causaliq-knowledge"
    with:
      action: "generate_graph"
      network_context: "{context_file.as_posix()}"
      output: "none"
      llm_cache: "none"
"""
    )

    executor = WorkflowExecutor()

    # Should parse and validate without error
    workflow = executor.parse_workflow(str(workflow_yaml))
    assert workflow["id"] == "test-workflow"
    assert len(workflow["steps"]) == 1


# Test workflow dry-run execution via causaliq-workflow.
def test_workflow_dry_run_execution(tmp_path: Path) -> None:
    """Test dry-run mode returns skipped status without executing."""
    from causaliq_workflow.workflow import WorkflowExecutor

    # Create a minimal model spec
    context_file = tmp_path / "model.json"
    context_file.write_text(
        '{"schema_version": "2.0", "network": "test", '
        '"domain": "test", "variables": [{"name": "x", "type": "binary"}]}'
    )

    # Create workflow file
    workflow_yaml = tmp_path / "workflow.yaml"
    workflow_yaml.write_text(
        f"""
description: "Test workflow"
id: "test-workflow"

steps:
  - name: "Generate graph"
    uses: "causaliq-knowledge"
    with:
      action: "generate_graph"
      network_context: "{context_file.as_posix()}"
      output: "none"
      llm_cache: "none"
      llm_model: "groq/llama-3.1-8b-instant"
"""
    )

    executor = WorkflowExecutor()
    workflow = executor.parse_workflow(str(workflow_yaml))

    # Execute in dry-run mode
    results = executor.execute_workflow(workflow, mode="dry-run")

    assert len(results) == 1
    step_results = results[0]["steps"]
    assert "Generate graph" in step_results
    assert step_results["Generate graph"]["status"] == "skipped"


# Test workflow run execution with mocked LLM.
def test_workflow_run_execution_with_mocked_llm(tmp_path: Path) -> None:
    """Test run mode executes graph generation via workflow."""
    from causaliq_workflow.workflow import WorkflowExecutor

    # Create a minimal model spec
    context_file = tmp_path / "model.json"
    context_file.write_text(
        '{"schema_version": "2.0", "network": "test", '
        '"domain": "test", "variables": ['
        '{"name": "smoking", "type": "binary"}, '
        '{"name": "cancer", "type": "binary"}]}'
    )

    # Create workflow file
    workflow_yaml = tmp_path / "workflow.yaml"
    workflow_yaml.write_text(
        f"""
description: "Test workflow"
id: "test-workflow"

steps:
  - name: "Generate graph"
    uses: "causaliq-knowledge"
    with:
      action: "generate_graph"
      network_context: "{context_file.as_posix()}"
      output: "none"
      llm_cache: "none"
      llm_model: "groq/llama-3.1-8b-instant"
"""
    )

    # Mock the graph generator to return PDG
    from causaliq_core.graph.pdg import PDG, EdgeProbabilities

    mock_pdg = PDG(
        ["smoking", "cancer"],
        {
            ("cancer", "smoking"): EdgeProbabilities(
                forward=0.1, backward=0.8, undirected=0.0, none=0.1
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

        executor = WorkflowExecutor()
        workflow = executor.parse_workflow(str(workflow_yaml))

        # Execute in run mode
        results = executor.execute_workflow(workflow, mode="run")

    assert len(results) == 1
    step_results = results[0]["steps"]
    assert "Generate graph" in step_results
    assert step_results["Generate graph"]["status"] == "success"
    # Metadata is flattened directly into result
    assert step_results["Generate graph"]["edge_count"] == 1


# Test workflow with output directory via causaliq-workflow.
def test_workflow_writes_output_file(tmp_path: Path) -> None:
    """Test workflow writes output files to directory when specified."""
    from causaliq_core.graph.pdg import PDG, EdgeProbabilities
    from causaliq_workflow.workflow import WorkflowExecutor

    # Create a minimal model spec
    context_file = tmp_path / "model.json"
    context_file.write_text(
        '{"schema_version": "2.0", "network": "test", '
        '"domain": "test", "variables": ['
        '{"name": "x", "type": "binary"}, '
        '{"name": "y", "type": "binary"}]}'
    )

    output_dir = tmp_path / "output"

    # Create workflow file
    workflow_yaml = tmp_path / "workflow.yaml"
    workflow_yaml.write_text(
        f"""
description: "Test workflow"
id: "test-workflow"

steps:
  - name: "Generate graph"
    uses: "causaliq-knowledge"
    with:
      action: "generate_graph"
      network_context: "{context_file.as_posix()}"
      output: "{output_dir.as_posix()}"
      llm_cache: "none"
"""
    )

    # Mock the graph generator to return PDG
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

        executor = WorkflowExecutor()
        workflow = executor.parse_workflow(str(workflow_yaml))
        results = executor.execute_workflow(workflow, mode="run")

    # Action now returns GraphML string for workflow cache compression
    step_results = results[0]["steps"]
    assert step_results["Generate graph"]["status"] == "success"
    # Check PDG returned as GraphML string for interchange
    objects = step_results["Generate graph"]["objects"]
    assert len(objects) == 1
    pdg_obj = next(o for o in objects if o["type"] == "pdg")
    assert isinstance(pdg_obj["content"], str)
    assert "graphml" in pdg_obj["content"]


# Test workflow rejects invalid action parameter.
def test_workflow_rejects_invalid_parameters(tmp_path: Path) -> None:
    """Test workflow execution fails for invalid parameters."""
    from causaliq_workflow.workflow import (
        WorkflowExecutionError,
        WorkflowExecutor,
    )

    # Create a minimal model spec
    context_file = tmp_path / "model.json"
    context_file.write_text(
        '{"schema_version": "2.0", "network": "test", '
        '"domain": "test", "variables": [{"name": "x", "type": "binary"}]}'
    )

    # Create workflow file with invalid llm_model (no provider prefix)
    workflow_yaml = tmp_path / "workflow.yaml"
    workflow_yaml.write_text(
        f"""
description: "Test workflow"
id: "test-workflow"

steps:
  - name: "Generate graph"
    uses: "causaliq-knowledge"
    with:
      action: "generate_graph"
      network_context: "{context_file.as_posix()}"
      output: "none"
      llm_cache: "none"
      llm_model: "invalid-model-no-provider"
"""
    )

    executor = WorkflowExecutor()
    workflow = executor.parse_workflow(str(workflow_yaml))

    # Validation happens during execute_workflow (in validate mode internally)
    with pytest.raises(WorkflowExecutionError) as exc_info:
        executor.execute_workflow(workflow, mode="dry-run")

    assert "provider" in str(exc_info.value).lower()


# Test workflow matrix expansion with causaliq-knowledge.
def test_workflow_matrix_expansion(tmp_path: Path) -> None:
    """Test workflow matrix expands correctly with causaliq-knowledge steps."""
    from causaliq_workflow.workflow import WorkflowExecutor

    # Create model specs
    model1 = tmp_path / "model1.json"
    model1.write_text(
        '{"schema_version": "2.0", "network": "m1", '
        '"domain": "test", "variables": [{"name": "a", "type": "binary"}]}'
    )
    model2 = tmp_path / "model2.json"
    model2.write_text(
        '{"schema_version": "2.0", "network": "m2", '
        '"domain": "test", "variables": [{"name": "b", "type": "binary"}]}'
    )

    # Create workflow with matrix
    workflow_yaml = tmp_path / "workflow.yaml"
    workflow_yaml.write_text(
        f"""
description: "Matrix workflow"
id: "matrix-test"

matrix:
  model:
    - "{model1.as_posix()}"
    - "{model2.as_posix()}"

steps:
  - name: "Generate graph for {{{{model}}}}"
    uses: "causaliq-knowledge"
    with:
      action: "generate_graph"
      network_context: "{{{{model}}}}"
      output: "none"
      llm_cache: "none"
"""
    )

    executor = WorkflowExecutor()
    workflow = executor.parse_workflow(str(workflow_yaml))

    # Execute in dry-run to verify matrix expansion
    results = executor.execute_workflow(workflow, mode="dry-run")

    # Should have 2 jobs (one per model in matrix)
    assert len(results) == 2
