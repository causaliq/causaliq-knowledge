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
    """Test that loaded action class is GenerateGraphAction."""
    from causaliq_workflow.registry import ActionRegistry

    from causaliq_knowledge.action import GenerateGraphAction

    registry = ActionRegistry()
    action_class = registry.get_action_class("causaliq-knowledge")

    assert action_class is GenerateGraphAction


# Test workflow validates causaliq-knowledge step.
def test_workflow_validates_causaliq_knowledge_step(tmp_path: Path) -> None:
    """Test workflow validation passes for valid causaliq-knowledge step."""
    from causaliq_workflow.workflow import WorkflowExecutor

    # Create a minimal model spec
    model_spec = tmp_path / "model.json"
    model_spec.write_text(
        '{"schema_version": "2.0", "dataset_id": "test", '
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
      model_spec: "{model_spec.as_posix()}"
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
    model_spec = tmp_path / "model.json"
    model_spec.write_text(
        '{"schema_version": "2.0", "dataset_id": "test", '
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
      model_spec: "{model_spec.as_posix()}"
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

    from causaliq_knowledge.graph.response import GeneratedGraph, ProposedEdge

    # Create a minimal model spec
    model_spec = tmp_path / "model.json"
    model_spec.write_text(
        '{"schema_version": "2.0", "dataset_id": "test", '
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
      model_spec: "{model_spec.as_posix()}"
      output: "none"
      llm_cache: "none"
      llm_model: "groq/llama-3.1-8b-instant"
"""
    )

    # Mock the graph generator
    mock_graph = GeneratedGraph(
        edges=[
            ProposedEdge(source="smoking", target="cancer", confidence=0.9)
        ],
        variables=["smoking", "cancer"],
        reasoning="Smoking causes cancer.",
    )

    with patch(
        "causaliq_knowledge.graph.generator.GraphGenerator"
    ) as mock_generator_class:
        mock_generator = MagicMock()
        mock_generator.generate_from_spec.return_value = mock_graph
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
    assert step_results["Generate graph"]["edge_count"] == 1


# Test workflow with output file via causaliq-workflow.
def test_workflow_writes_output_file(tmp_path: Path) -> None:
    """Test workflow writes output file when specified."""
    from causaliq_workflow.workflow import WorkflowExecutor

    from causaliq_knowledge.graph.response import GeneratedGraph, ProposedEdge

    # Create a minimal model spec
    model_spec = tmp_path / "model.json"
    model_spec.write_text(
        '{"schema_version": "2.0", "dataset_id": "test", '
        '"domain": "test", "variables": ['
        '{"name": "x", "type": "binary"}, '
        '{"name": "y", "type": "binary"}]}'
    )

    output_file = tmp_path / "output" / "graph.json"

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
      model_spec: "{model_spec.as_posix()}"
      output: "{output_file.as_posix()}"
      llm_cache: "none"
"""
    )

    # Mock the graph generator
    mock_graph = GeneratedGraph(
        edges=[ProposedEdge(source="x", target="y", confidence=0.8)],
        variables=["x", "y"],
        reasoning="Test.",
    )

    with patch(
        "causaliq_knowledge.graph.generator.GraphGenerator"
    ) as mock_generator_class:
        mock_generator = MagicMock()
        mock_generator.generate_from_spec.return_value = mock_graph
        mock_generator.get_stats.return_value = {"cache_hits": 0}
        mock_generator_class.return_value = mock_generator

        executor = WorkflowExecutor()
        workflow = executor.parse_workflow(str(workflow_yaml))
        executor.execute_workflow(workflow, mode="run")

    # Check output file was created
    assert output_file.exists(), f"Output file not created: {output_file}"

    # Verify content
    import json

    content = json.loads(output_file.read_text())
    assert "edges" in content
    assert len(content["edges"]) == 1


# Test workflow rejects invalid action parameter.
def test_workflow_rejects_invalid_parameters(tmp_path: Path) -> None:
    """Test workflow execution fails for invalid parameters."""
    from causaliq_workflow.workflow import (
        WorkflowExecutionError,
        WorkflowExecutor,
    )

    # Create a minimal model spec
    model_spec = tmp_path / "model.json"
    model_spec.write_text(
        '{"schema_version": "2.0", "dataset_id": "test", '
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
      model_spec: "{model_spec.as_posix()}"
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
        '{"schema_version": "2.0", "dataset_id": "m1", '
        '"domain": "test", "variables": [{"name": "a", "type": "binary"}]}'
    )
    model2 = tmp_path / "model2.json"
    model2.write_text(
        '{"schema_version": "2.0", "dataset_id": "m2", '
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
      model_spec: "{{{{model}}}}"
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
