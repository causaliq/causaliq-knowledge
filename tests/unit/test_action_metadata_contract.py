"""Contract tests for causaliq-knowledge action metadata."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from causaliq_core.graph.pdg import PDG, EdgeProbabilities
from causaliq_workflow.workflow.provenance import build_result_metadata

from causaliq_knowledge.action import KnowledgeActionProvider


def _make_mock_result(pdg: Any) -> MagicMock:
    """Create a minimal mock PDG generation result."""
    metadata = MagicMock()
    metadata.to_dict.return_value = {"llm_model": "test-model"}
    metadata.from_cache = False

    result = MagicMock()
    result.pdg = pdg
    result.metadata = metadata
    result.raw_response = '{"edges": []}'
    return result


# Test generate_graph flat metadata becomes the version 2 result block.
def test_generate_graph_metadata_contract(tmp_path: Path) -> None:
    """Test generate_graph metadata conforms to the version 2 contract."""
    context_file = tmp_path / "model.json"
    context_file.write_text(
        '{"schema_version": "2.0", "network": "test", '
        '"domain": "test", "variables": ['
        '{"name": "x", "type": "binary"}, '
        '{"name": "y", "type": "binary"}]}'
    )
    pdg = PDG(
        ["x", "y"],
        {
            ("x", "y"): EdgeProbabilities(
                forward=0.8, backward=0.1, undirected=0.0, none=0.1
            )
        },
    )

    with patch(
        "causaliq_knowledge.graph.generator.GraphGenerator"
    ) as generator_class:
        generator = MagicMock()
        generator.generate_pdg_from_context.return_value = _make_mock_result(
            pdg
        )
        generator_class.return_value = generator

        status, metadata, objects = KnowledgeActionProvider().run(
            "generate_graph",
            {
                "network_context": str(context_file),
                "output": "none",
                "llm_cache": "none",
            },
            mode="run",
        )

    # The framework must place flat action metadata in result
    envelope = build_result_metadata(
        {"provider": "causaliq-knowledge", "action": "generate_graph"},
        {"_action_metadata": metadata},
    )

    assert status == "success"
    assert envelope["parameters"] == {}
    assert envelope["result"]["edge_count"] == 1
    assert envelope["result"]["variable_count"] == 2
    assert envelope["provenance"]["provider"] == "causaliq-knowledge"
    assert envelope["provenance"]["action"] == "generate_graph"
    assert envelope["history"] == {}
    assert [obj["type"] for obj in objects] == ["pdg"]
