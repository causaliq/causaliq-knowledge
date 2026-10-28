"""Output writers for the graph generation CLI command.

Serialises a generated PDG to GraphML alongside its metadata document.
Kept apart from the command module so the command body stays within the
project limits.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:  # pragma: no cover
    from causaliq_core.graph.pdg import PDG

    from causaliq_knowledge.graph.models import NetworkContext
    from causaliq_knowledge.graph.response import GenerationMetadata

# Edge attributes written for every GraphML edge
_PROBABILITY_KEYS = ["p_forward", "p_backward", "p_undirected", "p_none"]


def _add_probability_keys(root: ET.Element) -> None:
    """Add the key definitions for the edge probability attributes."""
    for prob_name in _PROBABILITY_KEYS:
        key = ET.SubElement(root, "key", id=prob_name)
        key.set("for", "edge")
        key.set("attr.name", prob_name)
        key.set("attr.type", "double")


def _add_nodes(graph_elem: ET.Element, pdg: "PDG") -> None:
    """Add one GraphML node per variable, in order."""
    for node in pdg.nodes:
        ET.SubElement(graph_elem, "node", id=node)


def _add_edge_data(edge_elem: ET.Element, probs: Any) -> None:
    """Add the probability data attributes of one edge."""
    for attr_name, value in [
        ("p_forward", probs.forward),
        ("p_backward", probs.backward),
        ("p_undirected", probs.undirected),
        ("p_none", probs.none),
    ]:
        data = ET.SubElement(edge_elem, "data", key=attr_name)
        data.text = f"{value:.4f}"


def _add_edges(graph_elem: ET.Element, pdg: "PDG") -> None:
    """Add one GraphML edge per stored edge with a non-zero existence."""
    edge_id = 0
    for (source, target), probs in pdg.edges.items():
        # Only include edges with non-zero existence probability
        if probs.p_exist < 0.001:
            continue

        edge_id += 1
        edge_elem = ET.SubElement(
            graph_elem,
            "edge",
            id=f"e{edge_id}",
            source=source,
            target=target,
        )
        _add_edge_data(edge_elem, probs)


def write_pdg_graphml(pdg: "PDG", path: Path) -> None:
    """Write PDG to GraphML file with edge probability attributes.

    Creates a GraphML file where each edge has data attributes for
    the four probability values: p_forward, p_backward, p_undirected,
    p_none.

    Args:
        pdg: The PDG to write.
        path: Output file path.
    """
    graphml_ns = "http://graphml.graphdrawing.org/xmlns"

    # Build XML document
    root = ET.Element("graphml", xmlns=graphml_ns)

    # Add key definitions for edge probability attributes
    _add_probability_keys(root)

    # Create graph element
    graph_elem = ET.SubElement(root, "graph", id="G", edgedefault="directed")

    # Add nodes in order, then edges with probability attributes
    _add_nodes(graph_elem, pdg)
    _add_edges(graph_elem, pdg)

    # Write to file with XML declaration
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(str(path), encoding="unicode", xml_declaration=True)


def _build_metadata(
    pdg: "PDG",
    context: "NetworkContext",
    generation_metadata: "GenerationMetadata",
) -> Dict[str, Any]:
    """Build the metadata document written beside the graph."""
    # Count edges with significant existence probability
    edge_count = sum(1 for probs in pdg.edges.values() if probs.p_exist > 0.01)

    # Build comprehensive metadata dict using GenerationMetadata.to_dict()
    llm_metadata = generation_metadata.to_dict()

    return {
        # Network context info
        "network": context.network,
        "domain": context.domain,
        "variable_count": len(pdg.nodes),
        "edge_count": edge_count,
        # LLM generation metadata (from GenerationMetadata)
        **llm_metadata,
        # Output objects
        "objects": [
            {
                "type": "graphml",
                "name": "graph",
                "description": "PDG with edge probability attributes",
            },
        ],
    }


def write_to_directory(
    output_dir: Path,
    pdg: "PDG",
    context: "NetworkContext",
    generation_metadata: "GenerationMetadata",
) -> None:
    """Write PDG output files to a directory.

    Creates two files:

    - graph.graphml: The graph structure with edge probabilities

    - metadata.json: Comprehensive generation metadata

    Args:
        output_dir: Directory to write files to.
        pdg: The generated PDG result.
        context: The NetworkContext used.
        generation_metadata: Comprehensive LLM generation metadata.
    """
    # Create directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write GraphML with edge probabilities
    write_pdg_graphml(pdg, output_dir / "graph.graphml")

    metadata = _build_metadata(pdg, context, generation_metadata)

    # Write metadata.json
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
