"""Graph generation CLI commands.

This module provides commands for generating causal graphs from
network context using LLMs. Output is a PDG (Probabilistic Dependency
Graph) with edge probabilities.
"""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

import click
from pydantic import ValidationError

from causaliq_knowledge.graph.params import GenerateGraphParams
from causaliq_knowledge.graph.view_filter import PromptDetail

if TYPE_CHECKING:  # pragma: no cover
    from causaliq_core.graph.pdg import PDG

    from causaliq_knowledge.graph.models import NetworkContext
    from causaliq_knowledge.graph.response import GenerationMetadata


def _map_pdg_names(pdg: "PDG", mapping: Dict[str, str]) -> "PDG":
    """Map variable names in a PDG using a mapping dictionary.

    Args:
        pdg: The generated PDG with edges to map.
        mapping: Dictionary mapping old names to new names.

    Returns:
        New PDG with mapped variable names.
    """
    from causaliq_core.graph.pdg import PDG, EdgeProbabilities

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


@click.command("generate_graph")
@click.option(
    "--network-context",
    "-n",
    "network_context",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to network context JSON file.",
)
@click.option(
    "--prompt-detail",
    "-p",
    "prompt_detail",
    default="standard",
    type=click.Choice(["minimal", "standard", "rich"], case_sensitive=False),
    help="Detail level for variable information in prompts.",
)
@click.option(
    "--use-benchmark-names/--use-llm-names",
    "use_benchmark_names",
    default=False,
    help="Use benchmark names instead of LLM names (test memorisation).",
)
@click.option(
    "--llm-model",
    "-m",
    "llm_model",
    default="groq/llama-3.1-8b-instant",
    help="LLM model to use (e.g., groq/llama-3.1-8b-instant).",
)
@click.option(
    "--output",
    "-o",
    required=True,
    help="Output: directory path for GraphML/JSON, or 'none'.",
)
@click.option(
    "--llm-cache",
    "-c",
    "llm_cache",
    required=True,
    help="Path to cache database (.db) or 'none' to disable caching.",
)
@click.option(
    "--llm-temperature",
    "-t",
    type=float,
    default=0.1,
    help="LLM temperature (0.0-1.0). Lower = more deterministic.",
)
def generate_graph(
    network_context: Path,
    prompt_detail: str,
    use_benchmark_names: bool,
    llm_model: str,
    output: str,
    llm_cache: str,
    llm_temperature: float,
) -> None:
    """Generate a causal graph from a network context.

    Reads variable definitions from a JSON network context file and
    uses an LLM to propose causal relationships between variables
    with probability distributions.

    By default, LLM names are used in prompts to prevent memorisation.
    Use --use-benchmark-names to test with original benchmark names.

    Output behaviour:

    \b
    - If output is a directory: writes graph.graphml (with edge
      probabilities as attributes) and metadata.json
    - If output is 'none': prints edge probabilities to stderr

    Examples:

        cqknow generate_graph -n asia.json -c cache.db -o results/

        cqknow generate_graph -n asia.json -c none -o none

        cqknow generate_graph -n asia.json -c cache.db -o . --use-benchmark
    """
    # Import here to avoid slow startup for --help
    from causaliq_core.cache import TokenCache

    from causaliq_knowledge.graph import NetworkContext
    from causaliq_knowledge.graph.generator import (
        GraphGenerator,
        GraphGeneratorConfig,
    )

    # Validate all parameters using shared model
    try:
        params = GenerateGraphParams(
            network_context=network_context,
            prompt_detail=PromptDetail(prompt_detail.lower()),
            use_benchmark_names=use_benchmark_names,
            llm_model=llm_model,
            output=output,
            llm_cache=llm_cache,
            llm_temperature=llm_temperature,
        )
    except ValidationError as e:
        # Format Pydantic errors for CLI
        for error in e.errors():
            field = error.get("loc", ["unknown"])[0]
            msg = error.get("msg", "validation error")
            click.echo(f"Error: --{field}: {msg}", err=True)
        sys.exit(1)

    # Get effective paths from validated params
    output_path = params.get_effective_output_path()

    # Load network context
    try:
        ctx = NetworkContext.load(params.network_context)
        click.echo(
            f"Loaded network context: {ctx.network} "
            f"({len(ctx.variables)} variables)",
            err=True,
        )
    except Exception as e:
        click.echo(f"Error loading network context: {e}", err=True)
        sys.exit(1)

    # Track mapping for converting LLM output back to benchmark names
    llm_to_benchmark_mapping: Dict[str, str] = {}

    # Determine naming mode
    use_llm_names = not params.use_benchmark_names
    if use_llm_names and ctx.uses_distinct_llm_names():
        llm_to_benchmark_mapping = ctx.get_llm_to_name_mapping()
        click.echo(
            f"Using LLM names ({len(llm_to_benchmark_mapping)} mappings)",
            err=True,
        )
    elif params.use_benchmark_names:
        click.echo("Using benchmark names (memorisation test)", err=True)

    # Set up cache
    cache: Optional[TokenCache] = None
    cache_path = params.get_effective_cache_path()
    if cache_path is not None:
        try:
            cache = TokenCache(str(cache_path))
            cache.open()
            click.echo(f"Using cache: {cache_path}", err=True)
        except Exception as e:
            click.echo(f"Warning: Failed to open cache: {e}", err=True)
            cache = None
    else:
        click.echo("Cache disabled", err=True)

    # Create generator
    try:
        # Derive request_id from output filename stem
        if params.output.lower() == "none":
            request_id = "cli-none"
        else:
            request_id = Path(params.output).stem or "cli-output"

        config = GraphGeneratorConfig(
            temperature=params.llm_temperature,
            prompt_detail=params.prompt_detail,
            use_llm_names=use_llm_names,
            request_id=request_id,
        )
        generator = GraphGenerator(
            model=params.llm_model, config=config, cache=cache
        )
    except ValueError as e:
        click.echo(f"Error creating generator: {e}", err=True)
        sys.exit(1)

    # Generate PDG
    click.echo(f"Generating graph using {params.llm_model}...", err=True)
    click.echo(f"View level: {params.prompt_detail.value}", err=True)

    try:
        result = generator.generate_pdg_from_context(
            ctx, level=params.prompt_detail
        )
        pdg = result.pdg
        generation_metadata = result.metadata
    except Exception as e:
        click.echo(f"Error generating graph: {e}", err=True)
        sys.exit(1)

    # Map LLM names back to benchmark names
    if llm_to_benchmark_mapping:
        pdg = _map_pdg_names(pdg, llm_to_benchmark_mapping)
        click.echo("Mapped LLM names back to benchmark names", err=True)

    # Output results - always print edges summary to stderr
    _print_edges(pdg)
    _print_summary(pdg, err=True)

    if params.is_directory_output():
        # Write to directory: graph.graphml and metadata.json
        assert output_path is not None
        _write_to_directory(output_path, pdg, ctx, generation_metadata)
        click.echo(f"\nOutput written to: {output_path}/", err=True)

    # Show stats
    stats = generator.get_stats()
    if stats.get("client_call_count", 0) > 0:
        click.echo(
            f"\nLLM calls: {stats['client_call_count']}, "
            f"Generator calls: {stats['call_count']}",
            err=True,
        )

    # Close cache if opened
    if cache:
        cache.close()


def _write_to_directory(
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
    graphml_path = output_dir / "graph.graphml"
    _write_pdg_graphml(pdg, graphml_path)

    # Count edges with significant existence probability
    edge_count = sum(1 for probs in pdg.edges.values() if probs.p_exist > 0.01)

    # Build comprehensive metadata dict using GenerationMetadata.to_dict()
    llm_metadata = generation_metadata.to_dict()

    metadata: Dict[str, Any] = {
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

    # Write metadata.json
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _write_pdg_graphml(pdg: "PDG", path: Path) -> None:
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
    for prob_name in ["p_forward", "p_backward", "p_undirected", "p_none"]:
        key = ET.SubElement(root, "key", id=prob_name)
        key.set("for", "edge")
        key.set("attr.name", prob_name)
        key.set("attr.type", "double")

    # Create graph element
    graph_elem = ET.SubElement(root, "graph", id="G", edgedefault="directed")

    # Add nodes in order
    for node in pdg.nodes:
        ET.SubElement(graph_elem, "node", id=node)

    # Add edges with probability attributes
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

        # Add probability data attributes
        for attr_name, value in [
            ("p_forward", probs.forward),
            ("p_backward", probs.backward),
            ("p_undirected", probs.undirected),
            ("p_none", probs.none),
        ]:
            data = ET.SubElement(edge_elem, "data", key=attr_name)
            data.text = f"{value:.4f}"

    # Write to file with XML declaration
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(str(path), encoding="unicode", xml_declaration=True)


def _print_edges(pdg: "PDG") -> None:
    """Print proposed edges with probability bars.

    Args:
        pdg: The PDG result.
    """
    # Get edges with non-zero existence probability
    existing = list(pdg.existing_edges())

    if not existing:
        click.echo("\nNo edges proposed by the LLM.", err=True)
        return

    click.echo(f"\nProposed Edges ({len(existing)}):\n", err=True)

    # Sort by existence probability descending
    sorted_edges = sorted(existing, key=lambda e: e[2].p_exist, reverse=True)

    for i, (source, target, probs) in enumerate(sorted_edges, 1):
        exist_pct = probs.p_exist * 100
        exist_bar = "█" * int(probs.p_exist * 10) + "░" * (
            10 - int(probs.p_exist * 10)
        )

        # Determine most likely direction
        state = probs.most_likely_state()
        if state == "forward":
            direction = f"{source} -> {target}"
        elif state == "backward":
            direction = f"{target} -> {source}"
        elif state == "undirected":
            direction = f"{source} -- {target}"
        else:
            direction = f"{source} x {target}"

        click.echo(
            f"  {i:2d}. {direction}  "
            f"[{exist_bar}] {exist_pct:5.1f}% exists",
            err=True,
        )

        # Show probability breakdown for non-trivial cases
        if probs.p_directed > 0.01 and probs.undirected > 0.01:
            click.echo(
                f"      -> {probs.forward*100:4.1f}%  "
                f"<- {probs.backward*100:4.1f}%  "
                f"-- {probs.undirected*100:4.1f}%",
                err=True,
            )


def _print_summary(pdg: "PDG", err: bool = False) -> None:
    """Print a brief summary of the generated PDG.

    Args:
        pdg: The PDG result.
        err: Whether to print to stderr.
    """
    # Count edges by existence probability
    existing = list(pdg.existing_edges())
    edge_count = len(existing)

    high_exist = sum(1 for _, _, p in existing if p.p_exist >= 0.7)
    med_exist = sum(1 for _, _, p in existing if 0.3 <= p.p_exist < 0.7)
    low_exist = sum(1 for _, _, p in existing if p.p_exist < 0.3)

    click.echo(f"\nEdge Existence Summary ({edge_count} edges):", err=err)
    click.echo(f"  High probability (>=0.7): {high_exist}", err=err)
    click.echo(f"  Medium probability (0.3-0.7): {med_exist}", err=err)
    click.echo(f"  Low probability (<0.3): {low_exist}", err=err)

    # Count directed vs undirected
    directed = sum(1 for _, _, p in existing if p.p_directed > p.undirected)
    undirected = edge_count - directed

    click.echo("\nEdge Direction Summary:", err=err)
    click.echo(f"  Mostly directed: {directed}", err=err)
    click.echo(f"  Mostly undirected: {undirected}", err=err)
