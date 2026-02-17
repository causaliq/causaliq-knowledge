"""Graph generation CLI commands.

This module provides commands for generating causal graphs from
network context using LLMs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import click
from pydantic import ValidationError

from causaliq_knowledge.graph.params import GenerateGraphParams
from causaliq_knowledge.graph.view_filter import PromptDetail

if TYPE_CHECKING:  # pragma: no cover
    from causaliq_knowledge.graph.models import NetworkContext
    from causaliq_knowledge.graph.response import GeneratedGraph


def _map_graph_names(
    graph: "GeneratedGraph", mapping: dict[str, str]
) -> "GeneratedGraph":
    """Map variable names in a graph using a mapping dictionary.

    Args:
        graph: The generated graph with edges to map.
        mapping: Dictionary mapping old names to new names.

    Returns:
        New GeneratedGraph with mapped variable names.
    """
    from causaliq_knowledge.graph.response import GeneratedGraph, ProposedEdge

    new_edges = []
    for edge in graph.edges:
        new_edge = ProposedEdge(
            source=mapping.get(edge.source, edge.source),
            target=mapping.get(edge.target, edge.target),
            confidence=edge.confidence,
        )
        new_edges.append(new_edge)

    # Map variable names too
    new_variables = [mapping.get(v, v) for v in graph.variables]

    return GeneratedGraph(
        edges=new_edges,
        variables=new_variables,
        reasoning=graph.reasoning,
        metadata=graph.metadata,
    )


@click.command("generate_graph")
@click.option(
    "--network-context",
    "-n",
    "context",
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
    help="Output: directory path, Workflow Cache .db file, or 'none'.",
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
    context: Path,
    prompt_detail: str,
    use_benchmark_names: bool,
    llm_model: str,
    output: str,
    llm_cache: str,
    llm_temperature: float,
) -> None:
    """Generate a causal graph from a network context.

    Reads variable definitions from a JSON network context file and
    uses an LLM to propose causal relationships between variables.

    By default, LLM names are used in prompts to prevent memorisation.
    Use --use-benchmark-names to test with original benchmark names.

    Output behaviour:

    \b
    - If output ends with .db: writes to Workflow Cache database
    - If output is a directory: writes graph.graphml, metadata.json,
      and confidences.json to that directory
    - If output is 'none': prints adjacency matrix to stdout

    Examples:

        cqknow generate_graph -s asia.json -c cache.db -o workflow.db

        cqknow generate_graph -s asia.json -c cache.db -o results/

        cqknow generate_graph -s asia.json -c cache.db -o none

        cqknow generate_graph -s asia.json -c none -o none --use-benchmark
    """
    # Import here to avoid slow startup for --help
    from causaliq_core.cache import TokenCache

    from causaliq_knowledge.graph import NetworkContext
    from causaliq_knowledge.graph.generator import (
        GraphGenerator,
        GraphGeneratorConfig,
    )
    from causaliq_knowledge.graph.prompts import OutputFormat

    # Validate all parameters using shared model
    try:
        params = GenerateGraphParams(
            context=context,
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
        ctx = NetworkContext.load(params.context)
        click.echo(
            f"Loaded network context: {ctx.network} "
            f"({len(ctx.variables)} variables)",
            err=True,
        )
    except Exception as e:
        click.echo(f"Error loading network context: {e}", err=True)
        sys.exit(1)

    # Track mapping for converting LLM output back to benchmark names
    llm_to_benchmark_mapping: dict[str, str] = {}

    # Determine naming mode
    use_llm_names = not params.use_benchmark_names
    if use_llm_names and ctx.uses_distinct_llm_names():
        llm_to_benchmark_mapping = ctx.get_llm_to_name_mapping()
        click.echo("Using LLM names (prevents memorisation)", err=True)
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
            click.echo(f"Error opening cache: {e}", err=True)
            sys.exit(1)
    else:
        click.echo("Cache disabled", err=True)

    # Create generator - use edge_list format for structured output
    try:
        # Derive request_id from output filename stem
        if params.output.lower() == "none":
            request_id = "none"
        else:
            request_id = Path(params.output).stem

        config = GraphGeneratorConfig(
            temperature=params.llm_temperature,
            output_format=OutputFormat.EDGE_LIST,
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

    # Generate graph
    click.echo(f"Generating graph using {params.llm_model}...", err=True)
    click.echo(f"View level: {params.prompt_detail.value}", err=True)

    try:
        graph = generator.generate_from_context(
            ctx, level=params.prompt_detail
        )
    except Exception as e:
        click.echo(f"Error generating graph: {e}", err=True)
        sys.exit(1)

    # Map LLM names back to benchmark names
    if llm_to_benchmark_mapping:
        graph = _map_graph_names(graph, llm_to_benchmark_mapping)
        click.echo("Mapped LLM names back to benchmark names", err=True)

    # Output results - always print edges summary to stderr
    _print_edges(graph)
    _print_summary(graph, err=True)

    if params.is_directory_output():
        # Write to directory: graph.graphml, metadata.json, confidences.json
        assert output_path is not None  # Guaranteed by is_directory_output()
        _write_to_directory(
            output_path, graph, ctx, params.llm_model, params.prompt_detail
        )
        click.echo(f"\nOutput written to: {output_path}/", err=True)
    elif params.is_workflow_cache_output():
        # Write to Workflow Cache database
        assert output_path is not None
        _write_to_workflow_cache(output_path, graph, ctx)
        click.echo(f"\nOutput written to: {output_path}", err=True)
    else:
        # Print adjacency matrix to stdout
        click.echo()
        _print_adjacency_matrix(graph, ctx)

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


def _write_to_workflow_cache(
    output_path: Path,
    graph: "GeneratedGraph",
    context: "NetworkContext",
) -> None:
    """Write generated graph to Workflow Cache database.

    Creates a Workflow Cache database at the specified path and stores
    the generated graph using the network identifier as the cache key.

    Args:
        output_path: Path to the Workflow Cache .db file.
        graph: The GeneratedGraph result.
        context: The NetworkContext used (for cache key generation).
    """
    import base64

    from causaliq_workflow import WorkflowCache
    from causaliq_workflow.cache import CacheEntry

    from causaliq_knowledge.graph.cache import GraphEntryEncoder

    # Create parent directories if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build cache key from network context
    key_data = {"network": context.network}

    with WorkflowCache(str(output_path)) as wf_cache:
        encoder = GraphEntryEncoder()
        blob = encoder.encode_entry(graph, wf_cache.token_cache)
        # Base64-encode for JSON-safe storage in CacheEntry
        blob_b64 = base64.b64encode(blob).decode("ascii")
        entry = CacheEntry(metadata={"network": context.network})
        entry.add_object("graph", "graphml", blob_b64)
        wf_cache.put(key_data, entry)


def _write_to_directory(
    output_dir: Path,
    graph: "GeneratedGraph",
    context: "NetworkContext",
    llm_model: str,
    prompt_detail: PromptDetail,
) -> None:
    """Write graph output files to a directory.

    Creates three files:
    - graph.graphml: The graph structure in GraphML format
    - metadata.json: Dataset info, variables, reasoning, generation params
    - confidences.json: Edge confidences as {"source->target": confidence}

    Args:
        output_dir: Directory to write files to.
        graph: The GeneratedGraph result.
        context: The NetworkContext used.
        llm_model: LLM model identifier.
        prompt_detail: Prompt detail level used.
    """
    from causaliq_core.graph import SDG
    from causaliq_core.graph.io import graphml

    # Create directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build SDG from edges - SDG requires nodes list and edges as tuples
    nodes = [v.name for v in context.variables]
    edges = [(e.source, "->", e.target) for e in graph.edges]
    sdg = SDG(nodes, edges)

    # Write GraphML
    graphml_path = output_dir / "graph.graphml"
    graphml.write(sdg, str(graphml_path))

    # Build metadata dict with all fields at top level
    metadata: dict[str, Any] = {
        "network": context.network,
        "domain": context.domain,
        "llm_reasoning": graph.reasoning,
        "objects": [
            {
                "type": "graphml",
                "name": "graph",
            },
            {
                "type": "json",
                "name": "confidences",
            },
        ],
    }

    # Add generation metadata (flattened at top level)
    if graph.metadata:
        metadata.update(graph.metadata.to_dict())
    else:
        metadata["llm_model"] = llm_model
    # Add CLI-specific field not in GenerationMetadata
    metadata["llm_prompt_detail"] = prompt_detail.value

    # Write metadata.json
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Build confidences
    confidences = {
        f"{e.source}->{e.target}": e.confidence for e in graph.edges
    }

    # Write confidences.json
    confidences_path = output_dir / "confidences.json"
    confidences_path.write_text(
        json.dumps(confidences, indent=2), encoding="utf-8"
    )


def _print_edges(graph: "GeneratedGraph") -> None:
    """Print proposed edges with confidence bars.

    Args:
        graph: The GeneratedGraph result.
    """
    if not graph.edges:
        click.echo("\nNo edges proposed by the LLM.", err=True)
        return

    click.echo(f"\nProposed Edges ({len(graph.edges)}):\n", err=True)

    # Sort by confidence descending
    sorted_edges = sorted(
        graph.edges, key=lambda e: e.confidence, reverse=True
    )

    for i, edge in enumerate(sorted_edges, 1):
        conf_pct = edge.confidence * 100
        conf_bar = "█" * int(edge.confidence * 10) + "░" * (
            10 - int(edge.confidence * 10)
        )
        click.echo(
            f"  {i:2d}. {edge.source} → {edge.target}  "
            f"[{conf_bar}] {conf_pct:5.1f}%",
            err=True,
        )
        if edge.reasoning:
            # Wrap reasoning text
            reasoning = edge.reasoning[:100]
            if len(edge.reasoning) > 100:
                reasoning += "..."
            click.echo(f"      {reasoning}", err=True)


def _print_summary(graph: "GeneratedGraph", err: bool = False) -> None:
    """Print a brief summary of the generated graph.

    Args:
        graph: The GeneratedGraph result.
        err: Whether to print to stderr.
    """
    edge_count = len(graph.edges)
    high_conf = sum(1 for e in graph.edges if e.confidence >= 0.7)
    med_conf = sum(1 for e in graph.edges if 0.4 <= e.confidence < 0.7)
    low_conf = sum(1 for e in graph.edges if e.confidence < 0.4)

    click.echo(f"\nEdge Confidence Summary ({edge_count} edges):", err=err)
    click.echo(f"  High confidence (>=0.7): {high_conf}", err=err)
    click.echo(f"  Medium confidence (0.4-0.7): {med_conf}", err=err)
    click.echo(f"  Low confidence (<0.4): {low_conf}", err=err)


def _print_adjacency_matrix(
    graph: "GeneratedGraph", context: "NetworkContext"
) -> None:
    """Print adjacency matrix representation of the graph.

    Args:
        graph: The GeneratedGraph result.
        context: The NetworkContext used for variable names.
    """
    # Get variable names in order
    var_names = [v.name for v in context.variables]

    # Build edge lookup (source, target) -> confidence
    edge_lookup = {(e.source, e.target): e.confidence for e in graph.edges}

    click.echo("Adjacency Matrix:")
    click.echo()

    # Header row
    max_name_len = max(len(name) for name in var_names)
    header = " " * (max_name_len + 2)
    for name in var_names:
        header += f"{name[:3]:>4}"
    click.echo(header)

    # Data rows
    for i, row_name in enumerate(var_names):
        row = f"{row_name:<{max_name_len}}  "
        for j, col_name in enumerate(var_names):
            if (row_name, col_name) in edge_lookup:
                conf = edge_lookup[(row_name, col_name)]
                row += f"{conf:4.1f}"
            else:
                row += "   ."
        click.echo(row)
