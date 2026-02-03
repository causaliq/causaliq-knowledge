"""Graph generation CLI commands.

This module provides commands for generating causal graphs from
model specifications using LLMs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import click

from causaliq_knowledge.graph.view_filter import PromptDetail

if TYPE_CHECKING:  # pragma: no cover
    from causaliq_knowledge.graph.models import ModelSpec
    from causaliq_knowledge.graph.response import GeneratedGraph


def _parse_prompt_detail(value: str) -> PromptDetail:
    """Convert prompt detail string to enum.

    Args:
        value: Prompt detail string (minimal, standard, rich).

    Returns:
        PromptDetail enum value.
    """
    return PromptDetail(value.lower())


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


@click.group("generate")
def generate_group() -> None:
    """Generate causal structures using LLMs.

    Commands for generating causal graphs from model specifications.

    Examples:

        cqknow generate graph -s model.json -c model_llm.db -o none

        cqknow generate graph -s model.json -c cache.db -o graph.json

        cqknow generate graph -s model.json -c none -o none --prompt-detail
    """
    pass


@generate_group.command("graph")
@click.option(
    "--model-spec",
    "-s",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to model specification JSON file.",
)
@click.option(
    "--prompt-detail",
    "-v",
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
    help="Output: .json file path or 'none' for adjacency matrix to stdout.",
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
    model_spec: Path,
    prompt_detail: str,
    use_benchmark_names: bool,
    llm_model: str,
    output: str,
    llm_cache: str,
    llm_temperature: float,
) -> None:
    """Generate a causal graph from a model specification.

    Reads variable definitions from a JSON model specification file and
    uses an LLM to propose causal relationships between variables.

    By default, LLM names are used in prompts to prevent memorisation.
    Use --use-benchmark-names to test with original benchmark names.

    Output behaviour:
    - If output is a .json file: writes JSON to file, prints edges to stdout
    - If output is 'none': prints adjacency matrix to stdout, edges to stderr

    Examples:

        cqknow generate graph -s model.json -c cache.db -o graph.json

        cqknow generate graph -s model.json -c cache.db -o none

        cqknow generate graph -s model.json -c none -o none --use-benchmark
    """
    # Import here to avoid slow startup for --help
    from causaliq_knowledge.cache import TokenCache
    from causaliq_knowledge.graph import ModelLoader
    from causaliq_knowledge.graph.generator import (
        GraphGenerator,
        GraphGeneratorConfig,
    )
    from causaliq_knowledge.graph.prompts import OutputFormat

    # Parse enums
    level = _parse_prompt_detail(prompt_detail)

    # Validate output parameter
    output_path: Optional[Path] = None
    if output.lower() != "none":
        if not output.endswith(".json"):
            click.echo(
                "Error: --output must be 'none' or a path ending with .json",
                err=True,
            )
            sys.exit(1)
        output_path = Path(output)

    # Load model specification
    try:
        spec = ModelLoader.load(model_spec)
        click.echo(
            f"Loaded model specification: {spec.dataset_id} "
            f"({len(spec.variables)} variables)",
            err=True,
        )
    except Exception as e:
        click.echo(f"Error loading model specification: {e}", err=True)
        sys.exit(1)

    # Track mapping for converting LLM output back to benchmark names
    llm_to_benchmark_mapping: dict[str, str] = {}

    # Determine naming mode
    use_llm_names = not use_benchmark_names
    if use_llm_names and spec.uses_distinct_llm_names():
        llm_to_benchmark_mapping = spec.get_llm_to_name_mapping()
        click.echo("Using LLM names (prevents memorisation)", err=True)
    elif use_benchmark_names:
        click.echo("Using benchmark names (memorisation test)", err=True)

    # Set up cache
    cache: Optional[TokenCache] = None
    if llm_cache.lower() != "none":
        # Validate .db suffix
        if not llm_cache.endswith(".db"):
            click.echo(
                "Error: --llm-cache must be 'none' or a path ending with .db",
                err=True,
            )
            sys.exit(1)
        cache_path = Path(llm_cache)
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
        if output.lower() == "none":
            request_id = "none"
        else:
            request_id = Path(output).stem

        config = GraphGeneratorConfig(
            temperature=llm_temperature,
            output_format=OutputFormat.EDGE_LIST,
            prompt_detail=level,
            use_llm_names=use_llm_names,
            request_id=request_id,
        )
        generator = GraphGenerator(model=llm_model, config=config, cache=cache)
    except ValueError as e:
        click.echo(f"Error creating generator: {e}", err=True)
        sys.exit(1)

    # Generate graph
    click.echo(f"Generating graph using {llm_model}...", err=True)
    click.echo(f"View level: {level.value}", err=True)

    try:
        graph = generator.generate_from_spec(spec, level=level)
    except Exception as e:
        click.echo(f"Error generating graph: {e}", err=True)
        sys.exit(1)

    # Map LLM names back to benchmark names
    if llm_to_benchmark_mapping:
        graph = _map_graph_names(graph, llm_to_benchmark_mapping)
        click.echo("Mapped LLM names back to benchmark names", err=True)

    # Build JSON output
    result = _build_output(graph, spec, llm_model, level)

    # Output results - always print edges summary to stdout
    _print_edges(graph)
    _print_summary(graph, err=False)

    if output_path:
        # Write JSON to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        click.echo(f"\nOutput written to: {output_path}", err=True)
    else:
        # Print adjacency matrix to stdout
        click.echo()
        _print_adjacency_matrix(graph, spec)

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


def _build_output(
    graph: GeneratedGraph,
    spec: ModelSpec,
    llm_model: str,
    level: PromptDetail,
) -> dict:
    """Build output dictionary for the generated graph.

    Args:
        graph: The GeneratedGraph result.
        spec: The ModelSpec used.
        llm_model: LLM model identifier.
        level: View level used.

    Returns:
        Dictionary suitable for JSON output.
    """
    edges = []
    for edge in graph.edges:
        edge_dict = {
            "source": edge.source,
            "target": edge.target,
            "confidence": edge.confidence,
        }
        if edge.reasoning:
            edge_dict["reasoning"] = edge.reasoning
        edges.append(edge_dict)

    result = {
        "dataset_id": spec.dataset_id,
        "domain": spec.domain,
        "variable_count": len(spec.variables),
        "edge_count": len(edges),
        "edges": edges,
        "generation": {
            "model": llm_model,
            "prompt_detail": level.value,
        },
    }

    # Add metadata if available
    if graph.metadata:
        result["metadata"] = {
            "model": graph.metadata.model,
            "provider": graph.metadata.provider,
            "input_tokens": graph.metadata.input_tokens,
            "output_tokens": graph.metadata.output_tokens,
            "from_cache": graph.metadata.from_cache,
        }

    return result


def _print_edges(graph: GeneratedGraph) -> None:
    """Print proposed edges with confidence bars.

    Args:
        graph: The GeneratedGraph result.
    """
    if not graph.edges:
        click.echo("\nNo edges proposed by the LLM.")
        return

    click.echo(f"\nProposed Edges ({len(graph.edges)}):\n")

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
            f"[{conf_bar}] {conf_pct:5.1f}%"
        )
        if edge.reasoning:
            # Wrap reasoning text
            reasoning = edge.reasoning[:100]
            if len(edge.reasoning) > 100:
                reasoning += "..."
            click.echo(f"      {reasoning}")


def _print_summary(graph: GeneratedGraph, err: bool = False) -> None:
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


def _print_adjacency_matrix(graph: GeneratedGraph, spec: ModelSpec) -> None:
    """Print adjacency matrix representation of the graph.

    Args:
        graph: The GeneratedGraph result.
        spec: The ModelSpec used for variable names.
    """
    # Get variable names in order
    var_names = [v.name for v in spec.variables]

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
