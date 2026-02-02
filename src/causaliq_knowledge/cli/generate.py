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

from causaliq_knowledge.graph.prompts import OutputFormat
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


def _parse_output_format(value: str) -> OutputFormat:
    """Convert output format string to enum.

    Args:
        value: Output format string (edge_list, adjacency_matrix).

    Returns:
        OutputFormat enum value.
    """
    return OutputFormat(value.lower())


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

        cqknow generate graph -s model.json

        cqknow generate graph -s model.json -m groq/llama-3.1-8b-instant

        cqknow generate graph -s model.json --prompt-detail rich
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
    "--llm",
    "-m",
    "model",
    default="groq/llama-3.1-8b-instant",
    help="LLM model to use (e.g., groq/llama-3.1-8b-instant).",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output file path (JSON). Prints to stdout if not specified.",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    default="edge_list",
    type=click.Choice(["edge_list", "adjacency_matrix"], case_sensitive=False),
    help="Output format for the generated graph.",
)
@click.option(
    "--cache/--no-cache",
    "use_cache",
    default=True,
    help="Enable/disable LLM response caching. Default: enabled.",
)
@click.option(
    "--cache-path",
    "-c",
    "cache_path",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Path to cache database. Default: <model_spec>_cache.db",
)
@click.option(
    "--temperature",
    "-t",
    type=float,
    default=0.1,
    help="LLM temperature (0.0-1.0). Lower = more deterministic.",
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output result as JSON (default if --output is specified).",
)
@click.option(
    "--id",
    "request_id",
    default="cli",
    help="Request identifier for export filenames. Default: 'cli'.",
)
def generate_graph(
    model_spec: Path,
    prompt_detail: str,
    use_benchmark_names: bool,
    model: str,
    output: Optional[Path],
    output_format: str,
    use_cache: bool,
    cache_path: Optional[Path],
    temperature: float,
    output_json: bool,
    request_id: str,
) -> None:
    """Generate a causal graph from a model specification.

    Reads variable definitions from a JSON model specification file and
    uses an LLM to propose causal relationships between variables.

    By default, LLM names are used in prompts to prevent memorisation.
    Use --use-benchmark-names to test with original benchmark names.

    Examples:

        cqknow generate graph -s research/models/cancer/cancer.json

        cqknow generate graph -s model.json -m gemini/gemini-2.5-flash

        cqknow generate graph -s model.json --prompt-detail rich -o graph.json

        cqknow generate graph -s model.json --use-benchmark-names
    """
    # Import here to avoid slow startup for --help
    from causaliq_knowledge.cache import TokenCache
    from causaliq_knowledge.graph import ModelLoader
    from causaliq_knowledge.graph.generator import (
        GraphGenerator,
        GraphGeneratorConfig,
    )

    # Parse enums
    level = _parse_prompt_detail(prompt_detail)
    fmt = _parse_output_format(output_format)

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

    # Set up cache (enabled by default)
    cache: Optional[TokenCache] = None
    if use_cache:
        # Default cache path: alongside model spec file
        # e.g., cancer.json -> cancer_llm.db
        if cache_path is None:
            stem = model_spec.stem
            cache_path = model_spec.parent / f"{stem}_llm.db"
        try:
            cache = TokenCache(str(cache_path))
            cache.open()
            click.echo(f"Using cache: {cache_path}", err=True)
        except Exception as e:
            click.echo(f"Error opening cache: {e}", err=True)
            sys.exit(1)

    # Create generator
    try:
        config = GraphGeneratorConfig(
            temperature=temperature,
            output_format=fmt,
            prompt_detail=level,
            use_llm_names=use_llm_names,
            request_id=request_id,
        )
        generator = GraphGenerator(model=model, config=config, cache=cache)
    except ValueError as e:
        click.echo(f"Error creating generator: {e}", err=True)
        sys.exit(1)

    # Generate graph
    click.echo(f"Generating graph using {model}...", err=True)
    click.echo(f"View level: {level.value}, Format: {fmt.value}", err=True)

    try:
        graph = generator.generate_from_spec(spec, level=level)
    except Exception as e:
        click.echo(f"Error generating graph: {e}", err=True)
        sys.exit(1)

    # Map LLM names back to benchmark names
    if llm_to_benchmark_mapping:
        graph = _map_graph_names(graph, llm_to_benchmark_mapping)
        click.echo("Mapped LLM names back to benchmark names", err=True)

    # Build output
    result = _build_output(graph, spec, model, level, fmt)

    # Output results
    if output:
        # Write to file
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        click.echo(f"\nOutput written to: {output}", err=True)
        _print_summary(graph, err=True)
    elif output_json:
        # JSON to stdout
        click.echo(json.dumps(result, indent=2))
    else:
        # Human-readable to stdout
        _print_human_readable(graph, spec, model)

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
    model: str,
    level: PromptDetail,
    fmt: OutputFormat,
) -> dict:
    """Build output dictionary for the generated graph.

    Args:
        graph: The GeneratedGraph result.
        spec: The ModelSpec used.
        model: LLM model identifier.
        level: View level used.
        fmt: Output format used.

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
            "model": model,
            "prompt_detail": level.value,
            "output_format": fmt.value,
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

    click.echo(f"\nGenerated {edge_count} edges:", err=err)
    click.echo(f"  High confidence (>=0.7): {high_conf}", err=err)
    click.echo(f"  Medium confidence (0.4-0.7): {med_conf}", err=err)
    click.echo(f"  Low confidence (<0.4): {low_conf}", err=err)


def _print_human_readable(
    graph: GeneratedGraph, spec: ModelSpec, model: str
) -> None:
    """Print human-readable output for the generated graph.

    Args:
        graph: The GeneratedGraph result.
        spec: The ModelSpec used.
        model: LLM model identifier.
    """
    click.echo()
    click.echo("=" * 70)
    click.echo(f"Generated Causal Graph: {spec.dataset_id}")
    click.echo("=" * 70)
    click.echo(f"Domain:     {spec.domain}")
    click.echo(f"Variables:  {len(spec.variables)}")
    click.echo(f"Model:      {model}")
    click.echo("=" * 70)

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

    _print_summary(graph, err=False)
    click.echo()
