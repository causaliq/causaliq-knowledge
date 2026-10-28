"""Graph generation CLI commands.

This module provides commands for generating causal graphs from
network context using LLMs. Output is a PDG (Probabilistic Dependency
Graph) with edge probabilities.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, NoReturn, Optional

import click
from pydantic import ValidationError

from causaliq_knowledge.cli.generate_output import write_to_directory
from causaliq_knowledge.graph.params import GenerateGraphParams
from causaliq_knowledge.graph.view_filter import PromptDetail

if TYPE_CHECKING:  # pragma: no cover
    from causaliq_core.cache import TokenCache
    from causaliq_core.graph.pdg import PDG

    from causaliq_knowledge.graph.generator import GraphGenerator
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


@click.command("generate-graph")
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
    "-b/-B",
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
    help="LLM temperature (0.0-2.0). Lower = more deterministic.",
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

        cqknow generate-graph -n asia.json -c cache.db -o results/

        cqknow generate-graph -n asia.json -c none -o none

        cqknow generate-graph -n asia.json -c cache.db -o . --use-benchmark
    """
    params = _validate_params(
        network_context,
        prompt_detail,
        use_benchmark_names,
        llm_model,
        output,
        llm_cache,
        llm_temperature,
    )
    ctx = _load_context(params)
    use_llm_names, llm_to_benchmark_mapping = _resolve_name_mode(params, ctx)
    cache = _open_cache(params)
    generator = _create_generator(params, use_llm_names, cache)

    result = _generate_pdg(generator, ctx, params)
    pdg = _apply_name_mapping(result.pdg, llm_to_benchmark_mapping)

    _emit_output(params, pdg, result.metadata, ctx)
    _print_call_stats(generator)

    # Close cache if opened
    if cache:
        cache.close()


def _report_validation_errors(error: ValidationError) -> NoReturn:
    """Print Pydantic validation errors and exit the CLI."""
    # Format Pydantic errors for CLI
    for item in error.errors():
        field = item.get("loc", ["unknown"])[0]
        msg = item.get("msg", "validation error")
        click.echo(f"Error: --{field}: {msg}", err=True)
    sys.exit(1)


def _validate_params(
    network_context: Path,
    prompt_detail: str,
    use_benchmark_names: bool,
    llm_model: str,
    output: str,
    llm_cache: str,
    llm_temperature: float,
) -> GenerateGraphParams:
    """Validate the command options into a GenerateGraphParams model."""
    # Validate all parameters using shared model
    try:
        return GenerateGraphParams(
            network_context=network_context,
            prompt_detail=PromptDetail(prompt_detail.lower()),
            use_benchmark_names=use_benchmark_names,
            llm_model=llm_model,
            output=output,
            llm_cache=llm_cache,
            llm_temperature=llm_temperature,
        )
    except ValidationError as e:
        _report_validation_errors(e)


def _load_context(params: GenerateGraphParams) -> "NetworkContext":
    """Load the network context referenced by the parameters."""
    from causaliq_knowledge.graph import NetworkContext

    # Load network context
    try:
        ctx = NetworkContext.load(params.network_context)
        click.echo(
            f"Loaded network context: {ctx.network} "
            f"({len(ctx.variables)} variables)",
            err=True,
        )
        return ctx
    except Exception as e:
        click.echo(f"Error loading network context: {e}", err=True)
        sys.exit(1)


def _resolve_name_mode(
    params: GenerateGraphParams,
    ctx: "NetworkContext",
) -> tuple[bool, Dict[str, str]]:
    """Return the naming mode and the LLM-to-benchmark name mapping."""
    # Determine naming mode
    use_llm_names = not params.use_benchmark_names
    if use_llm_names and ctx.uses_distinct_llm_names():
        mapping = ctx.get_llm_to_name_mapping()
        click.echo(f"Using LLM names ({len(mapping)} mappings)", err=True)
        return use_llm_names, mapping
    if params.use_benchmark_names:
        click.echo("Using benchmark names (memorisation test)", err=True)
    return use_llm_names, {}


def _open_cache(params: GenerateGraphParams) -> Optional["TokenCache"]:
    """Open the LLM response cache when caching is enabled."""
    from causaliq_core.cache import TokenCache

    cache_path = params.get_effective_cache_path()
    if cache_path is None:
        click.echo("Cache disabled", err=True)
        return None
    try:
        cache = TokenCache(str(cache_path))
        cache.open()
        click.echo(f"Using cache: {cache_path}", err=True)
        return cache
    except Exception as e:
        click.echo(f"Warning: Failed to open cache: {e}", err=True)
        return None


def _create_generator(
    params: GenerateGraphParams,
    use_llm_names: bool,
    cache: Optional["TokenCache"],
) -> "GraphGenerator":
    """Create the graph generator for the requested model."""
    from causaliq_knowledge.graph.generator import (
        GraphGenerator,
        GraphGeneratorConfig,
    )

    # Derive request_id from output filename stem
    if params.output.lower() == "none":
        request_id = "cli-none"
    else:
        request_id = Path(params.output).stem or "cli-output"

    try:
        config = GraphGeneratorConfig(
            temperature=params.llm_temperature,
            timeout=params.llm_timeout,
            prompt_detail=params.prompt_detail,
            use_llm_names=use_llm_names,
            request_id=request_id,
        )
        return GraphGenerator(
            model=params.llm_model, config=config, cache=cache
        )
    except ValueError as e:
        click.echo(f"Error creating generator: {e}", err=True)
        sys.exit(1)


def _generate_pdg(
    generator: "GraphGenerator",
    ctx: "NetworkContext",
    params: GenerateGraphParams,
) -> Any:
    """Generate the PDG, reporting progress to stderr."""
    # Generate PDG
    click.echo(f"Generating graph using {params.llm_model}...", err=True)
    click.echo(f"View level: {params.prompt_detail.value}", err=True)

    try:
        return generator.generate_pdg_from_context(
            ctx, level=params.prompt_detail
        )
    except Exception as e:
        click.echo(f"Error generating graph: {e}", err=True)
        sys.exit(1)


def _apply_name_mapping(pdg: "PDG", mapping: Dict[str, str]) -> "PDG":
    """Map LLM names back to benchmark names when needed."""
    if not mapping:
        return pdg
    click.echo("Mapped LLM names back to benchmark names", err=True)
    return _map_pdg_names(pdg, mapping)


def _emit_output(
    params: GenerateGraphParams,
    pdg: "PDG",
    generation_metadata: "GenerationMetadata",
    ctx: "NetworkContext",
) -> None:
    """Print the edge summary and write the output files."""
    # Output results - always print edges summary to stderr
    _print_edges(pdg)
    _print_summary(pdg, err=True)

    if not params.is_directory_output():
        return

    # Write to directory: graph.graphml and metadata.json
    output_path = params.get_effective_output_path()
    assert output_path is not None
    write_to_directory(output_path, pdg, ctx, generation_metadata)
    click.echo(f"\nOutput written to: {output_path}/", err=True)


def _print_call_stats(generator: "GraphGenerator") -> None:
    """Print the LLM call statistics when any calls were made."""
    # Show stats
    stats = generator.get_stats()
    if stats.get("client_call_count", 0) > 0:
        click.echo(
            f"\nLLM calls: {stats['client_call_count']}, "
            f"Generator calls: {stats['call_count']}",
            err=True,
        )


def _format_direction(state: str, source: str, target: str) -> str:
    """Return the arrow notation for the most likely edge direction."""
    if state == "forward":
        return f"{source} -> {target}"
    if state == "backward":
        return f"{target} -> {source}"
    if state == "undirected":
        return f"{source} -- {target}"
    return f"{source} x {target}"


def _format_existence_bar(p_exist: float) -> str:
    """Return the unicode bar showing the existence probability."""
    filled = int(p_exist * 10)
    return "█" * filled + "░" * (10 - filled)


def _print_edge_breakdown(probs: Any) -> None:
    """Print the probability breakdown for mixed edges."""
    if probs.p_directed > 0.01 and probs.undirected > 0.01:
        click.echo(
            f"      -> {probs.forward*100:4.1f}%  "
            f"<- {probs.backward*100:4.1f}%  "
            f"-- {probs.undirected*100:4.1f}%",
            err=True,
        )


def _print_edge(index: int, source: str, target: str, probs: Any) -> None:
    """Print one proposed edge with its existence probability."""
    exist_pct = probs.p_exist * 100
    exist_bar = _format_existence_bar(probs.p_exist)
    direction = _format_direction(probs.most_likely_state(), source, target)

    click.echo(
        f"  {index:2d}. {direction}  "
        f"[{exist_bar}] {exist_pct:5.1f}% exists",
        err=True,
    )

    # Show probability breakdown for non-trivial cases
    _print_edge_breakdown(probs)


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
        _print_edge(i, source, target, probs)


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
