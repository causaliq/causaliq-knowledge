"""Main CLI entry point and core commands.

This module provides the main CLI group and the query command for
querying LLMs about causal relationships between variables.
"""

from __future__ import annotations

import json
import sys
from typing import Optional

import click

from causaliq_knowledge import __version__


@click.group()
@click.version_option(version=__version__)
def cli() -> None:
    """CausalIQ Knowledge - LLM knowledge for causal discovery.

    Query LLMs about causal relationships between variables.
    """
    pass


@cli.command("query")
@click.argument("node_a")
@click.argument("node_b")
@click.option(
    "--model",
    "-m",
    multiple=True,
    default=["groq/llama-3.1-8b-instant"],
    help="LLM model(s) to query. Can be specified multiple times.",
)
@click.option(
    "--domain",
    "-d",
    default=None,
    help="Domain context (e.g., 'medicine', 'economics').",
)
@click.option(
    "--strategy",
    "-s",
    type=click.Choice(["weighted_vote", "highest_confidence"]),
    default="weighted_vote",
    help="Consensus strategy for multi-model queries.",
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output result as JSON.",
)
@click.option(
    "--llm-temperature",
    "-t",
    type=float,
    default=0.1,
    help="LLM temperature (0.0-1.0).",
)
def query_edge(
    node_a: str,
    node_b: str,
    model: tuple[str, ...],
    domain: Optional[str],
    strategy: str,
    output_json: bool,
    llm_temperature: float,
) -> None:
    """Query LLMs about a causal relationship between two variables.

    NODE_A and NODE_B are the variable names to query about.

    Examples:

        cqknow query smoking lung_cancer

        cqknow query smoking lung_cancer --domain medicine

        cqknow query X Y --model groq/llama-3.1-8b-instant \
                         --model gemini/gemini-2.5-flash
    """
    # Import here to avoid slow startup for --help
    from causaliq_knowledge.llm import LLMKnowledge

    # Build context
    context = None
    if domain:
        context = {"domain": domain}

    # Create provider
    try:
        provider = LLMKnowledge(
            models=list(model),
            consensus_strategy=strategy,
            temperature=llm_temperature,
        )
    except Exception as e:
        click.echo(f"Error creating provider: {e}", err=True)
        sys.exit(1)

    # Query
    click.echo(
        f"Querying {len(model)} model(s) about: {node_a} -> {node_b}",
        err=True,
    )

    try:
        result = provider.query_edge(node_a, node_b, context=context)
    except Exception as e:
        click.echo(f"Error querying LLM: {e}", err=True)
        sys.exit(1)

    # Output
    if output_json:
        output = {
            "node_a": node_a,
            "node_b": node_b,
            "exists": result.exists,
            "direction": result.direction.value if result.direction else None,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
            "model": result.model,
        }
        click.echo(json.dumps(output, indent=2))
    else:
        # Human-readable output
        exists_map = {True: "Yes", False: "No", None: "Uncertain"}
        exists_str = exists_map[result.exists]
        direction_str = result.direction.value if result.direction else "N/A"

        click.echo(f"\n{'='*60}")
        click.echo(f"Query: Does '{node_a}' cause '{node_b}'?")
        click.echo("=" * 60)
        click.echo(f"Exists:     {exists_str}")
        click.echo(f"Direction:  {direction_str}")
        click.echo(f"Confidence: {result.confidence:.2f}")
        click.echo(f"Model(s):   {result.model or 'unknown'}")
        click.echo(f"{'='*60}")
        click.echo(f"Reasoning:  {result.reasoning}")
        click.echo()

    # Show stats
    stats = provider.get_stats()
    if stats["total_cost"] > 0:
        click.echo(
            f"Cost: ${stats['total_cost']:.6f} "
            f"({stats['total_calls']} call(s))",
            err=True,
        )


# Import and register command groups
from causaliq_knowledge.cli.cache import cache_group  # noqa: E402
from causaliq_knowledge.cli.generate import generate_group  # noqa: E402
from causaliq_knowledge.cli.models import list_models  # noqa: E402

cli.add_command(cache_group)
cli.add_command(generate_group)
cli.add_command(list_models)


def main() -> None:
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":  # pragma: no cover
    main()
