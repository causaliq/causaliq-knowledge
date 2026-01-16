"""Command-line interface for causaliq-knowledge."""

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
    "--temperature",
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
    temperature: float,
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
            temperature=temperature,
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


@cli.command("models")
@click.argument("provider", required=False, default=None)
def list_models(provider: Optional[str]) -> None:
    """List available LLM models from each provider.

    Queries each provider's API to show models accessible with your
    current configuration. Results are filtered by your API key's
    access level or locally installed models.

    Optionally specify PROVIDER to list models from a single provider:
    groq, anthropic, gemini, ollama, openai, deepseek, or mistral.

    Examples:

        cqknow models              # List all providers

        cqknow models groq         # List only Groq models

        cqknow models mistral      # List only Mistral models
    """
    from typing import Callable, List, Optional, Tuple, TypedDict

    from causaliq_knowledge.llm import (
        AnthropicClient,
        AnthropicConfig,
        DeepSeekClient,
        DeepSeekConfig,
        GeminiClient,
        GeminiConfig,
        GroqClient,
        GroqConfig,
        MistralClient,
        MistralConfig,
        OllamaClient,
        OllamaConfig,
        OpenAIClient,
        OpenAIConfig,
    )

    # Type for get_models functions
    GetModelsFunc = Callable[[], Tuple[bool, List[str], Optional[str]]]

    class ProviderInfo(TypedDict):
        name: str
        prefix: str
        env_var: Optional[str]
        url: str
        get_models: GetModelsFunc

    def get_groq_models() -> Tuple[bool, List[str], Optional[str]]:
        """Returns (available, models, error_msg)."""
        try:
            client = GroqClient(GroqConfig())
            if not client.is_available():
                return False, [], "GROQ_API_KEY not set"
            models = [f"groq/{m}" for m in client.list_models()]
            return True, models, None
        except ValueError as e:
            return False, [], str(e)

    def get_anthropic_models() -> Tuple[bool, List[str], Optional[str]]:
        """Returns (available, models, error_msg)."""
        try:
            client = AnthropicClient(AnthropicConfig())
            if not client.is_available():
                return False, [], "ANTHROPIC_API_KEY not set"
            models = [f"anthropic/{m}" for m in client.list_models()]
            return True, models, None
        except ValueError as e:
            return False, [], str(e)

    def get_gemini_models() -> Tuple[bool, List[str], Optional[str]]:
        """Returns (available, models, error_msg)."""
        try:
            client = GeminiClient(GeminiConfig())
            if not client.is_available():
                return False, [], "GEMINI_API_KEY not set"
            models = [f"gemini/{m}" for m in client.list_models()]
            return True, models, None
        except ValueError as e:
            return False, [], str(e)

    def get_ollama_models() -> Tuple[bool, List[str], Optional[str]]:
        """Returns (available, models, error_msg)."""
        try:
            client = OllamaClient(OllamaConfig())
            models = [f"ollama/{m}" for m in client.list_models()]
            if not models:
                msg = "No models installed. Run: ollama pull <model>"
                return True, [], msg
            return True, models, None
        except ValueError as e:
            return False, [], str(e)

    def get_openai_models() -> Tuple[bool, List[str], Optional[str]]:
        """Returns (available, models, error_msg)."""
        try:
            client = OpenAIClient(OpenAIConfig())
            if not client.is_available():
                return False, [], "OPENAI_API_KEY not set"
            models = [f"openai/{m}" for m in client.list_models()]
            return True, models, None
        except ValueError as e:
            return False, [], str(e)

    def get_deepseek_models() -> Tuple[bool, List[str], Optional[str]]:
        """Returns (available, models, error_msg)."""
        try:
            client = DeepSeekClient(DeepSeekConfig())
            if not client.is_available():
                return False, [], "DEEPSEEK_API_KEY not set"
            models = [f"deepseek/{m}" for m in client.list_models()]
            return True, models, None
        except ValueError as e:
            return False, [], str(e)

    def get_mistral_models() -> Tuple[bool, List[str], Optional[str]]:
        """Returns (available, models, error_msg)."""
        try:
            client = MistralClient(MistralConfig())
            if not client.is_available():
                return False, [], "MISTRAL_API_KEY not set"
            models = [f"mistral/{m}" for m in client.list_models()]
            return True, models, None
        except ValueError as e:
            return False, [], str(e)

    providers: List[ProviderInfo] = [
        {
            "name": "Groq",
            "prefix": "groq/",
            "env_var": "GROQ_API_KEY",
            "url": "https://console.groq.com",
            "get_models": get_groq_models,
        },
        {
            "name": "Anthropic",
            "prefix": "anthropic/",
            "env_var": "ANTHROPIC_API_KEY",
            "url": "https://console.anthropic.com",
            "get_models": get_anthropic_models,
        },
        {
            "name": "Gemini",
            "prefix": "gemini/",
            "env_var": "GEMINI_API_KEY",
            "url": "https://aistudio.google.com",
            "get_models": get_gemini_models,
        },
        {
            "name": "Ollama (Local)",
            "prefix": "ollama/",
            "env_var": None,
            "url": "https://ollama.ai",
            "get_models": get_ollama_models,
        },
        {
            "name": "OpenAI",
            "prefix": "openai/",
            "env_var": "OPENAI_API_KEY",
            "url": "https://platform.openai.com",
            "get_models": get_openai_models,
        },
        {
            "name": "DeepSeek",
            "prefix": "deepseek/",
            "env_var": "DEEPSEEK_API_KEY",
            "url": "https://platform.deepseek.com",
            "get_models": get_deepseek_models,
        },
        {
            "name": "Mistral",
            "prefix": "mistral/",
            "env_var": "MISTRAL_API_KEY",
            "url": "https://console.mistral.ai",
            "get_models": get_mistral_models,
        },
    ]

    # Filter providers if a specific one is requested
    valid_provider_names = [
        "groq",
        "anthropic",
        "gemini",
        "ollama",
        "openai",
        "deepseek",
        "mistral",
    ]
    if provider:
        provider_lower = provider.lower()
        if provider_lower not in valid_provider_names:
            click.echo(
                f"Unknown provider: {provider}. "
                f"Valid options: {', '.join(valid_provider_names)}",
                err=True,
            )
            sys.exit(1)
        providers = [
            p for p in providers if p["prefix"].rstrip("/") == provider_lower
        ]

    click.echo("\nAvailable LLM Models:\n")

    any_available = False
    for prov in providers:
        available, models, error = prov["get_models"]()

        if available and models:
            any_available = True
            status = click.style("[OK]", fg="green")
            count = len(models)
            click.echo(f"  {status} {prov['name']} ({count} models):")
            for m in models:
                click.echo(f"      {m}")
        elif available and not models:
            status = click.style("[!]", fg="yellow")
            click.echo(f"  {status} {prov['name']}:")
            click.echo(f"      {error}")
        else:
            status = click.style("[X]", fg="red")
            click.echo(f"  {status} {prov['name']}:")
            click.echo(f"      {error}")

        click.echo()

    click.echo("Provider Setup:")
    for prov in providers:
        available, _, _ = prov["get_models"]()
        if prov["env_var"]:
            status = "configured" if available else "not set"
            color = "green" if available else "yellow"
            click.echo(
                f"  {prov['env_var']}: "
                f"{click.style(status, fg=color)} - {prov['url']}"
            )
        else:
            status = "running" if available else "not running"
            color = "green" if available else "yellow"
            click.echo(
                f"  Ollama server: "
                f"{click.style(status, fg=color)} - {prov['url']}"
            )

    click.echo()
    click.echo(
        click.style("Note: ", fg="yellow")
        + "Some models may require a paid plan. "
        + "Free tier availability varies by provider."
    )
    click.echo()
    if any_available:
        click.echo("Default model: groq/llama-3.1-8b-instant")
    click.echo()


# ============================================================================
# Cache Commands
# ============================================================================


@cli.group("cache")
def cache_group() -> None:
    """Manage the LLM response cache.

    Commands for inspecting, exporting, and importing cached LLM responses.

    Examples:

        cqknow cache stats ./llm_cache.db

        cqknow cache export ./llm_cache.db ./export_dir

        cqknow cache import ./llm_cache.db ./import_dir
    """
    pass


@cache_group.command("stats")
@click.argument("cache_path", type=click.Path(exists=True))
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output result as JSON.",
)
def cache_stats(cache_path: str, output_json: bool) -> None:
    """Show cache statistics.

    CACHE_PATH is the path to the SQLite cache database.

    Examples:

        cqknow cache stats ./llm_cache.db

        cqknow cache stats ./llm_cache.db --json
    """
    from causaliq_knowledge.cache import TokenCache

    try:
        with TokenCache(cache_path) as cache:
            entry_count = cache.entry_count()
            token_count = cache.token_count()

            if output_json:
                output = {
                    "cache_path": cache_path,
                    "entry_count": entry_count,
                    "token_count": token_count,
                }
                click.echo(json.dumps(output, indent=2))
            else:
                click.echo(f"\nCache: {cache_path}")
                click.echo("=" * 40)
                click.echo(f"Entries:  {entry_count:,}")
                click.echo(f"Tokens:   {token_count:,}")
                click.echo()
    except Exception as e:
        click.echo(f"Error opening cache: {e}", err=True)
        sys.exit(1)


def main() -> None:
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":  # pragma: no cover
    main()
