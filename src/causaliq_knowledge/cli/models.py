"""Model listing CLI command.

This module provides the command for listing available LLM models
from each provider.
"""

from __future__ import annotations

import sys
from typing import Callable, List, Optional, Tuple, TypedDict

import click


class ProviderInfo(TypedDict):
    """Type definition for provider information."""

    name: str
    prefix: str
    env_var: Optional[str]
    url: str
    get_models: Callable[[], Tuple[bool, List[str], Optional[str]]]


def get_groq_models() -> Tuple[bool, List[str], Optional[str]]:
    """Get available Groq models.

    Returns:
        Tuple of (available, models, error_message).
    """
    from causaliq_knowledge.llm import GroqClient, GroqConfig

    try:
        client = GroqClient(GroqConfig())
        if not client.is_available():
            return False, [], "GROQ_API_KEY not set"
        models = [f"groq/{m}" for m in client.list_models()]
        return True, models, None
    except ValueError as e:
        return False, [], str(e)


def get_anthropic_models() -> Tuple[bool, List[str], Optional[str]]:
    """Get available Anthropic models.

    Returns:
        Tuple of (available, models, error_message).
    """
    from causaliq_knowledge.llm import AnthropicClient, AnthropicConfig

    try:
        client = AnthropicClient(AnthropicConfig())
        if not client.is_available():
            return False, [], "ANTHROPIC_API_KEY not set"
        models = [f"anthropic/{m}" for m in client.list_models()]
        return True, models, None
    except ValueError as e:
        return False, [], str(e)


def get_gemini_models() -> Tuple[bool, List[str], Optional[str]]:
    """Get available Gemini models.

    Returns:
        Tuple of (available, models, error_message).
    """
    from causaliq_knowledge.llm import GeminiClient, GeminiConfig

    try:
        client = GeminiClient(GeminiConfig())
        if not client.is_available():
            return False, [], "GEMINI_API_KEY not set"
        models = [f"gemini/{m}" for m in client.list_models()]
        return True, models, None
    except ValueError as e:
        return False, [], str(e)


def get_ollama_models() -> Tuple[bool, List[str], Optional[str]]:
    """Get available Ollama models.

    Returns:
        Tuple of (available, models, error_message).
    """
    from causaliq_knowledge.llm import OllamaClient, OllamaConfig

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
    """Get available OpenAI models.

    Returns:
        Tuple of (available, models, error_message).
    """
    from causaliq_knowledge.llm import OpenAIClient, OpenAIConfig

    try:
        client = OpenAIClient(OpenAIConfig())
        if not client.is_available():
            return False, [], "OPENAI_API_KEY not set"
        models = [f"openai/{m}" for m in client.list_models()]
        return True, models, None
    except ValueError as e:
        return False, [], str(e)


def get_deepseek_models() -> Tuple[bool, List[str], Optional[str]]:
    """Get available DeepSeek models.

    Returns:
        Tuple of (available, models, error_message).
    """
    from causaliq_knowledge.llm import DeepSeekClient, DeepSeekConfig

    try:
        client = DeepSeekClient(DeepSeekConfig())
        if not client.is_available():
            return False, [], "DEEPSEEK_API_KEY not set"
        models = [f"deepseek/{m}" for m in client.list_models()]
        return True, models, None
    except ValueError as e:
        return False, [], str(e)


def get_mistral_models() -> Tuple[bool, List[str], Optional[str]]:
    """Get available Mistral models.

    Returns:
        Tuple of (available, models, error_message).
    """
    from causaliq_knowledge.llm import MistralClient, MistralConfig

    try:
        client = MistralClient(MistralConfig())
        if not client.is_available():
            return False, [], "MISTRAL_API_KEY not set"
        models = [f"mistral/{m}" for m in client.list_models()]
        return True, models, None
    except ValueError as e:
        return False, [], str(e)


def get_all_providers() -> List[ProviderInfo]:
    """Get list of all provider configurations.

    Returns:
        List of ProviderInfo dictionaries.
    """
    return [
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


VALID_PROVIDER_NAMES = [
    "groq",
    "anthropic",
    "gemini",
    "ollama",
    "openai",
    "deepseek",
    "mistral",
]


@click.command("models")
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
    providers = get_all_providers()

    # Filter providers if a specific one is requested
    if provider:
        provider_lower = provider.lower()
        if provider_lower not in VALID_PROVIDER_NAMES:
            click.echo(
                f"Unknown provider: {provider}. "
                f"Valid options: {', '.join(VALID_PROVIDER_NAMES)}",
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
