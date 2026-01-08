"""
Functional tests for the CLI.

These tests use Click's CliRunner to invoke the CLI commands.
"""

from click.testing import CliRunner

from causaliq_knowledge.cli import cli


# Test no args shows usage with available commands.
def test_cli_no_args_shows_usage():
    runner = CliRunner()
    result = runner.invoke(cli, [])

    # Click shows usage info when no command provided
    assert "Usage:" in result.output
    assert "query" in result.output
    assert "models" in result.output


# Test query command requires node arguments.
def test_cli_query_requires_nodes():
    runner = CliRunner()
    result = runner.invoke(cli, ["query"])

    assert result.exit_code != 0
    assert "Missing argument" in result.output


# Test models command lists supported LLMs.
def test_cli_models_lists_providers():
    runner = CliRunner()
    result = runner.invoke(cli, ["models"])

    assert result.exit_code == 0
    # Check that all provider names are listed
    assert "Groq" in result.output
    assert "Gemini" in result.output
    assert "Ollama" in result.output
    # Note: Without API keys set, models won't be listed
    # Just verify the command runs and shows provider sections


# Test version flag shows version.
def test_cli_version():
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])

    assert result.exit_code == 0
    assert "0.1.0" in result.output
