"""
Functional tests for the CLI.

These tests use Click's CliRunner to invoke the CLI commands

monkeypatch only works on curent process, so CLI runner must be invoked
using standalone=False
"""

from click.testing import CliRunner
from pytest import fixture

from causaliq_knowledge.cli import cli

CLI_BASE_DIR = "tests/data/functional/cli"


# Provide a CLI runner for testing
@fixture
def cli_runner():
    return CliRunner()


# Test missing required NAME argument
def test_cli_missing_name_argument():
    runner = CliRunner()
    result = runner.invoke(cli, [])
    assert result.exit_code != 0  # Should fail
    assert "Missing argument" in result.output or "Usage:" in result.output


# Test help is shown when no arguments provided
def test_cli_no_args_shows_usage():
    runner = CliRunner()
    result = runner.invoke(cli, [])
    assert result.exit_code != 0
    assert "NAME" in result.output  # Should show usage info


# Test with NAME argument only
def test_cli_with_name_only():
    runner = CliRunner()
    result = runner.invoke(cli, ["Alice"])
    assert result.exit_code == 0
    assert "Hello, Alice!" in result.output


# Test with NAME and --greet option
def test_cli_with_name_and_greet():
    runner = CliRunner()
    result = runner.invoke(cli, ["--greet", "Hi", "Bob"])
    assert result.exit_code == 0
    assert "Hi, Bob!" in result.output


# Test that invoking script directly will run the CLI
def test_main_function(monkeypatch):
    called = {}

    def fake_cli(*args, **kwargs):
        called["cli"] = args != kwargs

    monkeypatch.setattr("causaliq_knowledge.cli.cli", fake_cli)
    from causaliq_knowledge.cli import main

    main()
    assert called.get("cli") is True
