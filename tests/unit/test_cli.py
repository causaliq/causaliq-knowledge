"""Unit tests for CLI."""

from click.testing import CliRunner

from causaliq_knowledge.cli import cli, main


# Test main function calls cli.
def test_main_calls_cli(monkeypatch):
    called = []

    def mock_cli():
        called.append(True)

    monkeypatch.setattr("causaliq_knowledge.cli.cli", mock_cli)
    main()

    assert len(called) == 1


# Test CLI shows version.
def test_cli_version():
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])

    assert result.exit_code == 0
    assert "0.1.0" in result.output


# Test CLI shows help.
def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "CausalIQ Knowledge" in result.output
    assert "query" in result.output
    assert "models" in result.output


# Test query command shows help.
def test_cli_query_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["query", "--help"])

    assert result.exit_code == 0
    assert "NODE_A" in result.output
    assert "NODE_B" in result.output
    assert "--model" in result.output
    assert "--domain" in result.output
    assert "--strategy" in result.output


# Test models command lists models.
def test_cli_models():
    runner = CliRunner()
    result = runner.invoke(cli, ["models"])

    assert result.exit_code == 0
    assert "Groq" in result.output
    assert "groq/llama-3.1-8b-instant" in result.output
    assert "Gemini" in result.output
    assert "gemini/gemini-2.5-flash" in result.output
    assert "GROQ_API_KEY" in result.output
    assert "GEMINI_API_KEY" in result.output


# Test query command with mocked provider.
def test_cli_query_success(monkeypatch):
    from causaliq_knowledge.models import EdgeDirection, EdgeKnowledge

    # Mock LLMKnowledge
    class MockProvider:
        def __init__(self, **kwargs):
            pass

        def query_edge(self, node_a, node_b, context=None):
            return EdgeKnowledge(
                exists=True,
                direction=EdgeDirection.A_TO_B,
                confidence=0.9,
                reasoning="Test reasoning",
                model="mock-model",
            )

        def get_stats(self):
            return {"total_cost": 0.001, "total_calls": 1}

    monkeypatch.setattr(
        "causaliq_knowledge.llm.LLMKnowledge",
        MockProvider,
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["query", "X", "Y"])

    assert result.exit_code == 0
    assert "Yes" in result.output
    assert "a_to_b" in result.output
    assert "0.90" in result.output
    assert "Test reasoning" in result.output


# Test query command with JSON output.
def test_cli_query_json_output(monkeypatch):
    import json

    from causaliq_knowledge.models import EdgeDirection, EdgeKnowledge

    class MockProvider:
        def __init__(self, **kwargs):
            pass

        def query_edge(self, node_a, node_b, context=None):
            return EdgeKnowledge(
                exists=True,
                direction=EdgeDirection.A_TO_B,
                confidence=0.85,
                reasoning="JSON test",
                model="mock",
            )

        def get_stats(self):
            return {"total_cost": 0.0, "total_calls": 1}

    monkeypatch.setattr(
        "causaliq_knowledge.llm.LLMKnowledge",
        MockProvider,
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["query", "A", "B", "--json"])

    assert result.exit_code == 0
    # Extract JSON from output (status message goes to stderr, JSON to stdout)
    # Find the JSON block in output
    output_lines = result.output.strip().split("\n")
    json_start = next(
        i for i, line in enumerate(output_lines) if line.startswith("{")
    )
    json_str = "\n".join(output_lines[json_start:])
    data = json.loads(json_str)
    assert data["exists"] is True
    assert data["direction"] == "a_to_b"
    assert data["confidence"] == 0.85


# Test query command with domain option.
def test_cli_query_with_domain(monkeypatch):
    from causaliq_knowledge.models import EdgeKnowledge

    captured_context = {}

    class MockProvider:
        def __init__(self, **kwargs):
            pass

        def query_edge(self, node_a, node_b, context=None):
            captured_context["context"] = context
            return EdgeKnowledge(
                exists=None,
                confidence=0.5,
                reasoning="Domain test",
                model="mock",
            )

        def get_stats(self):
            return {"total_cost": 0.0, "total_calls": 1}

    monkeypatch.setattr(
        "causaliq_knowledge.llm.LLMKnowledge",
        MockProvider,
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["query", "X", "Y", "--domain", "medicine"])

    assert result.exit_code == 0
    assert captured_context["context"] == {"domain": "medicine"}


# Test query command passes model options.
def test_cli_query_with_models(monkeypatch):
    from causaliq_knowledge.models import EdgeKnowledge

    captured_kwargs = {}

    class MockProvider:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

        def query_edge(self, node_a, node_b, context=None):
            return EdgeKnowledge(
                exists=True,
                confidence=0.8,
                reasoning="Multi-model test",
                model="m1, m2",
            )

        def get_stats(self):
            return {"total_cost": 0.0, "total_calls": 2}

    monkeypatch.setattr(
        "causaliq_knowledge.llm.LLMKnowledge",
        MockProvider,
    )

    runner = CliRunner()
    result = runner.invoke(
        cli, ["query", "X", "Y", "-m", "model1", "-m", "model2"]
    )

    assert result.exit_code == 0
    assert captured_kwargs["models"] == ["model1", "model2"]


# Test query command handles provider error.
def test_cli_query_provider_error(monkeypatch):
    def mock_init(**kwargs):
        raise ValueError("Invalid model")

    monkeypatch.setattr(
        "causaliq_knowledge.llm.LLMKnowledge",
        mock_init,
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["query", "X", "Y"])

    assert result.exit_code == 1
    assert "Error creating provider" in result.output


# Test query command handles query error.
def test_cli_query_llm_error(monkeypatch):
    class MockProvider:
        def __init__(self, **kwargs):
            pass

        def query_edge(self, node_a, node_b, context=None):
            raise RuntimeError("API Error")

        def get_stats(self):
            return {"total_cost": 0.0, "total_calls": 0}

    monkeypatch.setattr(
        "causaliq_knowledge.llm.LLMKnowledge",
        MockProvider,
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["query", "X", "Y"])

    assert result.exit_code == 1
    assert "Error querying LLM" in result.output


# Test query command with no direction (exists=False).
def test_cli_query_no_edge(monkeypatch):
    from causaliq_knowledge.models import EdgeKnowledge

    class MockProvider:
        def __init__(self, **kwargs):
            pass

        def query_edge(self, node_a, node_b, context=None):
            return EdgeKnowledge(
                exists=False,
                direction=None,
                confidence=0.95,
                reasoning="No causal link",
                model="mock",
            )

        def get_stats(self):
            return {"total_cost": 0.0, "total_calls": 1}

    monkeypatch.setattr(
        "causaliq_knowledge.llm.LLMKnowledge",
        MockProvider,
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["query", "X", "Y"])

    assert result.exit_code == 0
    assert "No" in result.output
    assert "N/A" in result.output
