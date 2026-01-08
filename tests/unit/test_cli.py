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
def test_cli_models(monkeypatch):
    # Ensure no API keys are set so we test the unavailable path
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    # Mock httpx to avoid real network calls
    import httpx

    def mock_client(*args, **kwargs):
        class MockClient:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def get(self, *args, **kwargs):
                raise httpx.ConnectError("Mocked connection error")

        return MockClient()

    monkeypatch.setattr("httpx.Client", mock_client)

    runner = CliRunner()
    result = runner.invoke(cli, ["models"])

    assert result.exit_code == 0
    assert "Groq" in result.output
    assert "Gemini" in result.output
    assert "Ollama" in result.output
    assert "GROQ_API_KEY" in result.output
    assert "GEMINI_API_KEY" in result.output
    assert "Ollama server" in result.output
    # When API keys not set, should show "not set" message
    assert "not set" in result.output


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


# Test models command with Groq available but is_available returns False
def test_cli_models_groq_not_available(monkeypatch):
    # Mock GroqConfig to not throw (so we can test is_available path)
    class MockGroqConfig:
        pass

    # Mock Groq to be configured but is_available returns False
    class MockGroqClient:
        def __init__(self, config):
            pass

        def is_available(self):
            return False

    # Mock Gemini to throw ValueError
    class MockGeminiConfig:
        def __init__(self):
            raise ValueError("GEMINI_API_KEY not set")

    # Mock Ollama to throw ValueError
    class MockOllamaClient:
        def __init__(self, config):
            pass

        def list_models(self):
            raise ValueError("Ollama not running")

    # Patch at both source and __init__ levels for full coverage
    monkeypatch.setattr(
        "causaliq_knowledge.llm.groq_client.GroqConfig", MockGroqConfig
    )
    monkeypatch.setattr("causaliq_knowledge.llm.GroqConfig", MockGroqConfig)
    monkeypatch.setattr(
        "causaliq_knowledge.llm.groq_client.GroqClient", MockGroqClient
    )
    monkeypatch.setattr("causaliq_knowledge.llm.GroqClient", MockGroqClient)
    monkeypatch.setattr(
        "causaliq_knowledge.llm.gemini_client.GeminiConfig", MockGeminiConfig
    )
    monkeypatch.setattr(
        "causaliq_knowledge.llm.GeminiConfig", MockGeminiConfig
    )
    monkeypatch.setattr(
        "causaliq_knowledge.llm.ollama_client.OllamaClient", MockOllamaClient
    )
    monkeypatch.setattr(
        "causaliq_knowledge.llm.OllamaClient", MockOllamaClient
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["models"])

    assert result.exit_code == 0
    assert "GROQ_API_KEY not set" in result.output


# Test models command with Gemini available but is_available returns False
def test_cli_models_gemini_not_available(monkeypatch):
    # Mock Groq to throw ValueError
    class MockGroqConfig:
        def __init__(self):
            raise ValueError("GROQ_API_KEY not set")

    # Mock GeminiConfig to not throw (so we can test is_available path)
    class MockGeminiConfig:
        pass

    # Mock Gemini to be configured but is_available returns False
    class MockGeminiClient:
        def __init__(self, config):
            pass

        def is_available(self):
            return False

    # Mock Ollama to throw ValueError
    class MockOllamaClient:
        def __init__(self, config):
            pass

        def list_models(self):
            raise ValueError("Ollama not running")

    # Patch at both source and __init__ levels for full coverage
    monkeypatch.setattr(
        "causaliq_knowledge.llm.groq_client.GroqConfig", MockGroqConfig
    )
    monkeypatch.setattr("causaliq_knowledge.llm.GroqConfig", MockGroqConfig)
    monkeypatch.setattr(
        "causaliq_knowledge.llm.gemini_client.GeminiConfig", MockGeminiConfig
    )
    monkeypatch.setattr(
        "causaliq_knowledge.llm.GeminiConfig", MockGeminiConfig
    )
    monkeypatch.setattr(
        "causaliq_knowledge.llm.gemini_client.GeminiClient", MockGeminiClient
    )
    monkeypatch.setattr(
        "causaliq_knowledge.llm.GeminiClient", MockGeminiClient
    )
    monkeypatch.setattr(
        "causaliq_knowledge.llm.ollama_client.OllamaClient", MockOllamaClient
    )
    monkeypatch.setattr(
        "causaliq_knowledge.llm.OllamaClient", MockOllamaClient
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["models"])

    assert result.exit_code == 0
    assert "GEMINI_API_KEY not set" in result.output


# Test models command with Ollama available but no models installed
def test_cli_models_ollama_no_models(monkeypatch):
    # Mock Groq to throw ValueError
    class MockGroqConfig:
        def __init__(self):
            raise ValueError("GROQ_API_KEY not set")

    # Mock Gemini to throw ValueError
    class MockGeminiConfig:
        def __init__(self):
            raise ValueError("GEMINI_API_KEY not set")

    # Mock Ollama to return empty list
    class MockOllamaClient:
        def __init__(self, config):
            pass

        def list_models(self):
            return []  # No models installed

    # Patch at both source and __init__ levels for full coverage
    monkeypatch.setattr(
        "causaliq_knowledge.llm.groq_client.GroqConfig", MockGroqConfig
    )
    monkeypatch.setattr("causaliq_knowledge.llm.GroqConfig", MockGroqConfig)
    monkeypatch.setattr(
        "causaliq_knowledge.llm.gemini_client.GeminiConfig", MockGeminiConfig
    )
    monkeypatch.setattr(
        "causaliq_knowledge.llm.GeminiConfig", MockGeminiConfig
    )
    monkeypatch.setattr(
        "causaliq_knowledge.llm.ollama_client.OllamaClient", MockOllamaClient
    )
    monkeypatch.setattr(
        "causaliq_knowledge.llm.OllamaClient", MockOllamaClient
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["models"])

    assert result.exit_code == 0
    assert "No models installed" in result.output
    assert "ollama pull" in result.output


# Test models command with successful provider (Groq available with models)
def test_cli_models_groq_success(monkeypatch):
    # Mock GroqConfig to not throw
    class MockGroqConfig:
        pass

    # Mock Groq to return models successfully
    class MockGroqClient:
        def __init__(self, config):
            pass

        def is_available(self):
            return True

        def list_models(self):
            return ["llama-3.1-8b-instant", "mixtral-8x7b"]

    # Mock Gemini to throw ValueError
    class MockGeminiConfig:
        def __init__(self):
            raise ValueError("GEMINI_API_KEY not set")

    # Mock Ollama to throw ValueError
    class MockOllamaClient:
        def __init__(self, config):
            pass

        def list_models(self):
            raise ValueError("Ollama not running")

    # Patch at both source and __init__ levels for full coverage
    monkeypatch.setattr(
        "causaliq_knowledge.llm.groq_client.GroqConfig", MockGroqConfig
    )
    monkeypatch.setattr("causaliq_knowledge.llm.GroqConfig", MockGroqConfig)
    monkeypatch.setattr(
        "causaliq_knowledge.llm.groq_client.GroqClient", MockGroqClient
    )
    monkeypatch.setattr("causaliq_knowledge.llm.GroqClient", MockGroqClient)
    monkeypatch.setattr(
        "causaliq_knowledge.llm.gemini_client.GeminiConfig", MockGeminiConfig
    )
    monkeypatch.setattr(
        "causaliq_knowledge.llm.GeminiConfig", MockGeminiConfig
    )
    monkeypatch.setattr(
        "causaliq_knowledge.llm.ollama_client.OllamaClient", MockOllamaClient
    )
    monkeypatch.setattr(
        "causaliq_knowledge.llm.OllamaClient", MockOllamaClient
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["models"])

    assert result.exit_code == 0
    assert "groq/llama-3.1-8b-instant" in result.output
    assert "groq/mixtral-8x7b" in result.output
    assert "[OK]" in result.output
    assert "2 models" in result.output
    assert "Default model:" in result.output


# Test models command with Gemini available with models
def test_cli_models_gemini_success(monkeypatch):
    # Mock Groq to throw ValueError
    class MockGroqConfig:
        def __init__(self):
            raise ValueError("GROQ_API_KEY not set")

    # Mock GeminiConfig to not throw
    class MockGeminiConfig:
        pass

    # Mock Gemini to return models successfully
    class MockGeminiClient:
        def __init__(self, config):
            pass

        def is_available(self):
            return True

        def list_models(self):
            return ["gemini-2.5-flash", "gemini-2.0-pro"]

    # Mock Ollama to throw ValueError
    class MockOllamaClient:
        def __init__(self, config):
            pass

        def list_models(self):
            raise ValueError("Ollama not running")

    # Patch at both source and __init__ levels for full coverage
    monkeypatch.setattr(
        "causaliq_knowledge.llm.groq_client.GroqConfig", MockGroqConfig
    )
    monkeypatch.setattr("causaliq_knowledge.llm.GroqConfig", MockGroqConfig)
    monkeypatch.setattr(
        "causaliq_knowledge.llm.gemini_client.GeminiConfig", MockGeminiConfig
    )
    monkeypatch.setattr(
        "causaliq_knowledge.llm.GeminiConfig", MockGeminiConfig
    )
    monkeypatch.setattr(
        "causaliq_knowledge.llm.gemini_client.GeminiClient", MockGeminiClient
    )
    monkeypatch.setattr(
        "causaliq_knowledge.llm.GeminiClient", MockGeminiClient
    )
    monkeypatch.setattr(
        "causaliq_knowledge.llm.ollama_client.OllamaClient", MockOllamaClient
    )
    monkeypatch.setattr(
        "causaliq_knowledge.llm.OllamaClient", MockOllamaClient
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["models"])

    assert result.exit_code == 0
    assert "gemini/gemini-2.5-flash" in result.output
    assert "gemini/gemini-2.0-pro" in result.output
    assert "[OK]" in result.output


# Test models command with Ollama available with models
def test_cli_models_ollama_success(monkeypatch):
    # Mock Groq to throw ValueError
    class MockGroqConfig:
        def __init__(self):
            raise ValueError("GROQ_API_KEY not set")

    # Mock Gemini to throw ValueError
    class MockGeminiConfig:
        def __init__(self):
            raise ValueError("GEMINI_API_KEY not set")

    # Mock Ollama to return models successfully
    class MockOllamaClient:
        def __init__(self, config):
            pass

        def list_models(self):
            return ["llama3.2:1b", "mistral:7b"]

    # Patch at both source and __init__ levels for full coverage
    monkeypatch.setattr(
        "causaliq_knowledge.llm.groq_client.GroqConfig", MockGroqConfig
    )
    monkeypatch.setattr("causaliq_knowledge.llm.GroqConfig", MockGroqConfig)
    monkeypatch.setattr(
        "causaliq_knowledge.llm.gemini_client.GeminiConfig", MockGeminiConfig
    )
    monkeypatch.setattr(
        "causaliq_knowledge.llm.GeminiConfig", MockGeminiConfig
    )
    monkeypatch.setattr(
        "causaliq_knowledge.llm.ollama_client.OllamaClient", MockOllamaClient
    )
    monkeypatch.setattr(
        "causaliq_knowledge.llm.OllamaClient", MockOllamaClient
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["models"])

    assert result.exit_code == 0
    assert "ollama/llama3.2:1b" in result.output
    assert "ollama/mistral:7b" in result.output
    assert "[OK]" in result.output
