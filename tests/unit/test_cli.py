"""Unit tests for CLI."""

from click.testing import CliRunner

from causaliq_knowledge.cli import cli


# Test compatibility shim module exports cli and main.
def test_cli_shim_exports():
    """Test that the cli.py shim module exports cli and main."""
    # The shim module at src/causaliq_knowledge/cli.py provides backward
    # compatibility. When Python sees `import causaliq_knowledge.cli`, it
    # resolves to the cli/ package, not cli.py. The shim is only used when
    # explicitly imported as a module file.
    #
    # We test the package exports which is what users actually import.
    from causaliq_knowledge.cli import cli, main

    assert callable(cli)
    assert callable(main)


# Test main function invokes cli.
def test_main_calls_cli(monkeypatch):
    """Test that main() calls the cli function."""
    import sys

    from causaliq_knowledge.cli import main

    # Mock sys.argv to provide --help flag so cli exits cleanly
    monkeypatch.setattr(sys, "argv", ["causaliq-knowledge", "--help"])

    # main() should call cli() which will show help and exit with code 0
    try:
        main()
    except SystemExit as e:
        assert e.code == 0


# Test CLI shows version.
def test_cli_version():
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])

    assert result.exit_code == 0
    assert "0.4.0" in result.output


# Test CLI shows help.
def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "CausalIQ Knowledge" in result.output
    assert "generate_graph" in result.output
    assert "list_models" in result.output
    assert "cache_stats" in result.output
    assert "export_cache" in result.output
    assert "import_cache" in result.output


# Test list_models command lists models.
def test_cli_list_models(monkeypatch):
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
    result = runner.invoke(cli, ["list_models"])

    assert result.exit_code == 0
    assert "Groq" in result.output
    assert "Gemini" in result.output
    assert "Ollama" in result.output
    assert "GROQ_API_KEY" in result.output
    assert "GEMINI_API_KEY" in result.output
    assert "Ollama server" in result.output
    # When API keys not set, should show "not set" message
    assert "not set" in result.output


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

    # Only need to mock Groq since we filter to just groq provider
    monkeypatch.setattr(
        "causaliq_knowledge.llm.groq_client.GroqConfig", MockGroqConfig
    )
    monkeypatch.setattr("causaliq_knowledge.llm.GroqConfig", MockGroqConfig)
    monkeypatch.setattr(
        "causaliq_knowledge.llm.groq_client.GroqClient", MockGroqClient
    )
    monkeypatch.setattr("causaliq_knowledge.llm.GroqClient", MockGroqClient)

    runner = CliRunner()
    result = runner.invoke(cli, ["list_models", "-p", "groq"])

    assert result.exit_code == 0
    assert "GROQ_API_KEY not set" in result.output


# Test models command with Gemini available but is_available returns False
def test_cli_models_gemini_not_available(monkeypatch):
    # Mock GeminiConfig to not throw (so we can test is_available path)
    class MockGeminiConfig:
        pass

    # Mock Gemini to be configured but is_available returns False
    class MockGeminiClient:
        def __init__(self, config):
            pass

        def is_available(self):
            return False

    # Only need to mock Gemini since we filter to just gemini provider
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

    runner = CliRunner()
    result = runner.invoke(cli, ["list_models", "-p", "gemini"])

    assert result.exit_code == 0
    assert "GEMINI_API_KEY not set" in result.output


# Test models command with Anthropic available but is_available returns False
def test_cli_models_anthropic_not_available(monkeypatch):
    # Mock AnthropicConfig to not throw (so we can test is_available path)
    class MockAnthropicConfig:
        pass

    # Mock Anthropic to be configured but is_available returns False
    class MockAnthropicClient:
        def __init__(self, config):
            pass

        def is_available(self):
            return False

    # Only need to mock Anthropic since we filter to just anthropic provider
    monkeypatch.setattr(
        "causaliq_knowledge.llm.anthropic_client.AnthropicConfig",
        MockAnthropicConfig,
    )
    monkeypatch.setattr(
        "causaliq_knowledge.llm.AnthropicConfig", MockAnthropicConfig
    )
    monkeypatch.setattr(
        "causaliq_knowledge.llm.anthropic_client.AnthropicClient",
        MockAnthropicClient,
    )
    monkeypatch.setattr(
        "causaliq_knowledge.llm.AnthropicClient", MockAnthropicClient
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["list_models", "-p", "anthropic"])

    assert result.exit_code == 0
    assert "ANTHROPIC_API_KEY not set" in result.output


# Test models command with Anthropic config raising ValueError
def test_cli_models_anthropic_config_error(monkeypatch):
    # Mock AnthropicConfig to throw ValueError (no API key set)
    class MockAnthropicConfig:
        def __init__(self):
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable is required"
            )

    # Only need to mock Anthropic since we filter to just anthropic provider
    monkeypatch.setattr(
        "causaliq_knowledge.llm.anthropic_client.AnthropicConfig",
        MockAnthropicConfig,
    )
    monkeypatch.setattr(
        "causaliq_knowledge.llm.AnthropicConfig", MockAnthropicConfig
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["list_models", "-p", "anthropic"])

    assert result.exit_code == 0
    assert (
        "ANTHROPIC_API_KEY environment variable is required" in result.output
    )


# Test models command with OpenAI available but is_available returns False
def test_cli_models_openai_not_available(monkeypatch):
    # Mock OpenAIConfig to not throw (so we can test is_available path)
    class MockOpenAIConfig:
        pass

    # Mock OpenAI to be configured but is_available returns False
    class MockOpenAIClient:
        def __init__(self, config):
            pass

        def is_available(self):
            return False

    # Only need to mock OpenAI since we filter to just openai provider
    monkeypatch.setattr(
        "causaliq_knowledge.llm.openai_client.OpenAIConfig", MockOpenAIConfig
    )
    monkeypatch.setattr(
        "causaliq_knowledge.llm.OpenAIConfig", MockOpenAIConfig
    )
    monkeypatch.setattr(
        "causaliq_knowledge.llm.openai_client.OpenAIClient", MockOpenAIClient
    )
    monkeypatch.setattr(
        "causaliq_knowledge.llm.OpenAIClient", MockOpenAIClient
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["list_models", "-p", "openai"])

    assert result.exit_code == 0
    assert "OPENAI_API_KEY not set" in result.output


# Test models command with Ollama available but no models installed
def test_cli_models_ollama_no_models(monkeypatch):
    # Mock Ollama to return empty list
    class MockOllamaClient:
        def __init__(self, config):
            pass

        def list_models(self):
            return []  # No models installed

    # Only need to mock Ollama since we filter to just ollama provider
    monkeypatch.setattr(
        "causaliq_knowledge.llm.ollama_client.OllamaClient", MockOllamaClient
    )
    monkeypatch.setattr(
        "causaliq_knowledge.llm.OllamaClient", MockOllamaClient
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["list_models", "-p", "ollama"])

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

    # Only need to mock Groq since we filter to just groq provider
    monkeypatch.setattr(
        "causaliq_knowledge.llm.groq_client.GroqConfig", MockGroqConfig
    )
    monkeypatch.setattr("causaliq_knowledge.llm.GroqConfig", MockGroqConfig)
    monkeypatch.setattr(
        "causaliq_knowledge.llm.groq_client.GroqClient", MockGroqClient
    )
    monkeypatch.setattr("causaliq_knowledge.llm.GroqClient", MockGroqClient)

    runner = CliRunner()
    result = runner.invoke(cli, ["list_models", "-p", "groq"])

    assert result.exit_code == 0
    assert "groq/llama-3.1-8b-instant" in result.output
    assert "groq/mixtral-8x7b" in result.output
    assert "[OK]" in result.output
    assert "2 models" in result.output
    assert "Default model:" in result.output


# Test models command with Gemini available with models
def test_cli_models_gemini_success(monkeypatch):
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

    # Only need to mock Gemini since we filter to just gemini provider
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

    runner = CliRunner()
    result = runner.invoke(cli, ["list_models", "-p", "gemini"])

    assert result.exit_code == 0
    assert "gemini/gemini-2.5-flash" in result.output
    assert "gemini/gemini-2.0-pro" in result.output
    assert "[OK]" in result.output


# Test models command with Ollama available with models
def test_cli_models_ollama_success(monkeypatch):
    # Mock Ollama to return models successfully
    class MockOllamaClient:
        def __init__(self, config):
            pass

        def list_models(self):
            return ["llama3.2:1b", "mistral:7b"]

    # Only need to mock Ollama since we filter to just ollama provider
    monkeypatch.setattr(
        "causaliq_knowledge.llm.ollama_client.OllamaClient", MockOllamaClient
    )
    monkeypatch.setattr(
        "causaliq_knowledge.llm.OllamaClient", MockOllamaClient
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["list_models", "-p", "ollama"])

    assert result.exit_code == 0
    assert "ollama/llama3.2:1b" in result.output
    assert "ollama/mistral:7b" in result.output
    assert "[OK]" in result.output


# Test models command with Anthropic available with models
def test_cli_models_anthropic_success(monkeypatch):
    # Mock AnthropicConfig to not throw
    class MockAnthropicConfig:
        pass

    # Mock Anthropic to return models successfully
    class MockAnthropicClient:
        def __init__(self, config):
            pass

        def is_available(self):
            return True

        def list_models(self):
            return ["claude-sonnet-4-20250514", "claude-3-5-sonnet-20241022"]

    # Only need to mock Anthropic since we filter to just anthropic provider
    monkeypatch.setattr(
        "causaliq_knowledge.llm.anthropic_client.AnthropicConfig",
        MockAnthropicConfig,
    )
    monkeypatch.setattr(
        "causaliq_knowledge.llm.AnthropicConfig", MockAnthropicConfig
    )
    monkeypatch.setattr(
        "causaliq_knowledge.llm.anthropic_client.AnthropicClient",
        MockAnthropicClient,
    )
    monkeypatch.setattr(
        "causaliq_knowledge.llm.AnthropicClient", MockAnthropicClient
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["list_models", "-p", "anthropic"])

    assert result.exit_code == 0
    assert "anthropic/claude-sonnet-4-20250514" in result.output
    assert "anthropic/claude-3-5-sonnet-20241022" in result.output
    assert "[OK]" in result.output
    assert "2 models" in result.output
    assert "Default model:" in result.output


# Test models command with OpenAI available with models
def test_cli_models_openai_success(monkeypatch):
    # Mock OpenAIConfig to not throw
    class MockOpenAIConfig:
        pass

    # Mock OpenAI to return models successfully
    class MockOpenAIClient:
        def __init__(self, config):
            pass

        def is_available(self):
            return True

        def list_models(self):
            return ["gpt-4o", "gpt-4o-mini"]

    # Only need to mock OpenAI since we filter to just openai provider
    monkeypatch.setattr(
        "causaliq_knowledge.llm.openai_client.OpenAIConfig",
        MockOpenAIConfig,
    )
    monkeypatch.setattr(
        "causaliq_knowledge.llm.OpenAIConfig", MockOpenAIConfig
    )
    monkeypatch.setattr(
        "causaliq_knowledge.llm.openai_client.OpenAIClient",
        MockOpenAIClient,
    )
    monkeypatch.setattr(
        "causaliq_knowledge.llm.OpenAIClient", MockOpenAIClient
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["list_models", "-p", "openai"])

    assert result.exit_code == 0
    assert "openai/gpt-4o" in result.output
    assert "openai/gpt-4o-mini" in result.output
    assert "[OK]" in result.output
    assert "2 models" in result.output
    assert "Default model:" in result.output


# Test list_models command with invalid provider name
def test_cli_list_models_invalid_provider():
    runner = CliRunner()
    result = runner.invoke(cli, ["list_models", "-p", "invalid_provider"])

    # Click validates the choice and returns exit code 2 for usage errors
    assert result.exit_code == 2
    assert "Invalid value" in result.output
    assert "invalid_provider" in result.output


# Test models command with DeepSeek config raising ValueError
def test_cli_models_deepseek_config_error(monkeypatch):
    # Mock DeepSeekConfig to throw ValueError (no API key set)
    class MockDeepSeekConfig:
        def __init__(self):
            raise ValueError(
                "DEEPSEEK_API_KEY environment variable is required"
            )

    # Mock both the direct module and the re-export in __init__.py
    monkeypatch.setattr(
        "causaliq_knowledge.llm.deepseek_client.DeepSeekConfig",
        MockDeepSeekConfig,
    )
    monkeypatch.setattr(
        "causaliq_knowledge.llm.DeepSeekConfig", MockDeepSeekConfig
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["list_models", "-p", "deepseek"])

    assert result.exit_code == 0
    assert "DEEPSEEK_API_KEY environment variable is required" in result.output


# Test models command with DeepSeek available but is_available returns False
def test_cli_models_deepseek_not_available(monkeypatch):
    class MockDeepSeekConfig:
        pass

    class MockDeepSeekClient:
        def __init__(self, config):
            pass

        def is_available(self):
            return False

    monkeypatch.setattr(
        "causaliq_knowledge.llm.deepseek_client.DeepSeekConfig",
        MockDeepSeekConfig,
    )
    monkeypatch.setattr(
        "causaliq_knowledge.llm.DeepSeekConfig", MockDeepSeekConfig
    )
    monkeypatch.setattr(
        "causaliq_knowledge.llm.deepseek_client.DeepSeekClient",
        MockDeepSeekClient,
    )
    monkeypatch.setattr(
        "causaliq_knowledge.llm.DeepSeekClient", MockDeepSeekClient
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["list_models", "-p", "deepseek"])

    assert result.exit_code == 0
    assert "DEEPSEEK_API_KEY not set" in result.output


# Test models command with DeepSeek success
def test_cli_models_deepseek_success(monkeypatch):
    class MockDeepSeekConfig:
        pass

    class MockDeepSeekClient:
        def __init__(self, config):
            pass

        def is_available(self):
            return True

        def list_models(self):
            return ["deepseek-chat", "deepseek-reasoner"]

    monkeypatch.setattr(
        "causaliq_knowledge.llm.deepseek_client.DeepSeekConfig",
        MockDeepSeekConfig,
    )
    monkeypatch.setattr(
        "causaliq_knowledge.llm.DeepSeekConfig", MockDeepSeekConfig
    )
    monkeypatch.setattr(
        "causaliq_knowledge.llm.deepseek_client.DeepSeekClient",
        MockDeepSeekClient,
    )
    monkeypatch.setattr(
        "causaliq_knowledge.llm.DeepSeekClient", MockDeepSeekClient
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["list_models", "-p", "deepseek"])

    assert result.exit_code == 0
    assert "deepseek/deepseek-chat" in result.output
    assert "deepseek/deepseek-reasoner" in result.output
    assert "[OK]" in result.output


# Test models command with Mistral config raising ValueError
def test_cli_models_mistral_config_error(monkeypatch):
    # Mock MistralConfig to throw ValueError (no API key set)
    class MockMistralConfig:
        def __init__(self):
            raise ValueError(
                "MISTRAL_API_KEY environment variable is required"
            )

    # Mock both the direct module and the re-export in __init__.py
    monkeypatch.setattr(
        "causaliq_knowledge.llm.mistral_client.MistralConfig",
        MockMistralConfig,
    )
    monkeypatch.setattr(
        "causaliq_knowledge.llm.MistralConfig", MockMistralConfig
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["list_models", "-p", "mistral"])

    assert result.exit_code == 0
    assert "MISTRAL_API_KEY environment variable is required" in result.output


# Test models command with Mistral available but is_available returns False
def test_cli_models_mistral_not_available(monkeypatch):
    class MockMistralConfig:
        pass

    class MockMistralClient:
        def __init__(self, config):
            pass

        def is_available(self):
            return False

    monkeypatch.setattr(
        "causaliq_knowledge.llm.mistral_client.MistralConfig",
        MockMistralConfig,
    )
    monkeypatch.setattr(
        "causaliq_knowledge.llm.MistralConfig", MockMistralConfig
    )
    monkeypatch.setattr(
        "causaliq_knowledge.llm.mistral_client.MistralClient",
        MockMistralClient,
    )
    monkeypatch.setattr(
        "causaliq_knowledge.llm.MistralClient", MockMistralClient
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["list_models", "-p", "mistral"])

    assert result.exit_code == 0
    assert "MISTRAL_API_KEY not set" in result.output


# Test models command with Mistral success
def test_cli_models_mistral_success(monkeypatch):
    class MockMistralConfig:
        pass

    class MockMistralClient:
        def __init__(self, config):
            pass

        def is_available(self):
            return True

        def list_models(self):
            return ["mistral-small-latest", "mistral-large-latest"]

    monkeypatch.setattr(
        "causaliq_knowledge.llm.mistral_client.MistralConfig",
        MockMistralConfig,
    )
    monkeypatch.setattr(
        "causaliq_knowledge.llm.MistralConfig", MockMistralConfig
    )
    monkeypatch.setattr(
        "causaliq_knowledge.llm.mistral_client.MistralClient",
        MockMistralClient,
    )
    monkeypatch.setattr(
        "causaliq_knowledge.llm.MistralClient", MockMistralClient
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["list_models", "-p", "mistral"])

    assert result.exit_code == 0
    assert "mistral/mistral-small-latest" in result.output
    assert "mistral/mistral-large-latest" in result.output
    assert "[OK]" in result.output


# =============================================================================
# Cache helper function tests
# =============================================================================


# Test _is_graph_entry returns False for non-dict input.
def test_is_graph_entry_non_dict() -> None:
    """Test _is_graph_entry returns False for non-dict input."""
    from causaliq_knowledge.cli.cache import _is_graph_entry

    assert _is_graph_entry("not a dict") is False
    assert _is_graph_entry(123) is False
    assert _is_graph_entry(None) is False
    assert _is_graph_entry(["a", "list"]) is False


# Test _is_graph_entry returns False for dict missing edges.
def test_is_graph_entry_missing_edges() -> None:
    """Test _is_graph_entry returns False when edges key is missing."""
    from causaliq_knowledge.cli.cache import _is_graph_entry

    data = {"variables": ["A", "B"]}
    assert _is_graph_entry(data) is False


# Test _is_graph_entry returns False when edges is not a list.
def test_is_graph_entry_edges_not_list() -> None:
    """Test _is_graph_entry returns False when edges is not a list."""
    from causaliq_knowledge.cli.cache import _is_graph_entry

    data = {"edges": "not a list", "variables": ["A", "B"]}
    assert _is_graph_entry(data) is False


# Test _is_graph_entry returns False for dict missing variables.
def test_is_graph_entry_missing_variables() -> None:
    """Test _is_graph_entry returns False when variables key is missing."""
    from causaliq_knowledge.cli.cache import _is_graph_entry

    data = {"edges": [{"source": "A", "target": "B"}]}
    assert _is_graph_entry(data) is False


# Test _is_graph_entry returns False when variables is not a list.
def test_is_graph_entry_variables_not_list() -> None:
    """Test _is_graph_entry returns False when variables is not a list."""
    from causaliq_knowledge.cli.cache import _is_graph_entry

    data = {"edges": [], "variables": "not a list"}
    assert _is_graph_entry(data) is False


# Test _is_graph_entry returns True for valid graph entry.
def test_is_graph_entry_valid() -> None:
    """Test _is_graph_entry returns True for valid graph data."""
    from causaliq_knowledge.cli.cache import _is_graph_entry

    data = {
        "edges": [{"source": "A", "target": "B"}],
        "variables": ["A", "B"],
    }
    assert _is_graph_entry(data) is True
