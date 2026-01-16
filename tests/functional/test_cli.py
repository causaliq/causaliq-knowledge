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
# Note: This test uses mocking to avoid slow network calls to providers.
def test_cli_models_lists_providers(monkeypatch):
    # Mock all the provider clients to avoid network calls
    # Patch environment to ensure no API keys are set
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)

    # Mock OllamaClient to avoid connection timeout to localhost
    from unittest.mock import MagicMock

    mock_ollama = MagicMock()
    mock_ollama.return_value.list_models.return_value = []
    monkeypatch.setattr("causaliq_knowledge.llm.OllamaClient", mock_ollama)

    runner = CliRunner()
    result = runner.invoke(cli, ["models"])

    assert result.exit_code == 0
    # Check that all provider names are listed (even without API keys)
    assert "Groq" in result.output
    assert "Gemini" in result.output
    assert "Ollama" in result.output


# Test version flag shows version.
def test_cli_version():
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])

    assert result.exit_code == 0
    assert "0.2.0" in result.output


# ============================================================================
# Cache CLI Tests
# ============================================================================


# Test cache command group shows help.
def test_cli_cache_shows_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["cache", "--help"])

    assert result.exit_code == 0
    assert "cache" in result.output
    assert "stats" in result.output


# Test cache command appears in main help.
def test_cli_main_help_shows_cache():
    runner = CliRunner()
    result = runner.invoke(cli, [])

    assert "cache" in result.output


# Test cache stats requires path argument.
def test_cli_cache_stats_requires_path():
    runner = CliRunner()
    result = runner.invoke(cli, ["cache", "stats"])

    assert result.exit_code != 0
    assert "Missing argument" in result.output


# Test cache stats shows entry and token counts.
def test_cli_cache_stats_shows_counts(tmp_path):
    from causaliq_knowledge.cache import TokenCache
    from causaliq_knowledge.cache.encoders import JsonEncoder

    # Create cache with some data
    cache_path = tmp_path / "test_cache.db"
    with TokenCache(str(cache_path)) as cache:
        cache.register_encoder("json", JsonEncoder())
        cache.put_data("hash1", "json", {"key": "value1"})
        cache.put_data("hash2", "json", {"key": "value2"})

    runner = CliRunner()
    result = runner.invoke(cli, ["cache", "stats", str(cache_path)])

    assert result.exit_code == 0
    assert "Entries:" in result.output
    assert "2" in result.output
    assert "Tokens:" in result.output


# Test cache stats JSON output.
def test_cli_cache_stats_json_output(tmp_path):
    import json

    from causaliq_knowledge.cache import TokenCache
    from causaliq_knowledge.cache.encoders import JsonEncoder

    cache_path = tmp_path / "test_cache.db"
    with TokenCache(str(cache_path)) as cache:
        cache.register_encoder("json", JsonEncoder())
        cache.put_data("hash1", "json", {"test": "data"})

    runner = CliRunner()
    result = runner.invoke(cli, ["cache", "stats", str(cache_path), "--json"])

    assert result.exit_code == 0
    output = json.loads(result.output)
    assert output["entry_count"] == 1
    assert "token_count" in output
    assert "cache_path" in output


# Test cache stats with non-existent file.
def test_cli_cache_stats_missing_file():
    runner = CliRunner()
    result = runner.invoke(cli, ["cache", "stats", "nonexistent.db"])

    assert result.exit_code != 0


# Test cache stats with corrupted/invalid database file.
def test_cli_cache_stats_invalid_db(tmp_path):
    # Create a file that exists but is not a valid SQLite database
    invalid_db = tmp_path / "invalid.db"
    invalid_db.write_text("this is not a valid sqlite database")

    runner = CliRunner()
    result = runner.invoke(cli, ["cache", "stats", str(invalid_db)])

    assert result.exit_code == 1
    assert "Error opening cache" in result.output


# ============================================================================
# Cache Export Tests
# ============================================================================


# Test cache export command appears in help.
def test_cli_cache_export_in_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["cache", "--help"])

    assert result.exit_code == 0
    assert "export" in result.output


# Test cache export creates files with human-readable names.
def test_cli_cache_export_creates_files(tmp_path):
    from causaliq_knowledge.cache import TokenCache
    from causaliq_knowledge.llm.cache import LLMCacheEntry, LLMEntryEncoder

    # Create cache with LLM data
    cache_path = tmp_path / "test_cache.db"
    with TokenCache(str(cache_path)) as cache:
        encoder = LLMEntryEncoder()
        cache.register_encoder("llm", encoder)
        entry = LLMCacheEntry.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "smoking and lung_cancer"}],
            content="Yes, there is a causal relationship.",
            provider="openai",
        )
        cache.put_data("abc123", "llm", entry.to_dict())

    # Export
    export_dir = tmp_path / "export"
    runner = CliRunner()
    result = runner.invoke(
        cli, ["cache", "export", str(cache_path), str(export_dir)]
    )

    assert result.exit_code == 0
    assert "Exported 1 entries" in result.output

    # Check file was created with human-readable name
    files = list(export_dir.glob("*.json"))
    assert len(files) == 1
    assert "gpt4" in files[0].name
    assert "smoking" in files[0].name
    assert "edge" in files[0].name


# Test cache export JSON output.
def test_cli_cache_export_json_output(tmp_path):
    import json

    from causaliq_knowledge.cache import TokenCache
    from causaliq_knowledge.llm.cache import LLMCacheEntry, LLMEntryEncoder

    cache_path = tmp_path / "test_cache.db"
    with TokenCache(str(cache_path)) as cache:
        encoder = LLMEntryEncoder()
        cache.register_encoder("llm", encoder)
        entry = LLMCacheEntry.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "X and Y"}],
            content="Response",
            provider="openai",
        )
        cache.put_data("hash1", "llm", entry.to_dict())

    export_dir = tmp_path / "export"
    runner = CliRunner()
    result = runner.invoke(
        cli, ["cache", "export", str(cache_path), str(export_dir), "--json"]
    )

    assert result.exit_code == 0
    output = json.loads(result.output)
    assert output["exported"] == 1
    assert output["entry_types"] == ["llm"]


# Test cache export with empty cache.
def test_cli_cache_export_empty_cache(tmp_path):
    from causaliq_knowledge.cache import TokenCache

    cache_path = tmp_path / "empty_cache.db"
    with TokenCache(str(cache_path)):
        pass  # Create empty cache

    export_dir = tmp_path / "export"
    runner = CliRunner()
    result = runner.invoke(
        cli, ["cache", "export", str(cache_path), str(export_dir)]
    )

    assert result.exit_code == 0
    assert "No entries to export" in result.output


# Test cache export empty cache with JSON output.
def test_cli_cache_export_empty_cache_json(tmp_path):
    import json

    from causaliq_knowledge.cache import TokenCache

    cache_path = tmp_path / "empty_cache.db"
    with TokenCache(str(cache_path)):
        pass  # Create empty cache

    export_dir = tmp_path / "export"
    runner = CliRunner()
    result = runner.invoke(
        cli, ["cache", "export", str(cache_path), str(export_dir), "--json"]
    )

    assert result.exit_code == 0
    output = json.loads(result.output)
    assert output["exported"] == 0


# Test cache export with non-LLM entry types uses generic export.
def test_cli_cache_export_non_llm_type(tmp_path):
    from causaliq_knowledge.cache import TokenCache
    from causaliq_knowledge.cache.encoders import JsonEncoder

    cache_path = tmp_path / "test_cache.db"
    with TokenCache(str(cache_path)) as cache:
        # Register a generic JSON encoder for a non-LLM type
        cache.register_encoder("json", JsonEncoder())
        cache.put_data("hash1", "json", {"key": "value"})

    export_dir = tmp_path / "export"
    runner = CliRunner()
    result = runner.invoke(
        cli, ["cache", "export", str(cache_path), str(export_dir)]
    )

    assert result.exit_code == 0
    assert "Exported 1 entries" in result.output
    # Check file was created
    files = list(export_dir.glob("*.json"))
    assert len(files) == 1


# Test cache export with invalid cache file.
def test_cli_cache_export_invalid_db(tmp_path):
    invalid_db = tmp_path / "invalid.db"
    invalid_db.write_text("not a database")

    export_dir = tmp_path / "export"
    runner = CliRunner()
    result = runner.invoke(
        cli, ["cache", "export", str(invalid_db), str(export_dir)]
    )

    assert result.exit_code == 1
    assert "Error exporting cache" in result.output


# ============================================================================
# Cache Export Zip Tests
# ============================================================================


# Test cache export to zip file.
def test_cli_cache_export_to_zip(tmp_path):
    import zipfile

    from causaliq_knowledge.cache import TokenCache
    from causaliq_knowledge.llm.cache import LLMCacheEntry, LLMEntryEncoder

    # Create cache with LLM data
    cache_path = tmp_path / "test_cache.db"
    with TokenCache(str(cache_path)) as cache:
        encoder = LLMEntryEncoder()
        cache.register_encoder("llm", encoder)
        entry = LLMCacheEntry.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "smoking and lung_cancer"}],
            content="Yes, there is a causal relationship.",
            provider="openai",
        )
        cache.put_data("abc123", "llm", entry.to_dict())

    # Export to zip
    zip_path = tmp_path / "export.zip"
    runner = CliRunner()
    result = runner.invoke(
        cli, ["cache", "export", str(cache_path), str(zip_path)]
    )

    assert result.exit_code == 0
    assert "zip archive" in result.output
    assert zip_path.exists()

    # Verify zip contents
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        assert len(names) == 1
        assert names[0].endswith(".json")
        assert "gpt4" in names[0]


# Test cache export to zip with JSON output.
def test_cli_cache_export_to_zip_json_output(tmp_path):
    import json

    from causaliq_knowledge.cache import TokenCache
    from causaliq_knowledge.llm.cache import LLMCacheEntry, LLMEntryEncoder

    cache_path = tmp_path / "test_cache.db"
    with TokenCache(str(cache_path)) as cache:
        encoder = LLMEntryEncoder()
        cache.register_encoder("llm", encoder)
        entry = LLMCacheEntry.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "X and Y"}],
            content="Response",
            provider="openai",
        )
        cache.put_data("hash1", "llm", entry.to_dict())

    zip_path = tmp_path / "export.zip"
    runner = CliRunner()
    result = runner.invoke(
        cli, ["cache", "export", str(cache_path), str(zip_path), "--json"]
    )

    assert result.exit_code == 0
    output = json.loads(result.output)
    assert output["format"] == "zip"
    assert output["exported"] == 1
    assert zip_path.exists()


# Test cache export to zip creates parent directories.
def test_cli_cache_export_to_zip_creates_parent_dirs(tmp_path):
    from causaliq_knowledge.cache import TokenCache
    from causaliq_knowledge.llm.cache import LLMCacheEntry, LLMEntryEncoder

    cache_path = tmp_path / "test_cache.db"
    with TokenCache(str(cache_path)) as cache:
        encoder = LLMEntryEncoder()
        cache.register_encoder("llm", encoder)
        entry = LLMCacheEntry.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "test"}],
            content="Response",
        )
        cache.put_data("h1", "llm", entry.to_dict())

    # Nested path that doesn't exist
    zip_path = tmp_path / "nested" / "dir" / "export.zip"
    runner = CliRunner()
    result = runner.invoke(
        cli, ["cache", "export", str(cache_path), str(zip_path)]
    )

    assert result.exit_code == 0
    assert zip_path.exists()
