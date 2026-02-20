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
    assert "list_models" in result.output
    assert "generate_graph" in result.output


# Test list_models command lists supported LLMs.
# Note: This test uses mocking to avoid slow network calls to providers.
def test_cli_list_models_lists_providers(monkeypatch):
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
    result = runner.invoke(cli, ["list_models"])

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
    assert "0.5.0" in result.output


# ============================================================================
# Cache CLI Tests
# ============================================================================


# Test cache_stats command appears in main help.
def test_cli_main_help_shows_cache_commands():
    runner = CliRunner()
    result = runner.invoke(cli, [])

    assert "cache_stats" in result.output
    assert "export_cache" in result.output
    assert "import_cache" in result.output


# Test cache_stats requires cache argument.
def test_cli_cache_stats_requires_cache():
    runner = CliRunner()
    result = runner.invoke(cli, ["cache_stats"])

    assert result.exit_code != 0
    assert "Missing option" in result.output


# Test cache stats shows entry and token counts.
def test_cli_cache_stats_shows_counts(tmp_path):
    from causaliq_core.cache import TokenCache
    from causaliq_core.cache.compressors import JsonCompressor

    # Create cache with some data
    cache_path = tmp_path / "test_cache.db"
    with TokenCache(str(cache_path)) as cache:
        cache.set_compressor(JsonCompressor())
        cache.put_data("hash1", {"key": "value1"})
        cache.put_data("hash2", {"key": "value2"})

    runner = CliRunner()
    result = runner.invoke(cli, ["cache_stats", "-c", str(cache_path)])

    assert result.exit_code == 0
    assert "Entries:" in result.output
    assert "2" in result.output
    assert "Token dictionary:" in result.output


# Test cache stats JSON output.
def test_cli_cache_stats_json_output(tmp_path):
    import json

    from causaliq_core.cache import TokenCache
    from causaliq_core.cache.compressors import JsonCompressor

    cache_path = tmp_path / "test_cache.db"
    with TokenCache(str(cache_path)) as cache:
        cache.set_compressor(JsonCompressor())
        cache.put_data("hash1", {"test": "data"})

    runner = CliRunner()
    result = runner.invoke(
        cli, ["cache_stats", "-c", str(cache_path), "--json"]
    )

    assert result.exit_code == 0
    output = json.loads(result.output)
    assert output["summary"]["entry_count"] == 1
    assert "token_count" in output["summary"]
    assert "cache_path" in output


# Test cache_stats with non-existent file.
def test_cli_cache_stats_missing_file():
    runner = CliRunner()
    result = runner.invoke(cli, ["cache_stats", "-c", "nonexistent.db"])

    assert result.exit_code != 0


# Test cache_stats with corrupted/invalid database file.
def test_cli_cache_stats_invalid_db(tmp_path):
    # Create a file that exists but is not a valid SQLite database
    invalid_db = tmp_path / "invalid.db"
    invalid_db.write_text("this is not a valid sqlite database")

    runner = CliRunner()
    result = runner.invoke(cli, ["cache_stats", "-c", str(invalid_db)])

    assert result.exit_code == 1
    assert "Error opening cache" in result.output


# ============================================================================
# Cache Export Tests
# ============================================================================


# Test export_cache command help.
def test_cli_export_cache_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["export_cache", "--help"])

    assert result.exit_code == 0
    assert "export" in result.output.lower()


# Test cache export creates files with human-readable names.
def test_cli_cache_export_creates_files(tmp_path):
    from causaliq_core.cache import TokenCache

    from causaliq_knowledge.llm.cache import LLMCacheEntry, LLMCompressor

    # Create cache with LLM data
    cache_path = tmp_path / "test_cache.db"
    with TokenCache(str(cache_path)) as cache:
        compressor = LLMCompressor()
        cache.set_compressor(compressor)
        entry = LLMCacheEntry.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "smoking and lung_cancer"}],
            content="Yes, there is a causal relationship.",
            provider="openai",
            request_id="test_export",
        )
        cache.put_data("abc123", entry.to_dict())

    # Export
    export_dir = tmp_path / "export"
    runner = CliRunner()
    result = runner.invoke(
        cli, ["export_cache", "-c", str(cache_path), "-o", str(export_dir)]
    )

    assert result.exit_code == 0
    assert "Exported 1 entries" in result.output

    # Check file was created with new format: id_timestamp_provider.json
    files = list(export_dir.glob("*.json"))
    assert len(files) == 1
    # Filename should start with request_id and end with provider
    assert files[0].name.startswith("test_export_")
    assert "_openai.json" in files[0].name


# Test cache export JSON output.
def test_cli_cache_export_json_output(tmp_path):
    import json

    from causaliq_core.cache import TokenCache

    from causaliq_knowledge.llm.cache import LLMCacheEntry, LLMCompressor

    cache_path = tmp_path / "test_cache.db"
    with TokenCache(str(cache_path)) as cache:
        compressor = LLMCompressor()
        cache.set_compressor(compressor)
        entry = LLMCacheEntry.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "X and Y"}],
            content="Response",
            provider="openai",
        )
        cache.put_data("hash1", entry.to_dict())

    export_dir = tmp_path / "export"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "export_cache",
            "-c",
            str(cache_path),
            "-o",
            str(export_dir),
            "--json",
        ],
    )

    assert result.exit_code == 0
    output = json.loads(result.output)
    assert output["exported"] == 1


# Test cache export with empty cache.
def test_cli_cache_export_empty_cache(tmp_path):
    from causaliq_core.cache import TokenCache

    cache_path = tmp_path / "empty_cache.db"
    with TokenCache(str(cache_path)):
        pass  # Create empty cache

    export_dir = tmp_path / "export"
    runner = CliRunner()
    result = runner.invoke(
        cli, ["export_cache", "-c", str(cache_path), "-o", str(export_dir)]
    )

    assert result.exit_code == 0
    assert "No entries to export" in result.output


# Test cache export empty cache with JSON output.
def test_cli_cache_export_empty_cache_json(tmp_path):
    import json

    from causaliq_core.cache import TokenCache

    cache_path = tmp_path / "empty_cache.db"
    with TokenCache(str(cache_path)):
        pass  # Create empty cache

    export_dir = tmp_path / "export"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "export_cache",
            "-c",
            str(cache_path),
            "-o",
            str(export_dir),
            "--json",
        ],
    )

    assert result.exit_code == 0
    output = json.loads(result.output)
    assert output["exported"] == 0


# Test cache export with non-LLM entry types uses generic export.
def test_cli_cache_export_non_llm_type(tmp_path):
    from causaliq_core.cache import TokenCache
    from causaliq_core.cache.compressors import JsonCompressor

    cache_path = tmp_path / "test_cache.db"
    with TokenCache(str(cache_path)) as cache:
        # Use JsonCompressor for generic JSON data
        cache.set_compressor(JsonCompressor())
        cache.put_data("hash1", {"key": "value"})

    export_dir = tmp_path / "export"
    runner = CliRunner()
    result = runner.invoke(
        cli, ["export_cache", "-c", str(cache_path), "-o", str(export_dir)]
    )

    assert result.exit_code == 0
    assert "Exported 1 entries" in result.output
    # Check file was created
    files = list(export_dir.glob("*.json"))
    assert len(files) == 1


# Test export_cache with invalid cache file.
def test_cli_export_cache_invalid_db(tmp_path):
    invalid_db = tmp_path / "invalid.db"
    invalid_db.write_text("not a database")

    export_dir = tmp_path / "export"
    runner = CliRunner()
    result = runner.invoke(
        cli, ["export_cache", "-c", str(invalid_db), "-o", str(export_dir)]
    )

    assert result.exit_code == 1
    assert "Error exporting cache" in result.output


# ============================================================================
# Cache Export Zip Tests
# ============================================================================


# Test cache export to zip file.
def test_cli_cache_export_to_zip(tmp_path):
    import zipfile

    from causaliq_core.cache import TokenCache

    from causaliq_knowledge.llm.cache import LLMCacheEntry, LLMCompressor

    # Create cache with LLM data
    cache_path = tmp_path / "test_cache.db"
    with TokenCache(str(cache_path)) as cache:
        compressor = LLMCompressor()
        cache.set_compressor(compressor)
        entry = LLMCacheEntry.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "smoking and lung_cancer"}],
            content="Yes, there is a causal relationship.",
            provider="openai",
            request_id="zip_test",
        )
        cache.put_data("abc123", entry.to_dict())

    # Export to zip
    zip_path = tmp_path / "export.zip"
    runner = CliRunner()
    result = runner.invoke(
        cli, ["export_cache", "-c", str(cache_path), "-o", str(zip_path)]
    )

    assert result.exit_code == 0
    assert "zip archive" in result.output
    assert zip_path.exists()

    # Verify zip contents - new format: id_timestamp_provider.json
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        assert len(names) == 1
        assert names[0].endswith(".json")
        assert names[0].startswith("zip_test_")
        assert "_openai.json" in names[0]


# Test cache export to zip with JSON output.
def test_cli_cache_export_to_zip_json_output(tmp_path):
    import json

    from causaliq_core.cache import TokenCache

    from causaliq_knowledge.llm.cache import LLMCacheEntry, LLMCompressor

    cache_path = tmp_path / "test_cache.db"
    with TokenCache(str(cache_path)) as cache:
        compressor = LLMCompressor()
        cache.set_compressor(compressor)
        entry = LLMCacheEntry.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "X and Y"}],
            content="Response",
            provider="openai",
        )
        cache.put_data("hash1", entry.to_dict())

    zip_path = tmp_path / "export.zip"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["export_cache", "-c", str(cache_path), "-o", str(zip_path), "--json"],
    )

    assert result.exit_code == 0
    output = json.loads(result.output)
    assert output["format"] == "zip"
    assert output["exported"] == 1
    assert zip_path.exists()


# Test cache export to zip creates parent directories.
def test_cli_cache_export_to_zip_creates_parent_dirs(tmp_path):
    from causaliq_core.cache import TokenCache

    from causaliq_knowledge.llm.cache import LLMCacheEntry, LLMCompressor

    cache_path = tmp_path / "test_cache.db"
    with TokenCache(str(cache_path)) as cache:
        compressor = LLMCompressor()
        cache.set_compressor(compressor)
        entry = LLMCacheEntry.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "test"}],
            content="Response",
        )
        cache.put_data("h1", entry.to_dict())

    # Nested path that doesn't exist
    zip_path = tmp_path / "nested" / "dir" / "export.zip"
    runner = CliRunner()
    result = runner.invoke(
        cli, ["export_cache", "-c", str(cache_path), "-o", str(zip_path)]
    )

    assert result.exit_code == 0
    assert zip_path.exists()


# ============================================================================
# Cache Import Tests
# ============================================================================


# Test import_cache command help.
def test_cli_import_cache_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["import_cache", "--help"])

    assert result.exit_code == 0
    assert "import" in result.output.lower()


# Test cache import from directory with LLM entries.
def test_cli_cache_import_from_directory(tmp_path):
    import json

    from causaliq_core.cache import TokenCache

    from causaliq_knowledge.llm.cache import LLMCompressor

    # Create import directory with LLM JSON file
    import_dir = tmp_path / "import"
    import_dir.mkdir()

    llm_data = {
        "cache_key": {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "test query"}],
            "temperature": 0.1,
            "max_tokens": 100,
        },
        "response": {"content": "test response", "finish_reason": "stop"},
        "metadata": {"provider": "openai"},
    }
    (import_dir / "test_entry.json").write_text(json.dumps(llm_data))

    # Import into new cache
    cache_path = tmp_path / "new_cache.db"
    runner = CliRunner()
    result = runner.invoke(
        cli, ["import_cache", "-c", str(cache_path), "-i", str(import_dir)]
    )

    assert result.exit_code == 0
    assert "Imported 1 entries" in result.output

    # Verify entry was imported
    with TokenCache(str(cache_path)) as cache:
        cache.set_compressor(LLMCompressor())
        assert cache.entry_count() == 1


# Test cache import from zip file.
def test_cli_cache_import_from_zip(tmp_path):
    import json
    import zipfile

    from causaliq_core.cache import TokenCache

    from causaliq_knowledge.llm.cache import LLMCompressor

    # Create zip file with LLM JSON
    zip_path = tmp_path / "import.zip"
    llm_data = {
        "cache_key": {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "zip test"}],
            "temperature": 0.1,
        },
        "response": {"content": "response from zip"},
        "metadata": {},
    }

    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("entry.json", json.dumps(llm_data))

    # Import
    cache_path = tmp_path / "cache.db"
    runner = CliRunner()
    result = runner.invoke(
        cli, ["import_cache", "-c", str(cache_path), "-i", str(zip_path)]
    )

    assert result.exit_code == 0
    assert "zip archive" in result.output
    assert "Imported 1 entries" in result.output

    # Verify
    with TokenCache(str(cache_path)) as cache:
        cache.set_compressor(LLMCompressor())
        assert cache.entry_count() == 1


# Test cache import skips generic JSON (only LLM entries are imported).
def test_cli_cache_import_skips_generic_json(tmp_path):
    import json

    from causaliq_core.cache import TokenCache

    # Create import directory with generic JSON (not LLM format)
    import_dir = tmp_path / "import"
    import_dir.mkdir()

    generic_data = {"key": "value", "number": 42}
    (import_dir / "generic.json").write_text(json.dumps(generic_data))

    cache_path = tmp_path / "cache.db"
    runner = CliRunner()
    result = runner.invoke(
        cli, ["import_cache", "-c", str(cache_path), "-i", str(import_dir)]
    )

    assert result.exit_code == 0
    assert "Imported 0 entries" in result.output
    assert "Skipped: 1" in result.output

    # Verify - generic entries are skipped
    with TokenCache(str(cache_path)) as cache:
        assert cache.entry_count() == 0


# Test cache import JSON output.
def test_cli_cache_import_json_output(tmp_path):
    import json

    # Create import directory
    import_dir = tmp_path / "import"
    import_dir.mkdir()

    llm_data = {
        "cache_key": {"model": "gpt-4", "messages": []},
        "response": {"content": "test"},
        "metadata": {},
    }
    (import_dir / "entry.json").write_text(json.dumps(llm_data))

    cache_path = tmp_path / "cache.db"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "import_cache",
            "-c",
            str(cache_path),
            "-i",
            str(import_dir),
            "--json",
        ],
    )

    assert result.exit_code == 0
    output = json.loads(result.output)
    assert output["imported"] == 1
    assert output["format"] == "directory"


# Test cache import skips invalid JSON files.
def test_cli_cache_import_skips_invalid_files(tmp_path):
    import json

    import_dir = tmp_path / "import"
    import_dir.mkdir()

    # Valid LLM JSON
    llm_data = {
        "cache_key": {"model": "gpt-4", "messages": []},
        "response": {"content": "test"},
        "metadata": {},
    }
    (import_dir / "valid.json").write_text(json.dumps(llm_data))
    # Invalid JSON
    (import_dir / "invalid.json").write_text("not valid json {{{")
    # Non-JSON file (should be ignored)
    (import_dir / "readme.txt").write_text("This is a readme")

    cache_path = tmp_path / "cache.db"
    runner = CliRunner()
    result = runner.invoke(
        cli, ["import_cache", "-c", str(cache_path), "-i", str(import_dir)]
    )

    assert result.exit_code == 0
    assert "Imported 1 entries" in result.output
    assert "Skipped: 1" in result.output


# Test import_cache with empty directory.
def test_cli_import_cache_empty_directory(tmp_path):
    import_dir = tmp_path / "empty"
    import_dir.mkdir()

    cache_path = tmp_path / "cache.db"
    runner = CliRunner()
    result = runner.invoke(
        cli, ["import_cache", "-c", str(cache_path), "-i", str(import_dir)]
    )

    assert result.exit_code == 0
    assert "Imported 0 entries" in result.output


# Test cache import round-trip (export then import).
def test_cli_cache_import_export_roundtrip(tmp_path):
    from causaliq_core.cache import TokenCache

    from causaliq_knowledge.llm.cache import LLMCacheEntry, LLMCompressor

    # Create original cache with data
    original_cache = tmp_path / "original.db"
    with TokenCache(str(original_cache)) as cache:
        compressor = LLMCompressor()
        cache.set_compressor(compressor)
        entry = LLMCacheEntry.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "roundtrip test"}],
            content="This is the response",
            provider="openai",
        )
        cache.put_data("original_key", entry.to_dict())

    # Export to zip
    zip_path = tmp_path / "export.zip"
    runner = CliRunner()
    result = runner.invoke(
        cli, ["export_cache", "-c", str(original_cache), "-o", str(zip_path)]
    )
    assert result.exit_code == 0

    # Import into new cache
    new_cache = tmp_path / "new.db"
    result = runner.invoke(
        cli, ["import_cache", "-c", str(new_cache), "-i", str(zip_path)]
    )
    assert result.exit_code == 0
    assert "Imported 1 entries" in result.output

    # Verify data matches
    with TokenCache(str(new_cache)) as cache:
        cache.set_compressor(LLMCompressor())
        assert cache.entry_count() == 1


# Test cache import skips JSON arrays (non-LLM format).
def test_cli_cache_import_skips_json_array(tmp_path):
    import json

    import_dir = tmp_path / "import"
    import_dir.mkdir()

    # JSON file containing an array - should be skipped (not LLM format)
    (import_dir / "array_data.json").write_text(json.dumps([1, 2, 3]))

    cache_path = tmp_path / "cache.db"
    runner = CliRunner()
    result = runner.invoke(
        cli, ["import_cache", "-c", str(cache_path), "-i", str(import_dir)]
    )

    assert result.exit_code == 0
    assert "Imported 0 entries" in result.output
    assert "Skipped: 1" in result.output


# Test import_cache skips graph entries (only imports LLM entries).
def test_cli_cache_import_skips_graph_entry(tmp_path):
    """Import only accepts LLM entries, graph entries are skipped."""
    import json

    import_dir = tmp_path / "import"
    import_dir.mkdir()

    # Create a graph entry JSON file (not LLM format)
    graph_data = {
        "edges": [
            {"source": "A", "target": "B", "confidence": 0.9},
            {"source": "B", "target": "C", "confidence": 0.8},
        ],
        "variables": ["A", "B", "C"],
        "reasoning": "Test graph import",
    }
    (import_dir / "test_graph.json").write_text(json.dumps(graph_data))

    cache_path = tmp_path / "cache.db"
    runner = CliRunner()
    result = runner.invoke(
        cli, ["import_cache", "-c", str(cache_path), "-i", str(import_dir)]
    )

    assert result.exit_code == 0
    assert "Imported 0 entries" in result.output
    assert "Skipped: 1" in result.output


# Test import_cache error handling with invalid cache path.
def test_cli_import_cache_error_invalid_cache(tmp_path):
    import json

    # Create valid import directory
    import_dir = tmp_path / "import"
    import_dir.mkdir()
    (import_dir / "test.json").write_text(json.dumps({"key": "value"}))

    # Use a directory as cache path (invalid - should be a file)
    invalid_cache = tmp_path / "invalid_cache_dir"
    invalid_cache.mkdir()
    # Create a subdirectory to make it non-empty/invalid for SQLite
    (invalid_cache / "subdir").mkdir()

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "import_cache",
            "-c",
            str(invalid_cache / "subdir"),
            "-i",
            str(import_dir),
        ],
    )

    assert result.exit_code == 1
    assert "Error importing cache" in result.output


# ============================================================================
# Generate CLI Tests
# ============================================================================


# Test generate_graph command shows help.
def test_cli_generate_graph_shows_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["generate_graph", "--help"])

    assert result.exit_code == 0
    assert "generate" in result.output.lower()
    assert "graph" in result.output.lower()


# Test generate_graph command appears in main help.
def test_cli_main_help_shows_generate_graph():
    runner = CliRunner()
    result = runner.invoke(cli, [])

    assert "generate_graph" in result.output


# Test generate_graph command shows options in help.
def test_cli_generate_graph_shows_options():
    runner = CliRunner()
    result = runner.invoke(cli, ["generate_graph", "--help"])

    assert result.exit_code == 0
    assert "--network-context" in result.output
    assert "--prompt-detail" in result.output
    assert "--llm" in result.output
    assert "--output" in result.output
    assert "--llm-temperature" in result.output


# Test generate graph requires context.
def test_cli_generate_graph_requires_context():
    runner = CliRunner()
    result = runner.invoke(cli, ["generate_graph"])

    assert result.exit_code != 0
    assert "Missing option" in result.output or "required" in result.output


# Test generate graph with non-existent file.
def test_cli_generate_graph_missing_file():
    runner = CliRunner()
    result = runner.invoke(cli, ["generate_graph", "-n", "nonexistent.json"])

    assert result.exit_code != 0


# Test generate graph loads network context.
def test_cli_generate_graph_loads_spec(tmp_path, mocker):
    import json

    # Create a valid network context
    spec_data = {
        "network": "test-model",
        "domain": "testing",
        "variables": [
            {"name": "A", "type": "binary", "short_description": "Variable A"},
            {"name": "B", "type": "binary", "short_description": "Variable B"},
        ],
    }
    spec_file = tmp_path / "model.json"
    spec_file.write_text(json.dumps(spec_data))

    # Mock the GraphGenerator to avoid LLM calls
    from causaliq_knowledge.graph.response import GeneratedGraph, ProposedEdge

    mock_graph = GeneratedGraph(
        edges=[ProposedEdge(source="A", target="B", confidence=0.8)],
        variables=["A", "B"],
    )
    mock_generator = mocker.MagicMock()
    mock_generator.generate_from_context.return_value = mock_graph
    mock_generator.get_stats.return_value = {
        "call_count": 1,
        "client_call_count": 1,
    }
    mocker.patch(
        "causaliq_knowledge.graph.generator.GraphGenerator",
        return_value=mock_generator,
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "generate_graph",
            "-n",
            str(spec_file),
            "-c",
            "none",
            "-o",
            "none",
        ],
    )

    assert result.exit_code == 0
    assert "test-model" in result.output or "A" in result.output
    assert "B" in result.output


# Test generate graph with output to file writes JSON.
def test_cli_generate_graph_json_output(tmp_path, mocker):
    import json

    # Create a valid network context
    spec_data = {
        "network": "json-test",
        "domain": "testing",
        "variables": [
            {"name": "X", "type": "binary", "short_description": "Variable X"},
            {"name": "Y", "type": "binary", "short_description": "Variable Y"},
        ],
    }
    spec_file = tmp_path / "model.json"
    spec_file.write_text(json.dumps(spec_data))

    output_dir = tmp_path / "output"

    # Mock the GraphGenerator
    from causaliq_knowledge.graph.response import GeneratedGraph, ProposedEdge

    mock_graph = GeneratedGraph(
        edges=[ProposedEdge(source="X", target="Y", confidence=0.9)],
        variables=["X", "Y"],
    )
    mock_generator = mocker.MagicMock()
    mock_generator.generate_from_context.return_value = mock_graph
    mock_generator.get_stats.return_value = {
        "call_count": 1,
        "client_call_count": 1,
    }
    mocker.patch(
        "causaliq_knowledge.graph.generator.GraphGenerator",
        return_value=mock_generator,
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "generate_graph",
            "-n",
            str(spec_file),
            "-c",
            "none",
            "-o",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0
    assert output_dir.exists()

    # Verify directory output files
    assert (output_dir / "graph.graphml").exists()
    assert (output_dir / "metadata.json").exists()
    assert (output_dir / "confidences.json").exists()

    metadata = json.loads((output_dir / "metadata.json").read_text())
    assert metadata["network"] == "json-test"

    confidences = json.loads((output_dir / "confidences.json").read_text())
    assert "X->Y" in confidences
    assert confidences["X->Y"] == 0.9


# Test generate graph with --output flag writes to directory.
def test_cli_generate_graph_output_file(tmp_path, mocker):
    import json

    # Create a valid network context
    spec_data = {
        "network": "file-output-test",
        "domain": "testing",
        "variables": [
            {"name": "P", "type": "binary", "short_description": "Variable P"},
            {"name": "Q", "type": "binary", "short_description": "Variable Q"},
        ],
    }
    spec_file = tmp_path / "model.json"
    spec_file.write_text(json.dumps(spec_data))

    output_dir = tmp_path / "graph_output"

    # Mock the GraphGenerator
    from causaliq_knowledge.graph.response import GeneratedGraph, ProposedEdge

    mock_graph = GeneratedGraph(
        edges=[ProposedEdge(source="P", target="Q", confidence=0.75)],
        variables=["P", "Q"],
    )
    mock_generator = mocker.MagicMock()
    mock_generator.generate_from_context.return_value = mock_graph
    mock_generator.get_stats.return_value = {
        "call_count": 1,
        "client_call_count": 1,
    }
    mocker.patch(
        "causaliq_knowledge.graph.generator.GraphGenerator",
        return_value=mock_generator,
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "generate_graph",
            "-n",
            str(spec_file),
            "-c",
            "none",
            "-o",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0
    assert output_dir.exists()

    # Verify directory output files
    metadata = json.loads((output_dir / "metadata.json").read_text())
    assert metadata["network"] == "file-output-test"
    confidences = json.loads((output_dir / "confidences.json").read_text())
    assert "P->Q" in confidences
    # Also check edges are printed to stderr
    assert "P → Q" in result.output or "P" in result.output


# Test generate graph with --use-benchmark-names flag.
def test_cli_generate_graph_use_benchmark_names(tmp_path, mocker):
    import json

    # Create a spec with distinct llm_name vs name
    spec_data = {
        "network": "benchmark-test",
        "domain": "testing",
        "variables": [
            {
                "name": "smoke",
                "llm_name": "tobacco_use",
                "type": "binary",
                "short_description": "Smoking status",
            },
            {
                "name": "lung",
                "llm_name": "cancer_status",
                "type": "binary",
                "short_description": "Lung cancer",
            },
        ],
    }
    spec_file = tmp_path / "model.json"
    spec_file.write_text(json.dumps(spec_data))

    from causaliq_knowledge.graph.response import GeneratedGraph, ProposedEdge

    mock_graph = GeneratedGraph(
        edges=[ProposedEdge(source="smoke", target="lung", confidence=0.8)],
        variables=["smoke", "lung"],
    )
    mock_generator = mocker.MagicMock()
    mock_generator.generate_from_context.return_value = mock_graph
    mock_generator.get_stats.return_value = {
        "call_count": 1,
        "client_call_count": 1,
    }
    mocker.patch(
        "causaliq_knowledge.graph.generator.GraphGenerator",
        return_value=mock_generator,
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "generate_graph",
            "-n",
            str(spec_file),
            "-c",
            "none",
            "-o",
            "none",
            "--use-benchmark-names",
        ],
    )

    assert result.exit_code == 0
    assert "benchmark names" in result.output.lower()


# Test generate graph with --prompt-detail minimal flag.
def test_cli_generate_graph_prompt_detail_minimal(tmp_path, mocker):
    import json

    spec_data = {
        "network": "prompt-detail-test",
        "domain": "testing",
        "variables": [
            {"name": "A", "type": "binary", "short_description": "Variable A"},
            {"name": "B", "type": "binary", "short_description": "Variable B"},
        ],
    }
    spec_file = tmp_path / "model.json"
    spec_file.write_text(json.dumps(spec_data))

    from causaliq_knowledge.graph.response import GeneratedGraph

    mock_graph = GeneratedGraph(edges=[], variables=["A", "B"])
    mock_generator = mocker.MagicMock()
    mock_generator.generate_from_context.return_value = mock_graph
    mock_generator.get_stats.return_value = {
        "call_count": 1,
        "client_call_count": 1,
    }
    mocker.patch(
        "causaliq_knowledge.graph.generator.GraphGenerator",
        return_value=mock_generator,
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "generate_graph",
            "-n",
            str(spec_file),
            "-c",
            "none",
            "-o",
            "none",
            "--prompt-detail",
            "minimal",
        ],
    )

    assert result.exit_code == 0
    assert "minimal" in result.output.lower()


# Test generate graph with output none prints adjacency matrix.
def test_cli_generate_graph_output_none_adjacency(tmp_path, mocker):
    import json

    spec_data = {
        "network": "adjacency-test",
        "domain": "testing",
        "variables": [
            {"name": "A", "type": "binary", "short_description": "Variable A"},
            {"name": "B", "type": "binary", "short_description": "Variable B"},
        ],
    }
    spec_file = tmp_path / "model.json"
    spec_file.write_text(json.dumps(spec_data))

    from causaliq_knowledge.graph.response import GeneratedGraph, ProposedEdge

    mock_graph = GeneratedGraph(
        edges=[ProposedEdge(source="A", target="B", confidence=0.8)],
        variables=["A", "B"],
    )
    mock_generator = mocker.MagicMock()
    mock_generator.generate_from_context.return_value = mock_graph
    mock_generator.get_stats.return_value = {
        "call_count": 1,
        "client_call_count": 1,
    }
    mocker.patch(
        "causaliq_knowledge.graph.generator.GraphGenerator",
        return_value=mock_generator,
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "generate_graph",
            "-n",
            str(spec_file),
            "-c",
            "none",
            "-o",
            "none",
        ],
    )

    assert result.exit_code == 0
    # Should print adjacency matrix
    assert "Adjacency Matrix" in result.output
    # Should also print proposed edges
    assert "A → B" in result.output or "Proposed Edges" in result.output


# Test generate graph with invalid context.
def test_cli_generate_graph_invalid_spec(tmp_path):
    # Create an invalid JSON file
    spec_file = tmp_path / "invalid.json"
    spec_file.write_text("not valid json")

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "generate_graph",
            "-n",
            str(spec_file),
            "-c",
            "none",
            "-o",
            "none",
        ],
    )

    assert result.exit_code == 1
    assert "Error loading" in result.output


# Test generate graph with missing required fields.
def test_cli_generate_graph_incomplete_spec(tmp_path):
    import json

    # Create a spec missing required fields
    spec_data = {"network": "incomplete"}  # Missing variables
    spec_file = tmp_path / "incomplete.json"
    spec_file.write_text(json.dumps(spec_data))

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "generate_graph",
            "-n",
            str(spec_file),
            "-c",
            "none",
            "-o",
            "none",
        ],
    )

    assert result.exit_code == 1
    assert "Error loading" in result.output


# Test generate graph with cache option.
def test_cli_generate_graph_with_cache(tmp_path, mocker):
    import json

    spec_data = {
        "network": "cache-test",
        "domain": "testing",
        "variables": [
            {"name": "A", "type": "binary", "short_description": "Variable A"},
            {"name": "B", "type": "binary", "short_description": "Variable B"},
        ],
    }
    spec_file = tmp_path / "model.json"
    spec_file.write_text(json.dumps(spec_data))

    cache_file = tmp_path / "cache.db"

    from causaliq_knowledge.graph.response import GeneratedGraph

    mock_graph = GeneratedGraph(edges=[], variables=["A", "B"])
    mock_generator = mocker.MagicMock()
    mock_generator.generate_from_context.return_value = mock_graph
    mock_generator.get_stats.return_value = {
        "call_count": 1,
        "client_call_count": 1,
    }
    mocker.patch(
        "causaliq_knowledge.graph.generator.GraphGenerator",
        return_value=mock_generator,
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "generate_graph",
            "-n",
            str(spec_file),
            "-c",
            str(cache_file),
            "-o",
            "none",
        ],
    )

    assert result.exit_code == 0
    assert "cache" in result.output.lower()


# Test generate graph empty result.
def test_cli_generate_graph_empty_edges(tmp_path, mocker):
    import json

    spec_data = {
        "network": "empty-test",
        "domain": "testing",
        "variables": [
            {"name": "A", "type": "binary", "short_description": "Variable A"},
            {"name": "B", "type": "binary", "short_description": "Variable B"},
        ],
    }
    spec_file = tmp_path / "model.json"
    spec_file.write_text(json.dumps(spec_data))

    from causaliq_knowledge.graph.response import GeneratedGraph

    mock_graph = GeneratedGraph(edges=[], variables=["A", "B"])
    mock_generator = mocker.MagicMock()
    mock_generator.generate_from_context.return_value = mock_graph
    mock_generator.get_stats.return_value = {
        "call_count": 1,
        "client_call_count": 1,
    }
    mocker.patch(
        "causaliq_knowledge.graph.generator.GraphGenerator",
        return_value=mock_generator,
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "generate_graph",
            "-n",
            str(spec_file),
            "-c",
            "none",
            "-o",
            "none",
        ],
    )

    assert result.exit_code == 0
    assert "No edges" in result.output or "0" in result.output


# Test generate graph with LLM model option.
def test_cli_generate_graph_llm_option(tmp_path, mocker):
    import json

    spec_data = {
        "network": "llm-option-test",
        "domain": "testing",
        "variables": [
            {"name": "A", "type": "binary", "short_description": "Variable A"},
            {"name": "B", "type": "binary", "short_description": "Variable B"},
        ],
    }
    spec_file = tmp_path / "model.json"
    spec_file.write_text(json.dumps(spec_data))

    from causaliq_knowledge.graph.response import GeneratedGraph

    mock_graph = GeneratedGraph(edges=[], variables=["A", "B"])
    mock_generator_class = mocker.patch(
        "causaliq_knowledge.graph.generator.GraphGenerator"
    )
    mock_generator = mocker.MagicMock()
    mock_generator.generate_from_context.return_value = mock_graph
    mock_generator.get_stats.return_value = {
        "call_count": 1,
        "client_call_count": 1,
    }
    mock_generator_class.return_value = mock_generator

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "generate_graph",
            "-n",
            str(spec_file),
            "-c",
            "none",
            "-o",
            "none",
            "-m",
            "gemini/gemini-2.5-flash",
        ],
    )

    assert result.exit_code == 0
    # Verify the model was passed to GraphGenerator
    mock_generator_class.assert_called_once()
    call_kwargs = mock_generator_class.call_args
    assert call_kwargs[1]["model"] == "gemini/gemini-2.5-flash"


# Test generate graph with invalid prompt_detail level.
def test_cli_generate_graph_invalid_prompt_detail_level(tmp_path):
    import json

    spec_data = {
        "network": "invalid-prompt-detail-test",
        "domain": "testing",
        "variables": [
            {"name": "A", "type": "binary", "short_description": "Variable A"},
        ],
    }
    spec_file = tmp_path / "model.json"
    spec_file.write_text(json.dumps(spec_data))

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "generate_graph",
            "-n",
            str(spec_file),
            "--prompt-detail",
            "invalid",
        ],
    )

    assert result.exit_code != 0
    # Click's choice validation produces this message
    assert "is not one of" in result.output


# Test generate graph accepts any path as directory output.
def test_cli_generate_graph_any_path_is_directory_output(tmp_path, mocker):
    import json

    spec_data = {
        "network": "any-path-test",
        "domain": "testing",
        "variables": [
            {"name": "A", "type": "binary", "short_description": "Variable A"},
            {"name": "B", "type": "binary", "short_description": "Variable B"},
        ],
    }
    spec_file = tmp_path / "model.json"
    spec_file.write_text(json.dumps(spec_data))

    # Any path (even with .txt) is treated as directory output
    output_dir = tmp_path / "output.txt"

    from causaliq_knowledge.graph.response import GeneratedGraph, ProposedEdge

    mock_graph = GeneratedGraph(
        edges=[ProposedEdge(source="A", target="B", confidence=0.8)],
        variables=["A", "B"],
    )
    mock_generator = mocker.MagicMock()
    mock_generator.generate_from_context.return_value = mock_graph
    mock_generator.get_stats.return_value = {
        "call_count": 1,
        "client_call_count": 1,
    }
    mocker.patch(
        "causaliq_knowledge.graph.generator.GraphGenerator",
        return_value=mock_generator,
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "generate_graph",
            "-n",
            str(spec_file),
            "-c",
            "none",
            "-o",
            str(output_dir),
        ],
    )

    # Should succeed - path is treated as directory
    assert result.exit_code == 0
    assert output_dir.exists()
    assert (output_dir / "graph.graphml").exists()


# Test generate graph with invalid llm_model triggers validation error.
def test_cli_generate_graph_invalid_llm_model(tmp_path):
    import json

    spec_data = {
        "network": "validation-test",
        "domain": "testing",
        "variables": [
            {"name": "A", "type": "binary", "short_description": "Variable A"},
        ],
    }
    spec_file = tmp_path / "model.json"
    spec_file.write_text(json.dumps(spec_data))

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "generate_graph",
            "-n",
            str(spec_file),
            "-c",
            "none",
            "-o",
            "none",
            "-m",
            "invalid-model-no-prefix",
        ],
    )

    assert result.exit_code == 1
    assert "Error:" in result.output
    assert "llm_model" in result.output or "prefix" in result.output


# Test generate graph with cache open error.
def test_cli_generate_graph_cache_error(tmp_path, mocker):
    import json

    spec_data = {
        "network": "cache-error-test",
        "domain": "testing",
        "variables": [
            {"name": "A", "type": "binary", "short_description": "Variable A"},
        ],
    }
    spec_file = tmp_path / "model.json"
    spec_file.write_text(json.dumps(spec_data))

    # Mock TokenCache to raise an exception - patch in core cache module
    mocker.patch(
        "causaliq_core.cache.TokenCache",
        side_effect=Exception("Cache error"),
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "generate_graph",
            "-n",
            str(spec_file),
            "-c",
            str(tmp_path / "cache.db"),
            "-o",
            "none",
        ],
    )

    assert result.exit_code == 1
    assert "Error opening cache" in result.output


# Test generate graph with generator creation error.
def test_cli_generate_graph_generator_error(tmp_path, mocker):
    import json

    spec_data = {
        "network": "generator-error-test",
        "domain": "testing",
        "variables": [
            {"name": "A", "type": "binary", "short_description": "Variable A"},
        ],
    }
    spec_file = tmp_path / "model.json"
    spec_file.write_text(json.dumps(spec_data))

    # Mock GraphGenerator to raise ValueError
    mocker.patch(
        "causaliq_knowledge.graph.generator.GraphGenerator",
        side_effect=ValueError("Invalid model"),
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "generate_graph",
            "-n",
            str(spec_file),
            "-c",
            "none",
            "-o",
            "none",
        ],
    )

    assert result.exit_code == 1
    assert "Error creating generator" in result.output


# Test generate graph with generation error.
def test_cli_generate_graph_generation_error(tmp_path, mocker):
    import json

    spec_data = {
        "network": "generation-error-test",
        "domain": "testing",
        "variables": [
            {"name": "A", "type": "binary", "short_description": "Variable A"},
        ],
    }
    spec_file = tmp_path / "model.json"
    spec_file.write_text(json.dumps(spec_data))

    # Mock GraphGenerator to raise exception on generate
    mock_generator = mocker.MagicMock()
    mock_generator.generate_from_context.side_effect = Exception("LLM error")
    mocker.patch(
        "causaliq_knowledge.graph.generator.GraphGenerator",
        return_value=mock_generator,
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "generate_graph",
            "-n",
            str(spec_file),
            "-c",
            "none",
            "-o",
            "none",
        ],
    )

    assert result.exit_code == 1
    assert "Error generating graph" in result.output


# Test generate graph with metadata in output.
def test_cli_generate_graph_with_metadata(tmp_path, mocker):
    import json

    spec_data = {
        "network": "metadata-test",
        "domain": "testing",
        "variables": [
            {"name": "X", "type": "binary", "short_description": "Variable X"},
            {"name": "Y", "type": "binary", "short_description": "Variable Y"},
        ],
    }
    spec_file = tmp_path / "model.json"
    spec_file.write_text(json.dumps(spec_data))

    output_dir = tmp_path / "output"

    from causaliq_knowledge.graph.response import (
        GeneratedGraph,
        GenerationMetadata,
        ProposedEdge,
    )

    mock_graph = GeneratedGraph(
        edges=[ProposedEdge(source="X", target="Y", confidence=0.8)],
        variables=["X", "Y"],
        metadata=GenerationMetadata(
            model="test-model",
            provider="test-provider",
            input_tokens=100,
            output_tokens=50,
            from_cache=False,
            messages=[{"role": "user", "content": "test prompt"}],
            temperature=0.1,
            max_tokens=2000,
            finish_reason="stop",
            llm_cost_usd=0.001,
        ),
    )
    mock_generator = mocker.MagicMock()
    mock_generator.generate_from_context.return_value = mock_graph
    mock_generator.get_stats.return_value = {
        "call_count": 1,
        "client_call_count": 1,
    }
    mocker.patch(
        "causaliq_knowledge.graph.generator.GraphGenerator",
        return_value=mock_generator,
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "generate_graph",
            "-n",
            str(spec_file),
            "-c",
            "none",
            "-o",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0
    metadata = json.loads((output_dir / "metadata.json").read_text())
    # Verify all generation metadata fields are present
    assert metadata["llm_input_tokens"] == 100
    assert metadata["llm_output_tokens"] == 50
    assert metadata["llm_provider"] == "test-provider"
    assert metadata["llm_messages"] == [
        {"role": "user", "content": "test prompt"}
    ]
    assert metadata["llm_temperature"] == 0.1
    assert metadata["llm_max_tokens"] == 2000
    assert metadata["llm_finish_reason"] == "stop"
    assert metadata["llm_cost_usd"] == 0.001
    assert "llm_timestamp" in metadata


# Test generate graph human-readable output with reasoning.
def test_cli_generate_graph_human_readable_with_reasoning(tmp_path, mocker):
    import json

    spec_data = {
        "network": "human-reasoning-test",
        "domain": "testing",
        "variables": [
            {"name": "X", "type": "binary", "short_description": "Variable X"},
            {"name": "Y", "type": "binary", "short_description": "Variable Y"},
        ],
    }
    spec_file = tmp_path / "model.json"
    spec_file.write_text(json.dumps(spec_data))

    from causaliq_knowledge.graph.response import GeneratedGraph, ProposedEdge

    mock_graph = GeneratedGraph(
        edges=[
            ProposedEdge(
                source="X",
                target="Y",
                confidence=0.85,
                reasoning=(
                    "This is a test reasoning that " "explains the causal link"
                ),
            )
        ],
        variables=["X", "Y"],
    )
    mock_generator = mocker.MagicMock()
    mock_generator.generate_from_context.return_value = mock_graph
    mock_generator.get_stats.return_value = {
        "call_count": 1,
        "client_call_count": 1,
    }
    mocker.patch(
        "causaliq_knowledge.graph.generator.GraphGenerator",
        return_value=mock_generator,
    )

    runner = CliRunner()
    # Use -o none to get adjacency matrix and human-readable output to stdout
    result = runner.invoke(
        cli,
        [
            "generate_graph",
            "-n",
            str(spec_file),
            "-c",
            "none",
            "-o",
            "none",
        ],
    )

    assert result.exit_code == 0
    assert "X → Y" in result.output
    assert "This is a test reasoning" in result.output


# Test generate graph human-readable with long reasoning gets truncated.
def test_cli_generate_graph_long_reasoning_truncated(tmp_path, mocker):
    import json

    spec_data = {
        "network": "long-reasoning-test",
        "domain": "testing",
        "variables": [
            {"name": "X", "type": "binary", "short_description": "Variable X"},
            {"name": "Y", "type": "binary", "short_description": "Variable Y"},
        ],
    }
    spec_file = tmp_path / "model.json"
    spec_file.write_text(json.dumps(spec_data))

    from causaliq_knowledge.graph.response import GeneratedGraph, ProposedEdge

    # Create a long reasoning string (>100 chars)
    long_reasoning = "A" * 150

    mock_graph = GeneratedGraph(
        edges=[
            ProposedEdge(
                source="X",
                target="Y",
                confidence=0.85,
                reasoning=long_reasoning,
            )
        ],
        variables=["X", "Y"],
    )
    mock_generator = mocker.MagicMock()
    mock_generator.generate_from_context.return_value = mock_graph
    mock_generator.get_stats.return_value = {
        "call_count": 1,
        "client_call_count": 1,
    }
    mocker.patch(
        "causaliq_knowledge.graph.generator.GraphGenerator",
        return_value=mock_generator,
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "generate_graph",
            "-n",
            str(spec_file),
            "-c",
            "none",
            "-o",
            "none",
        ],
    )

    assert result.exit_code == 0
    # Should truncate at 100 chars and add ...
    assert "A" * 100 + "..." in result.output


# Test generate graph with workflow cache output (.db file).
def test_cli_generate_graph_workflow_cache_output(tmp_path, mocker):
    """Test generate_graph writes to Workflow Cache when output is .db file."""
    import json

    spec_data = {
        "network": "cache-test",
        "domain": "testing",
        "variables": [
            {"name": "X", "type": "binary", "short_description": "Variable X"},
            {"name": "Y", "type": "binary", "short_description": "Variable Y"},
        ],
    }
    spec_file = tmp_path / "model.json"
    spec_file.write_text(json.dumps(spec_data))

    from causaliq_knowledge.graph.response import GeneratedGraph, ProposedEdge

    mock_graph = GeneratedGraph(
        edges=[ProposedEdge(source="X", target="Y", confidence=0.9)],
        variables=["X", "Y"],
    )
    mock_generator = mocker.MagicMock()
    mock_generator.generate_from_context.return_value = mock_graph
    mock_generator.get_stats.return_value = {
        "call_count": 1,
        "client_call_count": 1,
    }
    mocker.patch(
        "causaliq_knowledge.graph.generator.GraphGenerator",
        return_value=mock_generator,
    )

    output_db = tmp_path / "results.db"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "generate_graph",
            "-n",
            str(spec_file),
            "-c",
            "none",
            "-o",
            str(output_db),
        ],
    )

    assert result.exit_code == 0
    assert "Output written to:" in result.output
    assert str(output_db) in result.output

    # Verify the workflow cache was created and contains the graph
    import base64

    from causaliq_workflow import WorkflowCache

    from causaliq_knowledge.graph.cache import GraphCompressor

    assert output_db.exists()

    with WorkflowCache(str(output_db)) as wf_cache:
        compressor = GraphCompressor()
        entry = wf_cache.get({"network": "cache-test"})

        # Entry contains the graph object as a base64-encoded blob
        assert entry is not None
        graph_obj = entry.get_object("graph")
        assert graph_obj is not None

        # Decode the base64 blob and then decompress the graph
        blob = base64.b64decode(graph_obj.content)
        retrieved, _extra_blobs = compressor.decompress_entry(
            blob, wf_cache.token_cache
        )
    assert len(retrieved.edges) == 1
    assert retrieved.edges[0].source == "X"
    assert retrieved.edges[0].target == "Y"


# Test generate graph loads comprehensive model file from test data.
def test_cli_generate_graph_comprehensive_model_file(mocker):
    """Test generate_graph with comprehensive model from tests/data."""
    from pathlib import Path

    from causaliq_knowledge.graph.response import GeneratedGraph, ProposedEdge

    # Use the comprehensive test model file
    models_dir = (
        Path(__file__).parent.parent / "data" / "functional" / "models"
    )
    spec_file = models_dir / "comprehensive.json"

    mock_graph = GeneratedGraph(
        edges=[
            ProposedEdge(source="Exposure", target="Disease", confidence=0.9),
            ProposedEdge(source="Lifestyle", target="Disease", confidence=0.9),
            ProposedEdge(
                source="Disease", target="TestResult", confidence=0.8
            ),
            ProposedEdge(source="Disease", target="Symptom", confidence=0.8),
        ],
        variables=[
            "Exposure",
            "Lifestyle",
            "Disease",
            "TestResult",
            "Symptom",
        ],
    )
    mock_generator = mocker.MagicMock()
    mock_generator.generate_from_context.return_value = mock_graph
    mock_generator.get_stats.return_value = {
        "call_count": 1,
        "client_call_count": 1,
    }
    mocker.patch(
        "causaliq_knowledge.graph.generator.GraphGenerator",
        return_value=mock_generator,
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "generate_graph",
            "-n",
            str(spec_file),
            "-c",
            "none",
            "-o",
            "none",
        ],
    )

    assert result.exit_code == 0
    # Verify context was loaded and used
    assert "comprehensive_test" in result.output or "Exposure" in result.output
    assert "Disease" in result.output


# Test export_cache skips entries that fail to decompress as LLM entries.
def test_cli_export_cache_skips_invalid_entries(tmp_path):
    """Test export gracefully skips entries that fail to decompress."""
    from datetime import datetime, timezone

    from causaliq_core.cache import TokenCache

    from causaliq_knowledge.llm.cache import LLMCacheEntry, LLMCompressor

    cache_path = tmp_path / "mixed_cache.db"
    with TokenCache(str(cache_path)) as cache:
        # Store a valid LLM entry
        compressor = LLMCompressor()
        cache.set_compressor(compressor)
        entry = LLMCacheEntry.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "test"}],
            content="Response",
            provider="openai",
            request_id="valid_entry",
        )
        cache.put_data("valid_hash", entry.to_dict())

        # Insert invalid/corrupted entry directly that can't be decompressed
        timestamp = datetime.now(timezone.utc).isoformat()
        cache.conn.execute(
            "INSERT INTO cache_entries "
            "(hash, seq, key_json, data, created_at, hit_count) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                "invalid_hash",
                0,
                "",
                "corrupted_data_not_valid_tokens",
                timestamp,
                0,
            ),
        )
        cache.conn.commit()

    export_dir = tmp_path / "export"
    runner = CliRunner()
    result = runner.invoke(
        cli, ["export_cache", "-c", str(cache_path), "-o", str(export_dir)]
    )

    # Should succeed and export 1 valid entry (skip the invalid one)
    assert result.exit_code == 0
    assert "Exported 1 entries" in result.output
    files = list(export_dir.glob("*.json"))
    assert len(files) == 1
