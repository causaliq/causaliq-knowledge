"""
Functional tests for CLI cache stats command with LLM entries.

Tests the model breakdown table, cost calculations, and hit rate display.
"""

from click.testing import CliRunner

from causaliq_knowledge.cli import cli


# Test cache stats displays LLM model breakdown table.
def test_cli_cache_stats_llm_model_breakdown(tmp_path):
    from causaliq_core.cache import TokenCache

    from causaliq_knowledge.llm.cache import LLMCacheEntry, LLMEntryEncoder

    cache_path = tmp_path / "llm_cache.db"
    with TokenCache(str(cache_path)) as cache:
        cache.register_encoder("llm", LLMEntryEncoder())

        # Create an LLM cache entry
        entry = LLMCacheEntry.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            content='{"response": "Hi"}',
            provider="openai",
            latency_ms=500,
            input_tokens=10,
            output_tokens=5,
            cost_usd=0.001,
        )
        cache.put_data("hash1", "llm", entry.to_dict())

    runner = CliRunner()
    result = runner.invoke(cli, ["cache", "stats", str(cache_path)])

    assert result.exit_code == 0
    # Check table header
    assert "Model" in result.output
    assert "Entries" in result.output
    assert "Hits" in result.output
    assert "Hit Rate" in result.output
    assert "Tokens In" in result.output
    assert "Tokens Out" in result.output
    assert "Cost" in result.output
    assert "Latency" in result.output
    # Check model row
    assert "gpt-4" in result.output
    assert "$0.0010" in result.output


# Test cache stats displays multiple models.
def test_cli_cache_stats_multiple_models(tmp_path):
    from causaliq_core.cache import TokenCache

    from causaliq_knowledge.llm.cache import LLMCacheEntry, LLMEntryEncoder

    cache_path = tmp_path / "llm_cache.db"
    with TokenCache(str(cache_path)) as cache:
        cache.register_encoder("llm", LLMEntryEncoder())

        # Create entries for different models
        for model, cost in [("gpt-4", 0.01), ("claude-3", 0.005)]:
            entry = LLMCacheEntry.create(
                model=model,
                messages=[{"role": "user", "content": "Test"}],
                content="Response",
                provider="test",
                latency_ms=100,
                input_tokens=50,
                output_tokens=25,
                cost_usd=cost,
            )
            cache.put_data(f"hash_{model}", "llm", entry.to_dict())

    runner = CliRunner()
    result = runner.invoke(cli, ["cache", "stats", str(cache_path)])

    assert result.exit_code == 0
    assert "gpt-4" in result.output
    assert "claude-3" in result.output
    # Total cost should be sum
    assert "Total cost:" in result.output


# Test cache stats shows token totals.
def test_cli_cache_stats_token_totals(tmp_path):
    from causaliq_core.cache import TokenCache

    from causaliq_knowledge.llm.cache import LLMCacheEntry, LLMEntryEncoder

    cache_path = tmp_path / "llm_cache.db"
    with TokenCache(str(cache_path)) as cache:
        cache.register_encoder("llm", LLMEntryEncoder())

        entry = LLMCacheEntry.create(
            model="test-model",
            messages=[{"role": "user", "content": "Test"}],
            content="Response",
            provider="test",
            latency_ms=200,
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.002,
        )
        cache.put_data("hash1", "llm", entry.to_dict())

    runner = CliRunner()
    result = runner.invoke(cli, ["cache", "stats", str(cache_path)])

    assert result.exit_code == 0
    assert "Total tokens:" in result.output
    assert "100" in result.output  # input tokens
    assert "50" in result.output  # output tokens


# Test cache stats calculates estimated savings from hits.
def test_cli_cache_stats_estimated_savings(tmp_path):
    from causaliq_core.cache import TokenCache

    from causaliq_knowledge.llm.cache import LLMCacheEntry, LLMEntryEncoder

    cache_path = tmp_path / "llm_cache.db"
    with TokenCache(str(cache_path)) as cache:
        cache.register_encoder("llm", LLMEntryEncoder())

        entry = LLMCacheEntry.create(
            model="test-model",
            messages=[{"role": "user", "content": "Test"}],
            content="Response",
            provider="test",
            latency_ms=100,
            input_tokens=50,
            output_tokens=25,
            cost_usd=0.01,
        )
        cache.put_data("hash1", "llm", entry.to_dict())

        # Simulate cache hits by calling get multiple times
        cache.get_data("hash1", "llm")
        cache.get_data("hash1", "llm")

    runner = CliRunner()
    result = runner.invoke(cli, ["cache", "stats", str(cache_path)])

    assert result.exit_code == 0
    assert "Est. savings:" in result.output
    # With 2 hits at $0.01 per request, savings should be $0.02
    assert "$0.0200" in result.output


# Test cache stats calculates average latency.
def test_cli_cache_stats_average_latency(tmp_path):
    from causaliq_core.cache import TokenCache

    from causaliq_knowledge.llm.cache import LLMCacheEntry, LLMEntryEncoder

    cache_path = tmp_path / "llm_cache.db"
    with TokenCache(str(cache_path)) as cache:
        cache.register_encoder("llm", LLMEntryEncoder())

        # Add two entries with different latencies
        for i, latency in enumerate([200, 400]):
            entry = LLMCacheEntry.create(
                model="test-model",
                messages=[{"role": "user", "content": f"Test {i}"}],
                content="Response",
                provider="test",
                latency_ms=latency,
                input_tokens=50,
                output_tokens=25,
                cost_usd=0.001,
            )
            cache.put_data(f"hash{i}", "llm", entry.to_dict())

    runner = CliRunner()
    result = runner.invoke(cli, ["cache", "stats", str(cache_path)])

    assert result.exit_code == 0
    # Average latency should be (200 + 400) / 2 = 300
    assert "300 ms" in result.output


# Test cache stats aggregates entries per model.
def test_cli_cache_stats_aggregates_by_model(tmp_path):
    from causaliq_core.cache import TokenCache

    from causaliq_knowledge.llm.cache import LLMCacheEntry, LLMEntryEncoder

    cache_path = tmp_path / "llm_cache.db"
    with TokenCache(str(cache_path)) as cache:
        cache.register_encoder("llm", LLMEntryEncoder())

        # Add 3 entries for same model
        for i in range(3):
            entry = LLMCacheEntry.create(
                model="gpt-4",
                messages=[{"role": "user", "content": f"Test {i}"}],
                content="Response",
                provider="openai",
                latency_ms=100,
                input_tokens=10,
                output_tokens=5,
                cost_usd=0.001,
            )
            cache.put_data(f"hash{i}", "llm", entry.to_dict())

    runner = CliRunner()
    result = runner.invoke(cli, ["cache", "stats", str(cache_path)])

    assert result.exit_code == 0
    # Should show 3 entries for gpt-4
    # The table format has entries column
    assert "3" in result.output


# Test cache stats hit rate calculation.
def test_cli_cache_stats_hit_rate(tmp_path):
    from causaliq_core.cache import TokenCache

    from causaliq_knowledge.llm.cache import LLMCacheEntry, LLMEntryEncoder

    cache_path = tmp_path / "llm_cache.db"
    with TokenCache(str(cache_path)) as cache:
        cache.register_encoder("llm", LLMEntryEncoder())

        entry = LLMCacheEntry.create(
            model="test-model",
            messages=[{"role": "user", "content": "Test"}],
            content="Response",
            provider="test",
            latency_ms=100,
            input_tokens=50,
            output_tokens=25,
            cost_usd=0.01,
        )
        cache.put_data("hash1", "llm", entry.to_dict())

        # 1 entry + 1 hit = 50% hit rate
        cache.get_data("hash1", "llm")

    runner = CliRunner()
    result = runner.invoke(cli, ["cache", "stats", str(cache_path)])

    assert result.exit_code == 0
    # Hit rate should be 50%
    assert "50.0%" in result.output


# Test cache stats JSON output includes model breakdown.
def test_cli_cache_stats_json_includes_models(tmp_path):
    import json

    from causaliq_core.cache import TokenCache

    from causaliq_knowledge.llm.cache import LLMCacheEntry, LLMEntryEncoder

    cache_path = tmp_path / "llm_cache.db"
    with TokenCache(str(cache_path)) as cache:
        cache.register_encoder("llm", LLMEntryEncoder())

        entry = LLMCacheEntry.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Test"}],
            content="Response",
            provider="openai",
            latency_ms=100,
            input_tokens=50,
            output_tokens=25,
            cost_usd=0.001,
        )
        cache.put_data("hash1", "llm", entry.to_dict())

    runner = CliRunner()
    result = runner.invoke(cli, ["cache", "stats", str(cache_path), "--json"])

    assert result.exit_code == 0
    output = json.loads(result.output)
    assert "by_model" in output
    assert "gpt-4" in output["by_model"]
    assert output["by_model"]["gpt-4"]["entries"] == 1
    assert output["by_model"]["gpt-4"]["input_tokens"] == 50
    assert output["by_model"]["gpt-4"]["output_tokens"] == 25


# Test cache stats truncates long model names.
def test_cli_cache_stats_truncates_long_model_name(tmp_path):
    from causaliq_core.cache import TokenCache

    from causaliq_knowledge.llm.cache import LLMCacheEntry, LLMEntryEncoder

    cache_path = tmp_path / "llm_cache.db"
    with TokenCache(str(cache_path)) as cache:
        cache.register_encoder("llm", LLMEntryEncoder())

        long_model_name = "a-very-long-model-name-that-exceeds-32-characters"
        entry = LLMCacheEntry.create(
            model=long_model_name,
            messages=[{"role": "user", "content": "Test"}],
            content="Response",
            provider="test",
            latency_ms=100,
            input_tokens=50,
            output_tokens=25,
            cost_usd=0.001,
        )
        cache.put_data("hash1", "llm", entry.to_dict())

    runner = CliRunner()
    result = runner.invoke(cli, ["cache", "stats", str(cache_path)])

    assert result.exit_code == 0
    # Long name should be truncated with "..."
    assert "..." in result.output
    # Full name should NOT appear
    assert long_model_name not in result.output


# Test cache stats with zero entries shows no model table.
def test_cli_cache_stats_empty_llm_cache(tmp_path):
    from causaliq_core.cache import TokenCache

    cache_path = tmp_path / "empty_cache.db"
    with TokenCache(str(cache_path)):
        pass  # Create empty cache

    runner = CliRunner()
    result = runner.invoke(cli, ["cache", "stats", str(cache_path)])

    assert result.exit_code == 0
    assert "Entries:" in result.output
    assert "0" in result.output
    # Should not show model table headers
    assert "Hit Rate" not in result.output


# Test cache stats skips entries that cannot be decoded.
def test_cli_cache_stats_skips_invalid_entries(tmp_path):
    from datetime import datetime, timezone

    from causaliq_core.cache import TokenCache

    from causaliq_knowledge.llm.cache import LLMCacheEntry, LLMEntryEncoder

    cache_path = tmp_path / "mixed_cache.db"
    with TokenCache(str(cache_path)) as cache:
        cache.register_encoder("llm", LLMEntryEncoder())

        # Add a valid entry
        entry = LLMCacheEntry.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Test"}],
            content="Response",
            provider="openai",
            latency_ms=100,
            input_tokens=50,
            output_tokens=25,
            cost_usd=0.001,
        )
        cache.put_data("valid_hash", "llm", entry.to_dict())

        # Manually insert an invalid/corrupted entry directly into DB
        timestamp = datetime.now(timezone.utc).isoformat()
        cache.conn.execute(
            "INSERT INTO cache_entries "
            "(hash, entry_type, seq, key_json, data, created_at, hit_count) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                "invalid_hash",
                "llm",
                0,
                "",
                "corrupted_data_not_valid_tokens",
                timestamp,
                0,
            ),
        )
        cache.conn.commit()

    runner = CliRunner()
    result = runner.invoke(cli, ["cache", "stats", str(cache_path)])

    # Should still succeed, just skipping the invalid entry
    assert result.exit_code == 0
    # The valid entry should still be counted
    assert "gpt-4" in result.output
