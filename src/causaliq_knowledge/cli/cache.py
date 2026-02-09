"""Cache management CLI commands.

This module provides commands for managing the LLM response cache:
- cache_stats: Show cache statistics
- export_cache: Export cache entries to files
- import_cache: Import cache entries from files
"""

from __future__ import annotations

import json
import sys
from typing import Any

import click


@click.command("cache_stats")
@click.option(
    "--cache",
    "-c",
    "cache_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to the SQLite cache database.",
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output result as JSON.",
)
def cache_stats(cache_path: str, output_json: bool) -> None:
    """Show cache statistics.

    Shows entry counts, token dictionary size, cache hit statistics,
    and for LLM caches: breakdown by model with token usage and costs.

    Examples:

        cqknow cache_stats -c ./llm_cache.db

        cqknow cache_stats -c ./llm_cache.db --json
    """
    from causaliq_core.cache import TokenCache

    from causaliq_knowledge.llm.cache import LLMCacheEntry, LLMEntryEncoder

    try:
        with TokenCache(cache_path) as cache:
            entry_count = cache.entry_count()
            token_count = cache.token_count()
            total_hits = cache.total_hits()

            # Aggregate LLM-specific stats by model
            model_stats: dict[str, dict[str, Any]] = {}
            total_cost = 0.0
            total_input_tokens = 0
            total_output_tokens = 0

            if cache.has_encoder("llm") or entry_count > 0:
                # Register encoder if needed
                if not cache.has_encoder("llm"):
                    cache.register_encoder("llm", LLMEntryEncoder())

                # Query all LLM entries with hit counts
                cursor = cache.conn.execute(
                    "SELECT data, hit_count FROM cache_entries "
                    "WHERE entry_type = 'llm'"
                )
                encoder = LLMEntryEncoder()

                for row in cursor.fetchall():
                    try:
                        data = encoder.decode(row[0], cache)
                        entry = LLMCacheEntry.from_dict(data)
                        hit_count = row[1] or 0

                        model = entry.model
                        if model not in model_stats:
                            model_stats[model] = {
                                "provider": entry.metadata.provider,
                                "entries": 0,
                                "hits": 0,
                                "input_tokens": 0,
                                "output_tokens": 0,
                                "cost_usd": 0.0,
                                "avg_latency_ms": 0,
                                "total_latency_ms": 0,
                            }

                        stats = model_stats[model]
                        stats["entries"] += 1
                        stats["hits"] += hit_count
                        stats["input_tokens"] += entry.metadata.tokens.input
                        stats["output_tokens"] += entry.metadata.tokens.output
                        stats["cost_usd"] += entry.metadata.cost_usd
                        stats["total_latency_ms"] += entry.metadata.latency_ms

                        total_cost += entry.metadata.cost_usd
                        total_input_tokens += entry.metadata.tokens.input
                        total_output_tokens += entry.metadata.tokens.output
                    except Exception:
                        # Skip entries that can't be decoded
                        pass

                # Calculate averages
                for stats in model_stats.values():
                    if stats["entries"] > 0:
                        stats["avg_latency_ms"] = (
                            stats["total_latency_ms"] // stats["entries"]
                        )
                    del stats["total_latency_ms"]

            # Calculate savings (cost avoided by cache hits)
            total_requests = sum(
                s["entries"] + s["hits"] for s in model_stats.values()
            )
            if total_requests > 0 and entry_count > 0:
                avg_cost_per_request = total_cost / entry_count
                savings = total_hits * avg_cost_per_request
            else:
                savings = 0.0

            if output_json:
                output: dict[str, Any] = {
                    "cache_path": cache_path,
                    "summary": {
                        "entry_count": entry_count,
                        "token_count": token_count,
                        "total_hits": total_hits,
                        "total_cost_usd": round(total_cost, 6),
                        "estimated_savings_usd": round(savings, 6),
                        "total_input_tokens": total_input_tokens,
                        "total_output_tokens": total_output_tokens,
                    },
                    "by_model": model_stats,
                }
                click.echo(json.dumps(output, indent=2))
            else:
                click.echo(f"\nCache: {cache_path}")
                click.echo("=" * 60)
                click.echo(f"Entries:          {entry_count:,}")
                click.echo(f"Token dictionary: {token_count:,}")
                click.echo(f"Total cache hits: {total_hits:,}")

                if model_stats:
                    click.echo(f"\nTotal cost:       ${total_cost:.4f}")
                    click.echo(f"Est. savings:     ${savings:.4f}")
                    click.echo(
                        f"Total tokens:     {total_input_tokens:,} in / "
                        f"{total_output_tokens:,} out"
                    )

                    # Table header
                    click.echo()
                    click.echo(
                        f"{'Model':<32}  {'Entries':>8}  {'Hits':>8}  "
                        f"{'Hit Rate':>8}  {'Tokens In':>12}  "
                        f"{'Tokens Out':>12}  "
                        f"{'Cost':>10}  {'Latency':>10}"
                    )
                    click.echo("-" * 114)

                    for model, stats in sorted(model_stats.items()):
                        hit_rate = (
                            stats["hits"]
                            / (stats["entries"] + stats["hits"])
                            * 100
                            if (stats["entries"] + stats["hits"]) > 0
                            else 0
                        )
                        # Truncate model name if too long
                        model_display = (
                            model[:29] + "..." if len(model) > 32 else model
                        )
                        cost_str = f"${stats['cost_usd']:.4f}"
                        latency_str = f"{stats['avg_latency_ms']:,} ms"
                        click.echo(
                            f"{model_display:<32}  {stats['entries']:>8,}  "
                            f"{stats['hits']:>8,}  {hit_rate:>7.1f}%  "
                            f"{stats['input_tokens']:>12,}  "
                            f"{stats['output_tokens']:>12,}  "
                            f"{cost_str:>10}  {latency_str:>10}"
                        )

                click.echo()
    except Exception as e:
        click.echo(f"Error opening cache: {e}", err=True)
        sys.exit(1)


@click.command("export_cache")
@click.option(
    "--cache",
    "-c",
    "cache_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to the SQLite cache database to export from.",
)
@click.option(
    "--output",
    "-o",
    "output_dir",
    required=True,
    type=click.Path(),
    help="Output directory or .zip file path for exported entries.",
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output result as JSON.",
)
def export_cache(cache_path: str, output_dir: str, output_json: bool) -> None:
    """Export cache entries to human-readable files.

    If output path ends with .zip, entries are exported to a zip archive.
    Otherwise, entries are exported to a directory.

    Files are named using a human-readable format:
        {model}_{node_a}_{node_b}_edge_{hash}.json

    Examples:

        cqknow export_cache -c ./llm_cache.db -o ./export_dir

        cqknow export_cache -c ./llm_cache.db -o ./export.zip

        cqknow export_cache -c ./llm_cache.db -o ./export_dir --json
    """
    import tempfile
    import zipfile
    from pathlib import Path

    from causaliq_core.cache import TokenCache
    from causaliq_core.cache.encoders import JsonEncoder

    from causaliq_knowledge.graph.cache import GraphEntryEncoder
    from causaliq_knowledge.llm.cache import LLMCacheEntry, LLMEntryEncoder

    output_path = Path(output_dir)
    is_zip = output_path.suffix.lower() == ".zip"

    try:
        with TokenCache(cache_path) as cache:
            # Register encoders for decoding
            encoder = LLMEntryEncoder()
            cache.register_encoder("llm", encoder)

            # Register GraphEntryEncoder for graph types
            graph_encoder = GraphEntryEncoder()
            cache.register_encoder("graph", graph_encoder)

            # Register generic JsonEncoder for other types
            json_encoder = JsonEncoder()
            cache.register_encoder("json", json_encoder)

            # Get entry types in the cache
            entry_types = cache.list_entry_types()

            if not entry_types:
                if output_json:
                    click.echo(json.dumps({"exported": 0, "error": None}))
                else:
                    click.echo("No entries to export.")
                return

            # Determine export directory (temp if zipping)
            if is_zip:
                temp_dir = tempfile.mkdtemp()
                export_dir = Path(temp_dir)
            else:
                export_dir = output_path
                export_dir.mkdir(parents=True, exist_ok=True)

            # Export entries
            exported = 0
            for entry_type in entry_types:
                if entry_type == "llm":
                    # Query all entries of this type
                    cursor = cache.conn.execute(
                        "SELECT hash, data FROM cache_entries "
                        "WHERE entry_type = ?",
                        (entry_type,),
                    )
                    for cache_key, blob in cursor:
                        data = encoder.decode(blob, cache)
                        entry = LLMCacheEntry.from_dict(data)
                        filename = encoder.generate_export_filename(
                            entry, cache_key
                        )
                        file_path = export_dir / filename
                        encoder.export_entry(entry, file_path)
                        exported += 1
                else:
                    # For non-LLM types, use generic export
                    count = cache.export_entries(export_dir, entry_type)
                    exported += count

            # Create zip archive if requested
            if is_zip:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(
                    output_path, "w", zipfile.ZIP_DEFLATED
                ) as zf:
                    for file_path in export_dir.iterdir():
                        if file_path.is_file():
                            zf.write(file_path, file_path.name)
                # Clean up temp directory
                import shutil

                shutil.rmtree(temp_dir)

            # Output results
            if output_json:
                output = {
                    "cache_path": cache_path,
                    "output_path": str(output_path),
                    "format": "zip" if is_zip else "directory",
                    "exported": exported,
                    "entry_types": entry_types,
                }
                click.echo(json.dumps(output, indent=2))
            else:
                fmt = "zip archive" if is_zip else "directory"
                click.echo(
                    f"\nExported {exported} entries to {fmt}: {output_path}"
                )
                click.echo(f"Entry types: {', '.join(entry_types)}")
                click.echo()

    except Exception as e:
        click.echo(f"Error exporting cache: {e}", err=True)
        sys.exit(1)


def _is_llm_entry(data: Any) -> bool:
    """Check if JSON data represents an LLM cache entry.

    LLM entries have a specific structure with cache_key containing
    model and messages, plus a response object.
    """
    if not isinstance(data, dict):
        return False
    cache_key = data.get("cache_key", {})
    return (
        isinstance(cache_key, dict)
        and "model" in cache_key
        and "messages" in cache_key
        and "response" in data
    )


def _is_graph_entry(data: Any) -> bool:
    """Check if JSON data represents a graph cache entry.

    Graph entries have edges list, variables list, and optionally metadata.
    """
    if not isinstance(data, dict):
        return False
    return (
        "edges" in data
        and isinstance(data["edges"], list)
        and "variables" in data
        and isinstance(data["variables"], list)
    )


@click.command("import_cache")
@click.option(
    "--cache",
    "-c",
    "cache_path",
    required=True,
    type=click.Path(),
    help="Path to SQLite cache database (created if needed).",
)
@click.option(
    "--input",
    "-i",
    "input_path",
    required=True,
    type=click.Path(exists=True),
    help="Directory or .zip file containing JSON files to import.",
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output result as JSON.",
)
def import_cache(cache_path: str, input_path: str, output_json: bool) -> None:
    """Import cache entries from files.

    Entry types are auto-detected from JSON structure:
    - LLM entries: contain cache_key.model, cache_key.messages, response
    - Graph entries: contain edges list and variables list
    - Generic JSON: anything else

    Examples:

        cqknow import_cache -c ./llm_cache.db -i ./import_dir

        cqknow import_cache -c ./llm_cache.db -i ./export.zip

        cqknow import_cache -c ./llm_cache.db -i ./import_dir --json
    """
    import hashlib
    import tempfile
    import zipfile
    from pathlib import Path

    from causaliq_core.cache import TokenCache
    from causaliq_core.cache.encoders import JsonEncoder

    from causaliq_knowledge.graph.cache import GraphEntryEncoder
    from causaliq_knowledge.llm.cache import LLMEntryEncoder

    input_file = Path(input_path)
    is_zip = input_file.suffix.lower() == ".zip"

    try:
        with TokenCache(cache_path) as cache:
            # Register encoders
            llm_encoder = LLMEntryEncoder()
            graph_encoder = GraphEntryEncoder()
            json_encoder = JsonEncoder()
            cache.register_encoder("llm", llm_encoder)
            cache.register_encoder("graph", graph_encoder)
            cache.register_encoder("json", json_encoder)

            # Determine input directory
            if is_zip:
                temp_dir = tempfile.mkdtemp()
                import_dir = Path(temp_dir)
                with zipfile.ZipFile(input_file, "r") as zf:
                    zf.extractall(import_dir)
            else:
                import_dir = input_file
                temp_dir = None

            # Import entries
            imported = 0
            llm_count = 0
            graph_count = 0
            json_count = 0
            skipped = 0

            for file_path in import_dir.iterdir():
                if (
                    not file_path.is_file()
                    or file_path.suffix.lower() != ".json"
                ):
                    continue

                try:
                    data = json.loads(file_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    skipped += 1
                    continue

                # Detect entry type and generate cache key
                if _is_llm_entry(data):
                    # LLM entry - generate hash from cache_key contents
                    cache_key_data = data.get("cache_key", {})
                    key_str = json.dumps(cache_key_data, sort_keys=True)
                    cache_key = hashlib.sha256(key_str.encode()).hexdigest()[
                        :16
                    ]
                    cache.put_data(cache_key, "llm", data)
                    llm_count += 1
                elif _is_graph_entry(data):
                    # Graph entry - use filename stem as key
                    cache_key = file_path.stem
                    cache.put_data(cache_key, "graph", data)
                    graph_count += 1
                else:
                    # Generic JSON - use filename stem as key
                    cache_key = file_path.stem
                    cache.put_data(cache_key, "json", data)
                    json_count += 1

                imported += 1

            # Clean up temp directory
            if temp_dir:
                import shutil

                shutil.rmtree(temp_dir)

            # Output results
            if output_json:
                output = {
                    "cache_path": cache_path,
                    "input_path": str(input_file),
                    "format": "zip" if is_zip else "directory",
                    "imported": imported,
                    "llm_entries": llm_count,
                    "graph_entries": graph_count,
                    "json_entries": json_count,
                    "skipped": skipped,
                }
                click.echo(json.dumps(output, indent=2))
            else:
                fmt = "zip archive" if is_zip else "directory"
                click.echo(
                    f"\nImported {imported} entries from {fmt}: {input_file}"
                )
                if llm_count:
                    click.echo(f"  LLM entries: {llm_count}")
                if graph_count:
                    click.echo(f"  Graph entries: {graph_count}")
                if json_count:
                    click.echo(f"  JSON entries: {json_count}")
                if skipped:
                    click.echo(f"  Skipped: {skipped}")
                click.echo()

    except Exception as e:
        click.echo(f"Error importing cache: {e}", err=True)
        sys.exit(1)
