"""Helper functions for the cache management CLI commands.

These helpers keep the click command bodies small: they aggregate cache
statistics, export entries, and import entries. All user-facing output is
emitted by the commands themselves in `causaliq_knowledge.cli.cache`.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple


def is_llm_entry(data: Any) -> bool:
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


def is_graph_entry(data: Any) -> bool:
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


def count_cache_entries(cache: Any) -> int:
    """Return the number of entries stored in a cache."""
    cursor = cache.conn.execute("SELECT COUNT(*) FROM cache_entries")
    entry_count: int = cursor.fetchone()[0]
    return entry_count


def _accumulate_model_stats(
    model_stats: Dict[str, Dict[str, Any]],
    totals: Dict[str, float],
    entry: Any,
    hit_count: int,
) -> None:
    """Add one cache entry to the per-model and overall totals."""
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

    totals["cost"] += entry.metadata.cost_usd
    totals["input_tokens"] += entry.metadata.tokens.input
    totals["output_tokens"] += entry.metadata.tokens.output


def _collect_row(
    cache: Any,
    compressor: Any,
    model_stats: Dict[str, Dict[str, Any]],
    totals: Dict[str, float],
    row: Any,
) -> None:
    """Add one cache row to the aggregates, skipping unusable entries."""
    from causaliq_knowledge.llm.cache import LLMCacheEntry

    try:
        data = compressor.decompress(row[0], cache)
        entry = LLMCacheEntry.from_dict(data)
        _accumulate_model_stats(model_stats, totals, entry, row[1] or 0)
    except Exception:
        # Skip entries that can't be decompressed
        pass


def collect_model_stats(
    cache: Any,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, float]]:
    """Aggregate cache statistics per model, and overall totals."""
    from causaliq_knowledge.llm.cache import LLMCompressor

    # Use LLMCompressor for decompression
    compressor = LLMCompressor()
    cache.set_compressor(compressor)

    model_stats: dict[str, dict[str, Any]] = {}
    totals: Dict[str, float] = {
        "cost": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
    }

    # Query all entries
    cursor = cache.conn.execute("SELECT data, hit_count FROM cache_entries")
    for row in cursor.fetchall():
        _collect_row(cache, compressor, model_stats, totals, row)

    return model_stats, totals


def finalise_model_stats(model_stats: Dict[str, Dict[str, Any]]) -> None:
    """Calculate averages and drop the intermediate latency total."""
    # Calculate averages
    for stats in model_stats.values():
        if stats["entries"] > 0:
            stats["avg_latency_ms"] = (
                stats["total_latency_ms"] // stats["entries"]
            )
        del stats["total_latency_ms"]


def compute_savings(
    model_stats: Dict[str, Dict[str, Any]],
    total_cost: float,
    entry_count: int,
    total_hits: int,
) -> float:
    """Calculate savings (cost avoided by cache hits)."""
    total_requests = sum(
        s["entries"] + s["hits"] for s in model_stats.values()
    )
    if total_requests > 0 and entry_count > 0:
        avg_cost_per_request = total_cost / entry_count
        return total_hits * avg_cost_per_request
    return 0.0


def _hit_rate(stats: Dict[str, Any]) -> float:
    """Return the percentage of requests served from the cache."""
    requests = stats["entries"] + stats["hits"]
    rate: float = stats["hits"] / requests * 100 if requests > 0 else 0
    return rate


def _format_model_rows(
    model_stats: Dict[str, Dict[str, Any]],
) -> List[str]:
    """Format one report row per model."""
    rows = []
    for model, stats in sorted(model_stats.items()):
        hit_rate = _hit_rate(stats)
        # Truncate model name if too long
        model_display = model[:29] + "..." if len(model) > 32 else model
        cost_str = f"${stats['cost_usd']:.4f}"
        latency_str = f"{stats['avg_latency_ms']:,} ms"
        rows.append(
            f"{model_display:<32}  {stats['entries']:>8,}  "
            f"{stats['hits']:>8,}  {hit_rate:>7.1f}%  "
            f"{stats['input_tokens']:>12,}  "
            f"{stats['output_tokens']:>12,}  "
            f"{cost_str:>10}  {latency_str:>10}"
        )
    return rows


def _format_totals_lines(
    total_cost: float,
    total_input_tokens: int,
    total_output_tokens: int,
    savings: float,
) -> List[str]:
    """Format the totals section of the statistics report."""
    return [
        f"\nTotal cost:       ${total_cost:.4f}",
        f"Est. savings:     ${savings:.4f}",
        (
            f"Total tokens:     {total_input_tokens:,} in / "
            f"{total_output_tokens:,} out"
        ),
    ]


def format_stats_lines(
    input_path: str,
    entry_count: int,
    token_count: int,
    total_hits: int,
    model_stats: Dict[str, Dict[str, Any]],
    totals: Dict[str, float],
    savings: float,
) -> List[str]:
    """Build the cache statistics report lines."""
    lines: list[str] = []
    lines.append(f"\nCache: {input_path}")
    lines.append("=" * 60)
    lines.append(f"Entries:          {entry_count:,}")
    lines.append(f"Token dictionary: {token_count:,}")
    lines.append(f"Total cache hits: {total_hits:,}")

    if model_stats:
        lines.extend(
            _format_totals_lines(
                totals["cost"],
                int(totals["input_tokens"]),
                int(totals["output_tokens"]),
                savings,
            )
        )

        # Table header
        lines.append("")
        lines.append(
            f"{'Model':<32}  {'Entries':>8}  {'Hits':>8}  "
            f"{'Hit Rate':>8}  {'Tokens In':>12}  "
            f"{'Tokens Out':>12}  "
            f"{'Cost':>10}  {'Latency':>10}"
        )
        lines.append("-" * 114)
        lines.extend(_format_model_rows(model_stats))

    lines.append("")
    return lines


def prepare_export_dir(dest_path: Path, is_zip: bool) -> Tuple[Path, str]:
    """Return the export directory and any temporary directory to remove."""
    # Determine export directory (temp if zipping)
    if is_zip:
        temp_dir = tempfile.mkdtemp()
        return Path(temp_dir), temp_dir
    dest_path.mkdir(parents=True, exist_ok=True)
    return dest_path, ""


def _export_entry(
    cache: Any,
    compressor: Any,
    export_dir: Path,
    cache_key: str,
    blob: Any,
) -> bool:
    """Export one cache entry, returning False when it cannot be read."""
    from causaliq_knowledge.llm.cache import LLMCacheEntry

    try:
        data = compressor.decompress(blob, cache)
        entry = LLMCacheEntry.from_dict(data)
        filename = compressor.generate_export_filename(entry, cache_key)
        compressor.export_entry(entry, export_dir / filename)
    except Exception:
        # Skip entries that can't be decompressed as LLM entries
        return False
    return True


def export_entries(cache: Any, compressor: Any, export_dir: Path) -> int:
    """Export each cache entry as a JSON file, returning the count."""
    exported = 0
    cursor = cache.conn.execute("SELECT hash, data FROM cache_entries")
    for cache_key, blob in cursor:
        if _export_entry(cache, compressor, export_dir, cache_key, blob):
            exported += 1
    return exported


def _write_archive_entries(zf: zipfile.ZipFile, export_dir: Path) -> None:
    """Add each file of the export directory to the archive."""
    for file_path in export_dir.iterdir():
        if file_path.is_file():
            zf.write(file_path, file_path.name)


def archive_export_dir(
    dest_path: Path,
    export_dir: Path,
    temp_dir: str,
) -> None:
    """Create the zip archive and remove the temporary directory."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest_path, "w", zipfile.ZIP_DEFLATED) as zf:
        _write_archive_entries(zf, export_dir)
    # Clean up temp directory
    shutil.rmtree(temp_dir)


def export_entries_to_target(
    cache: Any,
    compressor: Any,
    dest_path: Path,
    is_zip: bool,
) -> int:
    """Export entries to a directory or zip archive and return the count."""
    export_dir, temp_dir = prepare_export_dir(dest_path, is_zip)
    exported = export_entries(cache, compressor, export_dir)
    if is_zip:
        archive_export_dir(dest_path, export_dir, temp_dir)
    return exported


def prepare_import_dir(input_file: Path, is_zip: bool) -> Tuple[Path, str]:
    """Return the import directory and any temporary directory to remove."""
    # Determine input directory
    if not is_zip:
        return input_file, ""
    temp_dir = tempfile.mkdtemp()
    import_dir = Path(temp_dir)
    with zipfile.ZipFile(input_file, "r") as zf:
        zf.extractall(import_dir)
    return import_dir, temp_dir


def _entry_cache_key(data: Dict[str, Any]) -> str:
    """Return the cache key of an LLM entry payload."""
    cache_key_data = data.get("cache_key", {})
    key_str = json.dumps(cache_key_data, sort_keys=True)
    return hashlib.sha256(key_str.encode()).hexdigest()[:16]


def _import_file(
    cache: Any,
    file_path: Path,
    imported: int,
    skipped: int,
) -> Tuple[int, int]:
    """Import one JSON file, returning the updated counters."""
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return imported, skipped + 1

    # Only import LLM entries
    if is_llm_entry(data):
        cache.put_data(_entry_cache_key(data), data)
        return imported + 1, skipped
    return imported, skipped + 1


def import_entries(cache: Any, import_dir: Path) -> Tuple[int, int]:
    """Import cache entries from a directory, returning counts."""
    imported = 0
    skipped = 0
    for file_path in import_dir.iterdir():
        if not file_path.is_file() or file_path.suffix.lower() != ".json":
            continue
        imported, skipped = _import_file(cache, file_path, imported, skipped)
    return imported, skipped


def remove_temp_dir(temp_dir: str) -> None:
    """Remove a temporary directory when one was created."""
    if temp_dir:
        shutil.rmtree(temp_dir)
