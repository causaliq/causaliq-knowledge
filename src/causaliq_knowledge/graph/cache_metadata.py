"""Conversion of cached metadata dictionaries to GenerationMetadata.

Cached graphs record their generation provenance as a JSON dictionary. The
multi-blob format and the legacy dictionary format use the same key
fall-backs, so the conversion lives here once.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

from causaliq_knowledge.graph.response import GenerationMetadata


def _parse_llm_timestamp(meta: Dict[str, Any]) -> datetime:
    """Return the LLM timestamp, falling back to the request timestamp."""
    llm_ts_str = meta.get("llm_timestamp") or meta.get("timestamp")
    if llm_ts_str:
        return datetime.fromisoformat(llm_ts_str)
    return datetime.now(timezone.utc)


def _parse_request_timestamp(
    meta: Dict[str, Any],
    llm_ts: datetime,
) -> datetime:
    """Return the request timestamp, falling back to the LLM timestamp."""
    ts_str = meta.get("timestamp")
    if ts_str:
        return datetime.fromisoformat(ts_str)
    return llm_ts  # Fall back to llm_timestamp


def metadata_from_dict(meta: Dict[str, Any]) -> GenerationMetadata:
    """Build GenerationMetadata from a cached metadata dictionary.

    Args:
        meta: Metadata dictionary recorded in a cache entry.

    Returns:
        GenerationMetadata with the recorded generation provenance.
    """
    # Parse llm_timestamp - fall back to timestamp for compatibility
    llm_ts = _parse_llm_timestamp(meta)
    # Parse current request timestamp
    ts = _parse_request_timestamp(meta, llm_ts)
    # Parse cost - prefer llm_cost_usd, fall back to initial_cost_usd
    llm_cost = meta.get("llm_cost_usd", meta.get("initial_cost_usd", 0.0))
    # Parse latency - prefer llm_latency_ms, fall back to latency_ms
    llm_latency = meta.get("llm_latency_ms", meta.get("latency_ms", 0))

    return GenerationMetadata(
        model=meta.get("llm_model", meta.get("model", "")),
        provider=meta.get("llm_provider", meta.get("provider", "")),
        timestamp=ts,
        llm_timestamp=llm_ts,
        llm_latency_ms=llm_latency,
        input_tokens=meta.get("llm_input_tokens", meta.get("input_tokens", 0)),
        output_tokens=meta.get(
            "llm_output_tokens", meta.get("output_tokens", 0)
        ),
        from_cache=meta.get("from_cache", False),
        messages=meta.get("llm_messages", meta.get("messages", [])),
        temperature=meta.get("llm_temperature", meta.get("temperature", 0.1)),
        max_tokens=meta.get("llm_max_tokens", meta.get("max_tokens", 2000)),
        finish_reason=meta.get(
            "llm_finish_reason", meta.get("finish_reason", "stop")
        ),
        llm_cost_usd=llm_cost,
    )
