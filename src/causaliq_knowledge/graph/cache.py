"""
Graph cache encoder for storing generated graphs in Workflow Caches.

This module provides GraphEntryEncoder for compact binary storage of
LLM-generated causal graphs with associated metadata. The encoder uses
a flexible multi-blob format that can accommodate additional binary
data (e.g., execution traces) in future iterations.

Multi-blob Binary Format
------------------------
The format stores multiple named binary blobs with shared metadata:

    Header (3 bytes):
    - 1 byte: format version (currently 0x01)
    - 2 bytes: blob count (uint16, big-endian)

    For each blob:
    - 2 bytes: blob type token ID (uint16, from TokenCache)
    - 4 bytes: blob length (uint32, big-endian)
    - N bytes: blob data

    Remaining bytes: tokenised JSON metadata

This format is extensible - new blob types can be added without
breaking compatibility. Blob types are strings stored via TokenCache
(e.g., "graph", "trace", "model").
"""

from __future__ import annotations

import json
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from causaliq_core.cache.encoders import JsonEncoder
from causaliq_core.graph import SDG

from causaliq_knowledge.graph.response import (
    GeneratedGraph,
    GenerationMetadata,
    ProposedEdge,
)

if TYPE_CHECKING:  # pragma: no cover
    from causaliq_core.cache import TokenCache


# Format version for multi-blob encoding
_FORMAT_VERSION = 0x01

# Blob type identifiers (stored as tokenised strings)
BLOB_TYPE_GRAPH = "graph"
BLOB_TYPE_CONFIDENCES = "confidences"
BLOB_TYPE_TRACE = "trace"


class GraphEntryEncoder(JsonEncoder):
    """Encoder for storing GeneratedGraph objects in Workflow Caches.

    This encoder combines:
    - Compact binary encoding for graph structure via SDG.encode()
    - Flexible multi-blob format for additional binary data
    - Tokenised JSON for metadata (provenance, edge confidences)

    The format achieves good compression while preserving all graph
    metadata including edge confidences and generation provenance.

    The multi-blob format allows storing additional binary data alongside
    the graph (e.g., execution traces) without breaking compatibility.

    Example:
        >>> from causaliq_core.cache import TokenCache
        >>> from causaliq_knowledge.graph.cache import GraphEntryEncoder
        >>> from causaliq_knowledge.graph.response import (
        ...     GeneratedGraph, ProposedEdge, GenerationMetadata
        ... )
        >>> with TokenCache(":memory:") as cache:
        ...     encoder = GraphEntryEncoder()
        ...     graph = GeneratedGraph(
        ...         edges=[ProposedEdge(
        ...             source="A", target="B", confidence=0.9
        ...         )],
        ...         variables=["A", "B"],
        ...         reasoning="A causes B",
        ...         metadata=GenerationMetadata(model="test"),
        ...     )
        ...     blob = encoder.encode_entry(graph, cache)
        ...     restored = encoder.decode_entry(blob, cache)
        ...     assert restored.edges[0].source == "A"

        # Multi-blob encoding with additional data:
        >>> with TokenCache(":memory:") as cache:
        ...     encoder = GraphEntryEncoder()
        ...     blobs = {"graph": sdg.encode(), "trace": trace_bytes}
        ...     metadata = {"model": "gpt-4", "variables": ["A", "B"]}
        ...     encoded = encoder.encode_multi(blobs, metadata, cache)
        ...     result_blobs, result_meta = encoder.decode_multi(
        ...         encoded, cache
        ...     )
    """

    @property
    def default_export_format(self) -> str:
        """Default file extension for exports."""
        return "json"

    # -------------------------------------------------------------------------
    # Multi-blob encoding/decoding (flexible format)
    # -------------------------------------------------------------------------

    def encode_multi(
        self,
        blobs: Dict[str, bytes],
        metadata: Dict[str, Any],
        cache: "TokenCache",
    ) -> bytes:
        """Encode multiple named blobs with shared metadata.

        This is the core flexible encoding method. Each blob is identified
        by a string type name (e.g., "graph", "trace") which is tokenised
        for compact storage.

        Args:
            blobs: Dictionary of blob_type -> binary data.
            metadata: Shared metadata dictionary (JSON-serialisable).
            cache: TokenCache for shared token dictionary.

        Returns:
            Combined binary representation.

        Example:
            >>> blobs = {"graph": graph_bytes, "trace": trace_bytes}
            >>> metadata = {"model": "gpt-4", "variables": ["A", "B"]}
            >>> encoded = encoder.encode_multi(blobs, metadata, cache)
        """
        result = bytearray()

        # Header: version (1 byte) + blob count (2 bytes)
        result.append(_FORMAT_VERSION)
        result.extend(struct.pack(">H", len(blobs)))

        # Each blob: type token ID (2 bytes) + length (4 bytes) + data
        for blob_type, blob_data in blobs.items():
            type_token_id = cache.get_or_create_token(blob_type)
            result.extend(struct.pack(">H", type_token_id))
            result.extend(struct.pack(">I", len(blob_data)))
            result.extend(blob_data)

        # Metadata as tokenised JSON
        meta_blob = super().encode(metadata, cache)
        result.extend(meta_blob)

        return bytes(result)

    def decode_multi(
        self, blob: bytes, cache: "TokenCache"
    ) -> tuple[Dict[str, bytes], Dict[str, Any]]:
        """Decode multiple named blobs with shared metadata.

        Args:
            blob: Binary data from cache.
            cache: TokenCache for shared token dictionary.

        Returns:
            Tuple of (blobs dict, metadata dict).

        Raises:
            ValueError: If format version is unsupported or data corrupted.

        Example:
            >>> blobs, metadata = encoder.decode_multi(encoded, cache)
            >>> graph_bytes = blobs.get("graph")
            >>> trace_bytes = blobs.get("trace")
        """
        offset = 0

        # Read header
        version = blob[offset]
        offset += 1

        if version != _FORMAT_VERSION:
            raise ValueError(
                f"Unsupported format version: {version} "
                f"(expected {_FORMAT_VERSION})"
            )

        blob_count = struct.unpack(">H", blob[offset : offset + 2])[0]
        offset += 2

        # Read each blob
        blobs_dict: Dict[str, bytes] = {}
        for _ in range(blob_count):
            type_token_id = struct.unpack(">H", blob[offset : offset + 2])[0]
            offset += 2

            blob_len = struct.unpack(">I", blob[offset : offset + 4])[0]
            offset += 4

            blob_data = blob[offset : offset + blob_len]
            offset += blob_len

            # Look up blob type from token
            blob_type = cache.get_token(type_token_id)
            if blob_type is None:
                raise ValueError(
                    f"Unknown blob type token ID: {type_token_id}"
                )

            blobs_dict[blob_type] = blob_data

        # Decode metadata from remaining bytes
        meta_blob = blob[offset:]
        metadata = super().decode(meta_blob, cache)

        return blobs_dict, metadata

    # -------------------------------------------------------------------------
    # Graph-specific encoding (uses multi-blob format internally)
    # -------------------------------------------------------------------------

    def _graph_to_sdg(self, graph: GeneratedGraph) -> SDG:
        """Convert GeneratedGraph to SDG for binary encoding.

        Only edges with confidence >= 0.5 are included as directed edges.
        Lower confidence edges are excluded from the SDG structure.

        Args:
            graph: The generated graph to convert.

        Returns:
            SDG instance representing the graph structure.
        """
        # Build edges as directed edges (->)
        edges = []
        for edge in graph.edges:
            if edge.confidence >= 0.5:
                edges.append((edge.source, "->", edge.target))

        return SDG(list(graph.variables), edges)

    def _sdg_to_edges(
        self,
        sdg: SDG,
        confidence_map: Dict[str, float],
        reasoning_map: Dict[str, str],
    ) -> List[ProposedEdge]:
        """Convert SDG edges back to ProposedEdge list.

        Args:
            sdg: The SDG instance to convert.
            confidence_map: Map of "source->target" to confidence.
            reasoning_map: Map of "source->target" to reasoning.

        Returns:
            List of ProposedEdge objects.
        """
        edges = []
        for (source, target), _ in sdg.edges.items():
            key = f"{source}->{target}"
            edges.append(
                ProposedEdge(
                    source=source,
                    target=target,
                    confidence=confidence_map.get(key, 0.5),
                    reasoning=reasoning_map.get(key),
                )
            )
        return edges

    def _build_metadata_dict(self, graph: GeneratedGraph) -> Dict[str, Any]:
        """Build metadata dictionary for JSON encoding.

        Note: Edge confidences are stored in a separate blob, not in
        metadata. This keeps metadata focused on provenance info.

        Args:
            graph: The generated graph.

        Returns:
            Dictionary containing graph metadata (excluding confidences).
        """
        # Build reasoning map (confidences handled separately)
        reasoning_map = {}
        for edge in graph.edges:
            if edge.reasoning:
                key = f"{edge.source}->{edge.target}"
                reasoning_map[key] = edge.reasoning

        # Build metadata dict (no confidences - they go in separate blob)
        meta: Dict[str, Any] = {
            "variables": graph.variables,
            "reasoning": graph.reasoning,
        }

        if reasoning_map:
            meta["edge_reasoning"] = reasoning_map

        if graph.metadata:
            meta["generation"] = graph.metadata.to_dict()

        return meta

    def _build_confidences_dict(
        self, graph: GeneratedGraph
    ) -> Optional[Dict[str, float]]:
        """Build edge confidences dictionary.

        Returns None if no edges have non-default confidences, allowing
        the confidences blob to be omitted (e.g., for structure learning
        results which don't have confidence scores).

        Args:
            graph: The generated graph.

        Returns:
            Dictionary of "source->target" to confidence, or None.
        """
        confidence_map = {}
        has_non_default = False

        for edge in graph.edges:
            key = f"{edge.source}->{edge.target}"
            confidence_map[key] = edge.confidence
            # Check if any confidence differs from default (0.5)
            if edge.confidence != 0.5:
                has_non_default = True

        # Only return confidences if there are meaningful values
        return confidence_map if has_non_default else None

    def encode_entry(
        self,
        graph: GeneratedGraph,
        cache: "TokenCache",
        extra_blobs: Optional[Dict[str, bytes]] = None,
        include_confidences: bool = True,
    ) -> bytes:
        """Encode a GeneratedGraph to compact binary format.

        Uses the multi-blob format internally, storing the graph structure
        as a "graph" blob and optionally edge confidences as a separate
        "confidences" blob.

        Args:
            graph: The graph to encode.
            cache: TokenCache for shared token dictionary.
            extra_blobs: Optional additional blobs to include
                (e.g., {"trace": trace_bytes}).
            include_confidences: Whether to include confidences blob.
                Set False for structure learning results without scores.

        Returns:
            Compact binary representation.
        """
        # Build blobs dict with graph structure
        sdg = self._graph_to_sdg(graph)
        blobs = {BLOB_TYPE_GRAPH: sdg.encode()}

        # Add confidences blob if requested and present
        if include_confidences:
            confidences = self._build_confidences_dict(graph)
            if confidences:
                # Encode confidences as tokenised JSON
                conf_blob = super().encode(confidences, cache)
                blobs[BLOB_TYPE_CONFIDENCES] = conf_blob

        # Add any extra blobs
        if extra_blobs:
            blobs.update(extra_blobs)

        # Build metadata dict (excludes confidences)
        metadata = self._build_metadata_dict(graph)

        return self.encode_multi(blobs, metadata, cache)

    def decode_entry(
        self,
        blob: bytes,
        cache: "TokenCache",
    ) -> tuple[GeneratedGraph, Dict[str, bytes]]:
        """Decode a GeneratedGraph from binary format.

        Returns both the reconstructed graph and any extra blobs that
        were stored alongside it (e.g., execution traces).

        Args:
            blob: Binary data from cache.
            cache: TokenCache for shared token dictionary.

        Returns:
            Tuple of (GeneratedGraph, extra_blobs dict).
            Extra blobs excludes "graph" and "confidences" blobs.

        Raises:
            ValueError: If no graph blob found in data.
        """
        # Decode using multi-blob format
        blobs, meta_dict = self.decode_multi(blob, cache)

        # Extract graph blob
        if BLOB_TYPE_GRAPH not in blobs:
            raise ValueError("No graph blob found in encoded data")

        graph_blob = blobs.pop(BLOB_TYPE_GRAPH)
        sdg = SDG.decode(graph_blob)

        # Extract confidences from blob (if present)
        confidence_map: Dict[str, float] = {}
        if BLOB_TYPE_CONFIDENCES in blobs:
            conf_blob = blobs.pop(BLOB_TYPE_CONFIDENCES)
            confidence_map = super().decode(conf_blob, cache)

        # Extract reasoning from metadata
        reasoning_map = meta_dict.get("edge_reasoning", {})

        # Build edges from SDG + confidences + reasoning
        edges = self._sdg_to_edges(sdg, confidence_map, reasoning_map)

        # Build generation metadata
        gen_meta = None
        if "generation" in meta_dict:
            gen = meta_dict["generation"]
            # Parse timestamp - prefer llm_timestamp, fall back to timestamp
            llm_ts_str = gen.get("llm_timestamp") or gen.get("timestamp")
            llm_ts = (
                datetime.fromisoformat(llm_ts_str)
                if llm_ts_str
                else datetime.now(timezone.utc)
            )
            # Parse cost - prefer llm_cost_usd, fall back to initial_cost_usd
            llm_cost = gen.get(
                "llm_cost_usd", gen.get("initial_cost_usd", 0.0)
            )

            gen_meta = GenerationMetadata(
                model=gen.get("model", ""),
                provider=gen.get("provider", ""),
                timestamp=llm_ts,
                latency_ms=gen.get("latency_ms", 0),
                input_tokens=gen.get("input_tokens", 0),
                output_tokens=gen.get("output_tokens", 0),
                cost_usd=gen.get("cost_usd", 0.0),
                from_cache=gen.get("from_cache", False),
                messages=gen.get("messages", []),
                temperature=gen.get("temperature", 0.1),
                max_tokens=gen.get("max_tokens", 2000),
                finish_reason=gen.get("finish_reason", "stop"),
                llm_cost_usd=llm_cost,
            )

        graph = GeneratedGraph(
            edges=edges,
            variables=meta_dict.get("variables", list(sdg.nodes)),
            reasoning=meta_dict.get("reasoning", ""),
            metadata=gen_meta,
        )

        # Return graph and any extra blobs (graph + confidences already popped)
        return graph, blobs

    def encode(self, data: Any, cache: "TokenCache") -> bytes:
        """Encode data to binary format.

        Args:
            data: The data to encode (GeneratedGraph or dict with edges).
            cache: TokenCache for shared token dictionary.

        Returns:
            Compact binary representation.
        """
        if isinstance(data, GeneratedGraph):
            return self.encode_entry(data, cache)
        elif isinstance(data, dict) and "edges" in data:
            graph = self._dict_to_graph(data)
            return self.encode_entry(graph, cache)
        return super().encode(data, cache)

    def decode(
        self, blob: bytes, cache: "TokenCache"
    ) -> tuple[GeneratedGraph, Dict[str, bytes]]:
        """Decode binary data to GeneratedGraph and extra blobs.

        Args:
            blob: Binary data from cache.
            cache: TokenCache for shared token dictionary.

        Returns:
            Tuple of (GeneratedGraph, extra_blobs dict).
        """
        return self.decode_entry(blob, cache)

    def _dict_to_graph(self, data: Dict[str, Any]) -> GeneratedGraph:
        """Convert dictionary to GeneratedGraph.

        Args:
            data: Dictionary with graph data.

        Returns:
            GeneratedGraph instance.
        """
        edges = []
        for edge_data in data.get("edges", []):
            if isinstance(edge_data, ProposedEdge):
                edges.append(edge_data)
            else:
                edges.append(
                    ProposedEdge(
                        source=edge_data.get("source", ""),
                        target=edge_data.get("target", ""),
                        confidence=edge_data.get("confidence", 0.5),
                        reasoning=edge_data.get("reasoning"),
                    )
                )

        metadata = None
        if "metadata" in data and data["metadata"]:
            meta = data["metadata"]
            if isinstance(meta, GenerationMetadata):
                metadata = meta
            else:
                # Parse timestamp (prefer llm_timestamp, fallback timestamp)
                llm_ts_str = meta.get("llm_timestamp") or meta.get("timestamp")
                llm_ts = (
                    datetime.fromisoformat(llm_ts_str)
                    if llm_ts_str
                    else datetime.now(timezone.utc)
                )
                # Parse cost (prefer llm_cost_usd, fallback initial_cost_usd)
                llm_cost = meta.get(
                    "llm_cost_usd", meta.get("initial_cost_usd", 0.0)
                )
                metadata = GenerationMetadata(
                    model=meta.get("model", ""),
                    provider=meta.get("provider", ""),
                    timestamp=llm_ts,
                    latency_ms=meta.get("latency_ms", 0),
                    input_tokens=meta.get("input_tokens", 0),
                    output_tokens=meta.get("output_tokens", 0),
                    cost_usd=meta.get("cost_usd", 0.0),
                    from_cache=meta.get("from_cache", False),
                    messages=meta.get("messages", []),
                    temperature=meta.get("temperature", 0.1),
                    max_tokens=meta.get("max_tokens", 2000),
                    finish_reason=meta.get("finish_reason", "stop"),
                    llm_cost_usd=llm_cost,
                )

        return GeneratedGraph(
            edges=edges,
            variables=data.get("variables", []),
            reasoning=data.get("reasoning", ""),
            metadata=metadata,
        )

    def _graph_to_export_dict(self, graph: GeneratedGraph) -> Dict[str, Any]:
        """Convert GeneratedGraph to export dictionary.

        Args:
            graph: The graph to export.

        Returns:
            Dictionary suitable for JSON export.
        """
        result: Dict[str, Any] = {
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "confidence": e.confidence,
                    **({"reasoning": e.reasoning} if e.reasoning else {}),
                }
                for e in graph.edges
            ],
            "variables": graph.variables,
            "reasoning": graph.reasoning,
        }

        if graph.metadata:
            meta_dict: Dict[str, Any] = {
                "model": graph.metadata.model,
                "provider": graph.metadata.provider,
                "llm_timestamp": graph.metadata.timestamp.isoformat(),
                "latency_ms": graph.metadata.latency_ms,
                "input_tokens": graph.metadata.input_tokens,
                "output_tokens": graph.metadata.output_tokens,
                "cost_usd": graph.metadata.cost_usd,
                "from_cache": graph.metadata.from_cache,
                "messages": graph.metadata.messages,
                "temperature": graph.metadata.temperature,
                "max_tokens": graph.metadata.max_tokens,
                "finish_reason": graph.metadata.finish_reason,
                "llm_cost_usd": graph.metadata.initial_cost_usd,
            }
            result["metadata"] = meta_dict

        return result

    def export(self, data: Any, path: Path) -> None:
        """Export graph to JSON and GraphML files.

        Creates two files:
        - {path}.json: Full metadata and edge details
        - {path}.graphml: Graph structure for visualisation

        Args:
            data: GeneratedGraph, tuple (GeneratedGraph, extra_blobs), or dict.
            path: Destination file path (extension will be replaced/added).
        """
        from causaliq_core.graph.io import graphml

        # Handle tuple from decode() which returns (graph, extra_blobs)
        if isinstance(data, tuple) and len(data) == 2:
            data = data[0]

        if isinstance(data, GeneratedGraph):
            graph = data
            export_dict = self._graph_to_export_dict(data)
        else:
            export_dict = data
            graph = self._dict_to_graph(data)

        # Write JSON with full metadata
        json_path = path.with_suffix(".json")
        json_path.write_text(json.dumps(export_dict, indent=2))

        # Write GraphML for graph structure
        sdg = self._graph_to_sdg(graph)
        graphml_path = path.with_suffix(".graphml")
        graphml.write(sdg, str(graphml_path))

    def import_(self, path: Path) -> GeneratedGraph:
        """Import graph from JSON file.

        Args:
            path: Source file path.

        Returns:
            GeneratedGraph instance.
        """
        data = json.loads(path.read_text())
        return self._dict_to_graph(data)
