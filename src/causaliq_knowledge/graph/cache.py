"""
Graph cache compressor for storing generated graphs in Workflow Caches.

This module provides GraphCompressor for compact binary storage of
LLM-generated causal graphs with associated metadata. The compressor uses
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
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from causaliq_core.cache.compressors import JsonCompressor
from causaliq_core.graph import SDG

from causaliq_knowledge.graph.cache_metadata import metadata_from_dict
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


def _read_blob_header(blob: bytes) -> tuple[int, int]:
    """Return the offset after the header and the number of blobs."""
    version = blob[0]
    if version != _FORMAT_VERSION:
        raise ValueError(
            f"Unsupported format version: {version} "
            f"(expected {_FORMAT_VERSION})"
        )
    blob_count = struct.unpack(">H", blob[1:3])[0]
    return 3, blob_count


def _read_blob(
    blob: bytes,
    offset: int,
    cache: "TokenCache",
) -> tuple[str, bytes, int]:
    """Return one blob's type, data, and the offset of the next blob."""
    type_token_id = struct.unpack(">H", blob[offset : offset + 2])[0]
    offset += 2

    blob_len = struct.unpack(">I", blob[offset : offset + 4])[0]
    offset += 4

    blob_data = blob[offset : offset + blob_len]
    offset += blob_len

    # Look up blob type from token
    blob_type = cache.get_token(type_token_id)
    if blob_type is None:
        raise ValueError(f"Unknown blob type token ID: {type_token_id}")

    return blob_type, blob_data, offset


def _extract_sdg(blobs: Dict[str, bytes]) -> SDG:
    """Pop and decompress the graph blob."""
    if BLOB_TYPE_GRAPH not in blobs:
        raise ValueError("No graph blob found in compressed data")
    return SDG.decompress(blobs.pop(BLOB_TYPE_GRAPH))


def _proposed_edge_from_data(edge_data: Any) -> ProposedEdge:
    """Convert one cached edge payload to a ProposedEdge."""
    if isinstance(edge_data, ProposedEdge):
        return edge_data
    return ProposedEdge(
        source=edge_data.get("source", ""),
        target=edge_data.get("target", ""),
        confidence=edge_data.get("confidence", 0.5),
        reasoning=edge_data.get("llm_reasoning", edge_data.get("reasoning")),
    )


def _read_cached_metadata(
    data: Dict[str, Any],
) -> Optional[GenerationMetadata]:
    """Return the metadata recorded in a cached graph payload."""
    if "metadata" not in data or not data["metadata"]:
        return None
    meta = data["metadata"]
    if isinstance(meta, GenerationMetadata):
        return meta
    return metadata_from_dict(meta)


class GraphCompressor(JsonCompressor):
    """Compressor for storing GeneratedGraph objects in Workflow Caches.

    This compressor combines:
    - Compact binary compression for graph structure via SDG.compress()
    - Flexible multi-blob format for additional binary data
    - Tokenised JSON for metadata (provenance, edge confidences)

    The format achieves good compression while preserving all graph
    metadata including edge confidences and generation provenance.

    The multi-blob format allows storing additional binary data alongside
    the graph (e.g., execution traces) without breaking compatibility.

    Example:
        >>> from causaliq_core.cache import TokenCache
        >>> from causaliq_knowledge.graph.cache import GraphCompressor
        >>> from causaliq_knowledge.graph.response import (
        ...     GeneratedGraph, ProposedEdge, GenerationMetadata
        ... )
        >>> with TokenCache(":memory:") as cache:
        ...     compressor = GraphCompressor()
        ...     graph = GeneratedGraph(
        ...         edges=[ProposedEdge(
        ...             source="A", target="B", confidence=0.9
        ...         )],
        ...         variables=["A", "B"],
        ...         reasoning="A causes B",
        ...         metadata=GenerationMetadata(model="test"),
        ...     )
        ...     blob = compressor.compress_entry(graph, cache)
        ...     restored = compressor.decompress_entry(blob, cache)
        ...     assert restored.edges[0].source == "A"

        # Multi-blob compression with additional data:
        >>> with TokenCache(":memory:") as cache:
        ...     compressor = GraphCompressor()
        ...     blobs = {"graph": sdg.compress(), "trace": trace_bytes}
        ...     metadata = {"model": "gpt-4", "variables": ["A", "B"]}
        ...     compressed = compressor.compress_multi(blobs, metadata, cache)
        ...     result_blobs, result_meta = compressor.decompress_multi(
        ...         compressed, cache
        ...     )
    """

    @property
    def default_export_format(self) -> str:
        """Default file extension for exports."""
        return "json"

    # -------------------------------------------------------------------------
    # Multi-blob compression/decompression (flexible format)
    # -------------------------------------------------------------------------

    def compress_multi(
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
            >>> compressed = compressor.compress_multi(blobs, metadata, cache)
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
        meta_blob = super().compress(metadata, cache)
        result.extend(meta_blob)

        return bytes(result)

    def decompress_multi(
        self, blob: bytes, cache: "TokenCache"
    ) -> tuple[Dict[str, bytes], Dict[str, Any]]:
        """Decompress multiple named blobs with shared metadata.

        Args:
            blob: Binary data from cache.
            cache: TokenCache for shared token dictionary.

        Returns:
            Tuple of (blobs dict, metadata dict).

        Raises:
            ValueError: If format version is unsupported or data corrupted.

        Example:
            >>> blobs, meta = compressor.decompress_multi(compressed, cache)
            >>> graph_bytes = blobs.get("graph")
            >>> trace_bytes = blobs.get("trace")
        """
        offset, blob_count = _read_blob_header(blob)

        # Read each blob
        blobs_dict: Dict[str, bytes] = {}
        for _ in range(blob_count):
            blob_type, blob_data, offset = _read_blob(blob, offset, cache)
            blobs_dict[blob_type] = blob_data

        # Decode metadata from remaining bytes
        meta_blob = blob[offset:]
        metadata = super().decompress(meta_blob, cache)

        return blobs_dict, metadata

    # -------------------------------------------------------------------------
    # Graph-specific compression (uses multi-blob format internally)
    # -------------------------------------------------------------------------

    def _graph_to_sdg(self, graph: GeneratedGraph) -> SDG:
        """Convert GeneratedGraph to SDG for binary compression.

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
            "llm_reasoning": graph.reasoning,
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

    def compress_entry(
        self,
        graph: GeneratedGraph,
        cache: "TokenCache",
        extra_blobs: Optional[Dict[str, bytes]] = None,
        include_confidences: bool = True,
    ) -> bytes:
        """Compress a GeneratedGraph to compact binary format.

        Uses the multi-blob format internally, storing the graph structure
        as a "graph" blob and optionally edge confidences as a separate
        "confidences" blob.

        Args:
            graph: The graph to compress.
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
        blobs = {BLOB_TYPE_GRAPH: sdg.compress()}

        # Add confidences blob if requested and present
        if include_confidences:
            confidences = self._build_confidences_dict(graph)
            if confidences:
                # Compress confidences as tokenised JSON
                conf_blob = super().compress(confidences, cache)
                blobs[BLOB_TYPE_CONFIDENCES] = conf_blob

        # Add any extra blobs
        if extra_blobs:
            blobs.update(extra_blobs)

        # Build metadata dict (excludes confidences)
        metadata = self._build_metadata_dict(graph)

        return self.compress_multi(blobs, metadata, cache)

    def decompress_entry(
        self,
        blob: bytes,
        cache: "TokenCache",
    ) -> tuple[GeneratedGraph, Dict[str, bytes]]:
        """Decompress a GeneratedGraph from binary format.

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
        # Decompress using multi-blob format
        blobs, meta_dict = self.decompress_multi(blob, cache)

        # Extract graph blob
        sdg = _extract_sdg(blobs)

        # Extract confidences from blob (if present)
        confidence_map = self._extract_confidences(blobs, cache)

        # Extract reasoning from metadata
        reasoning_map = meta_dict.get("edge_reasoning", {})

        # Build edges from SDG + confidences + reasoning
        edges = self._sdg_to_edges(sdg, confidence_map, reasoning_map)

        # Build generation metadata
        gen_meta = None
        if "generation" in meta_dict:
            gen_meta = metadata_from_dict(meta_dict["generation"])

        graph = GeneratedGraph(
            edges=edges,
            variables=meta_dict.get("variables", list(sdg.nodes)),
            reasoning=meta_dict.get(
                "llm_reasoning", meta_dict.get("reasoning", "")
            ),
            metadata=gen_meta,
        )

        # Return graph and any extra blobs (graph + confidences already popped)
        return graph, blobs

    def _extract_confidences(
        self,
        blobs: Dict[str, bytes],
        cache: "TokenCache",
    ) -> Dict[str, float]:
        """Pop and decompress the confidences blob when present."""
        if BLOB_TYPE_CONFIDENCES not in blobs:
            return {}
        conf_blob = blobs.pop(BLOB_TYPE_CONFIDENCES)
        confidence_map: Dict[str, float] = super().decompress(conf_blob, cache)
        return confidence_map

    def compress(self, data: Any, cache: "TokenCache") -> bytes:
        """Compress data to binary format.

        Args:
            data: The data to compress (GeneratedGraph or dict with edges).
            cache: TokenCache for shared token dictionary.

        Returns:
            Compact binary representation.
        """
        if isinstance(data, GeneratedGraph):
            return self.compress_entry(data, cache)
        elif isinstance(data, dict) and "edges" in data:
            graph = self._dict_to_graph(data)
            return self.compress_entry(graph, cache)
        return super().compress(data, cache)

    def decompress(
        self, blob: bytes, cache: "TokenCache"
    ) -> tuple[GeneratedGraph, Dict[str, bytes]]:
        """Decompress binary data to GeneratedGraph and extra blobs.

        Args:
            blob: Binary data from cache.
            cache: TokenCache for shared token dictionary.

        Returns:
            Tuple of (GeneratedGraph, extra_blobs dict).
        """
        return self.decompress_entry(blob, cache)

    def _dict_to_graph(self, data: Dict[str, Any]) -> GeneratedGraph:
        """Convert dictionary to GeneratedGraph.

        Args:
            data: Dictionary with graph data.

        Returns:
            GeneratedGraph instance.
        """
        edges = [_proposed_edge_from_data(e) for e in data.get("edges", [])]

        metadata = _read_cached_metadata(data)

        return GeneratedGraph(
            edges=edges,
            variables=data.get("variables", []),
            reasoning=data.get("llm_reasoning", data.get("reasoning", "")),
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
                    **({"llm_reasoning": e.reasoning} if e.reasoning else {}),
                }
                for e in graph.edges
            ],
            "variables": graph.variables,
            "llm_reasoning": graph.reasoning,
        }

        if graph.metadata:
            meta_dict: Dict[str, Any] = {
                "llm_model": graph.metadata.model,
                "llm_provider": graph.metadata.provider,
                "timestamp": graph.metadata.timestamp.isoformat(),
                "llm_timestamp": graph.metadata.llm_timestamp.isoformat(),
                "llm_latency_ms": graph.metadata.llm_latency_ms,
                "llm_input_tokens": graph.metadata.input_tokens,
                "llm_output_tokens": graph.metadata.output_tokens,
                "from_cache": graph.metadata.from_cache,
                "llm_messages": graph.metadata.messages,
                "llm_temperature": graph.metadata.temperature,
                "llm_max_tokens": graph.metadata.max_tokens,
                "llm_finish_reason": graph.metadata.finish_reason,
                "llm_cost_usd": graph.metadata.llm_cost_usd,
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

        # Handle tuple from decompress() which returns (graph, extra_blobs)
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
