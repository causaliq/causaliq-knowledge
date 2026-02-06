"""Unit tests for GraphEntryEncoder.

Tests cover encoding/decoding round-trips for generated graphs with
various configurations of edges, metadata, and confidences. Also tests
the multi-blob format for storing additional binary data.

Confidences are stored in a separate blob, not in metadata. This allows
structure learning workflows (which don't have confidence scores) to
omit the confidences blob entirely.
"""

from datetime import datetime, timezone

import pytest
from causaliq_core.cache import TokenCache

from causaliq_knowledge.graph.cache import (
    BLOB_TYPE_CONFIDENCES,
    BLOB_TYPE_GRAPH,
    BLOB_TYPE_TRACE,
    GraphEntryEncoder,
)
from causaliq_knowledge.graph.response import (
    GeneratedGraph,
    GenerationMetadata,
    ProposedEdge,
)


# Test basic encode/decode round-trip with simple graph.
def test_encode_decode_simple_graph() -> None:
    """Round-trip encode/decode preserves simple graph structure."""
    with TokenCache(":memory:") as cache:
        encoder = GraphEntryEncoder()

        graph = GeneratedGraph(
            edges=[
                ProposedEdge(source="A", target="B", confidence=0.9),
                ProposedEdge(source="B", target="C", confidence=0.8),
            ],
            variables=["A", "B", "C"],
            reasoning="A causes B, B causes C",
        )

        blob = encoder.encode_entry(graph, cache)
        restored, extra_blobs = encoder.decode_entry(blob, cache)

        assert restored.variables == ["A", "B", "C"]
        assert len(restored.edges) == 2
        assert restored.reasoning == "A causes B, B causes C"
        assert extra_blobs == {}


# Test confidence values are preserved through round-trip.
def test_encode_decode_preserves_confidence() -> None:
    """Confidence values are preserved exactly."""
    with TokenCache(":memory:") as cache:
        encoder = GraphEntryEncoder()

        graph = GeneratedGraph(
            edges=[
                ProposedEdge(source="X", target="Y", confidence=0.95),
                ProposedEdge(source="Y", target="Z", confidence=0.72),
            ],
            variables=["X", "Y", "Z"],
        )

        blob = encoder.encode_entry(graph, cache)
        restored, _ = encoder.decode_entry(blob, cache)

        # Build lookup by source->target
        confidence_map = {
            f"{e.source}->{e.target}": e.confidence for e in restored.edges
        }
        assert confidence_map["X->Y"] == 0.95
        assert confidence_map["Y->Z"] == 0.72


# Test generation metadata is preserved.
def test_encode_decode_with_metadata() -> None:
    """Generation metadata is preserved through round-trip."""
    with TokenCache(":memory:") as cache:
        encoder = GraphEntryEncoder()

        timestamp = datetime(2026, 2, 6, 10, 30, 0, tzinfo=timezone.utc)
        metadata = GenerationMetadata(
            model="groq/llama-3.1-8b-instant",
            provider="groq",
            timestamp=timestamp,
            latency_ms=1500,
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.001,
            from_cache=False,
        )

        graph = GeneratedGraph(
            edges=[ProposedEdge(source="A", target="B", confidence=0.9)],
            variables=["A", "B"],
            metadata=metadata,
        )

        blob = encoder.encode_entry(graph, cache)
        restored, _ = encoder.decode_entry(blob, cache)

        assert restored.metadata is not None
        assert restored.metadata.model == "groq/llama-3.1-8b-instant"
        assert restored.metadata.provider == "groq"
        assert restored.metadata.timestamp == timestamp
        assert restored.metadata.latency_ms == 1500
        assert restored.metadata.input_tokens == 100
        assert restored.metadata.output_tokens == 50
        assert restored.metadata.cost_usd == 0.001
        assert restored.metadata.from_cache is False


# Test edge reasoning is preserved.
def test_encode_decode_with_edge_reasoning() -> None:
    """Per-edge reasoning is preserved through round-trip."""
    with TokenCache(":memory:") as cache:
        encoder = GraphEntryEncoder()

        graph = GeneratedGraph(
            edges=[
                ProposedEdge(
                    source="smoking",
                    target="lung_cancer",
                    confidence=0.95,
                    reasoning="Established causal link from epidemiology",
                ),
                ProposedEdge(
                    source="genetics",
                    target="lung_cancer",
                    confidence=0.8,
                    reasoning="Genetic predisposition increases risk",
                ),
            ],
            variables=["smoking", "genetics", "lung_cancer"],
        )

        blob = encoder.encode_entry(graph, cache)
        restored, _ = encoder.decode_entry(blob, cache)

        # Build lookup by source
        reasoning_map = {e.source: e.reasoning for e in restored.edges}
        assert "epidemiology" in reasoning_map["smoking"]
        assert "Genetic" in reasoning_map["genetics"]


# Test empty graph with no edges.
def test_encode_decode_empty_graph() -> None:
    """Empty graph (no edges) encodes and decodes correctly."""
    with TokenCache(":memory:") as cache:
        encoder = GraphEntryEncoder()

        graph = GeneratedGraph(
            edges=[],
            variables=["A", "B", "C"],
            reasoning="No causal relationships found",
        )

        blob = encoder.encode_entry(graph, cache)
        restored, _ = encoder.decode_entry(blob, cache)

        assert restored.variables == ["A", "B", "C"]
        assert len(restored.edges) == 0
        assert restored.reasoning == "No causal relationships found"


# Test low confidence edges are excluded from SDG structure.
def test_low_confidence_edges_excluded_from_sdg() -> None:
    """Edges with confidence < 0.5 are excluded from SDG structure."""
    with TokenCache(":memory:") as cache:
        encoder = GraphEntryEncoder()

        graph = GeneratedGraph(
            edges=[
                ProposedEdge(source="A", target="B", confidence=0.9),
                ProposedEdge(source="B", target="C", confidence=0.3),  # Low
            ],
            variables=["A", "B", "C"],
        )

        # After decode, only high-confidence edges appear (SDG only stores
        # edges >= 0.5). The low-confidence edge is lost.
        blob = encoder.encode_entry(graph, cache)
        restored, _ = encoder.decode_entry(blob, cache)

        assert len(restored.edges) == 1
        assert restored.edges[0].source == "A"
        assert restored.edges[0].target == "B"


# Test generic encode method with GeneratedGraph.
def test_encode_method_with_generated_graph() -> None:
    """Generic encode() method handles GeneratedGraph objects."""
    with TokenCache(":memory:") as cache:
        encoder = GraphEntryEncoder()

        graph = GeneratedGraph(
            edges=[ProposedEdge(source="A", target="B", confidence=0.9)],
            variables=["A", "B"],
        )

        blob = encoder.encode(graph, cache)
        restored, _ = encoder.decode(blob, cache)

        assert isinstance(restored, GeneratedGraph)
        assert len(restored.edges) == 1


# Test generic encode method with dict input.
def test_encode_method_with_dict() -> None:
    """Generic encode() method handles dict with edges key."""
    with TokenCache(":memory:") as cache:
        encoder = GraphEntryEncoder()

        data = {
            "edges": [
                {"source": "A", "target": "B", "confidence": 0.9},
            ],
            "variables": ["A", "B"],
            "reasoning": "Test reasoning",
        }

        blob = encoder.encode(data, cache)
        restored, _ = encoder.decode(blob, cache)

        assert isinstance(restored, GeneratedGraph)
        assert len(restored.edges) == 1
        assert restored.reasoning == "Test reasoning"


# Test export to JSON file.
def test_export_to_json(tmp_path) -> None:
    """Export writes valid JSON file."""
    encoder = GraphEntryEncoder()

    graph = GeneratedGraph(
        edges=[
            ProposedEdge(source="A", target="B", confidence=0.9),
        ],
        variables=["A", "B"],
        reasoning="Test",
        metadata=GenerationMetadata(
            model="test-model",
            provider="test",
        ),
    )

    output_path = tmp_path / "graph.json"
    encoder.export(graph, output_path)

    assert output_path.exists()
    import json

    data = json.loads(output_path.read_text())
    assert "edges" in data
    assert "variables" in data
    assert data["edges"][0]["source"] == "A"


# Test import from JSON file.
def test_import_from_json(tmp_path) -> None:
    """Import reads JSON file and creates GeneratedGraph."""
    encoder = GraphEntryEncoder()

    json_data = {
        "edges": [
            {"source": "X", "target": "Y", "confidence": 0.85},
        ],
        "variables": ["X", "Y"],
        "reasoning": "Imported graph",
    }

    input_path = tmp_path / "input.json"
    input_path.write_text(__import__("json").dumps(json_data))

    graph = encoder.import_(input_path)

    assert isinstance(graph, GeneratedGraph)
    assert len(graph.edges) == 1
    assert graph.edges[0].source == "X"
    assert graph.edges[0].confidence == 0.85
    assert graph.reasoning == "Imported graph"


# Test round-trip through export and import.
def test_export_import_round_trip(tmp_path) -> None:
    """Export then import preserves graph data."""
    encoder = GraphEntryEncoder()

    original = GeneratedGraph(
        edges=[
            ProposedEdge(
                source="cause",
                target="effect",
                confidence=0.88,
                reasoning="Direct causation",
            ),
        ],
        variables=["cause", "effect"],
        reasoning="Simple causal model",
        metadata=GenerationMetadata(
            model="test/model",
            provider="test",
            latency_ms=500,
        ),
    )

    path = tmp_path / "roundtrip.json"
    encoder.export(original, path)
    restored = encoder.import_(path)

    assert restored.variables == original.variables
    assert restored.reasoning == original.reasoning
    assert len(restored.edges) == 1
    assert restored.edges[0].source == "cause"
    assert restored.edges[0].confidence == 0.88


# Test default export format is json.
def test_default_export_format() -> None:
    """Default export format is json."""
    encoder = GraphEntryEncoder()
    assert encoder.default_export_format == "json"


# Test graph with many edges.
def test_encode_decode_many_edges() -> None:
    """Graphs with many edges encode and decode correctly."""
    with TokenCache(":memory:") as cache:
        encoder = GraphEntryEncoder()

        # Create a graph with 20 nodes and ~50 edges
        variables = [f"V{i}" for i in range(20)]
        edges = []
        for i in range(19):
            for j in range(i + 1, min(i + 4, 20)):
                edges.append(
                    ProposedEdge(
                        source=f"V{i}",
                        target=f"V{j}",
                        confidence=0.5 + (i + j) * 0.01,
                    )
                )

        graph = GeneratedGraph(
            edges=edges,
            variables=variables,
            reasoning="Complex network",
        )

        blob = encoder.encode_entry(graph, cache)
        restored, _ = encoder.decode_entry(blob, cache)

        assert len(restored.variables) == 20
        # All edges should be restored (all confidence >= 0.5)
        assert len(restored.edges) == len(edges)


# -------------------------------------------------------------------------
# Multi-blob format tests
# -------------------------------------------------------------------------


# Test encode_multi and decode_multi round-trip.
def test_encode_decode_multi_basic() -> None:
    """Multi-blob encode/decode round-trip works."""
    with TokenCache(":memory:") as cache:
        encoder = GraphEntryEncoder()

        blobs = {
            "graph": b"\x01\x02\x03\x04",
            "trace": b"\x05\x06\x07\x08\x09\x0a",
        }
        metadata = {"model": "test", "variables": ["A", "B"]}

        encoded = encoder.encode_multi(blobs, metadata, cache)
        decoded_blobs, decoded_meta = encoder.decode_multi(encoded, cache)

        assert decoded_blobs == blobs
        assert decoded_meta == metadata


# Test extra blobs stored alongside graph.
def test_encode_entry_with_extra_blobs() -> None:
    """Extra blobs can be stored alongside graph."""
    with TokenCache(":memory:") as cache:
        encoder = GraphEntryEncoder()

        graph = GeneratedGraph(
            edges=[ProposedEdge(source="A", target="B", confidence=0.9)],
            variables=["A", "B"],
        )

        trace_data = b"execution trace binary data here"
        blob = encoder.encode_entry(
            graph, cache, extra_blobs={BLOB_TYPE_TRACE: trace_data}
        )

        restored, extra_blobs = encoder.decode_entry(blob, cache)

        assert restored.edges[0].source == "A"
        assert BLOB_TYPE_TRACE in extra_blobs
        assert extra_blobs[BLOB_TYPE_TRACE] == trace_data


# Test multi-blob with empty blobs dict.
def test_encode_multi_empty_blobs() -> None:
    """Multi-blob format works with no blobs (metadata only)."""
    with TokenCache(":memory:") as cache:
        encoder = GraphEntryEncoder()

        blobs: dict[str, bytes] = {}
        metadata = {"info": "metadata only"}

        encoded = encoder.encode_multi(blobs, metadata, cache)
        decoded_blobs, decoded_meta = encoder.decode_multi(encoded, cache)

        assert decoded_blobs == {}
        assert decoded_meta == metadata


# Test blob type constants are correct.
def test_blob_type_constants() -> None:
    """Blob type constants have expected values."""
    assert BLOB_TYPE_GRAPH == "graph"
    assert BLOB_TYPE_CONFIDENCES == "confidences"
    assert BLOB_TYPE_TRACE == "trace"


# Test unsupported format version raises error.
def test_decode_multi_unsupported_version() -> None:
    """Decode raises error for unsupported format version."""
    with TokenCache(":memory:") as cache:
        encoder = GraphEntryEncoder()

        # Create blob with wrong version byte
        bad_blob = b"\xff\x00\x00"  # Version 255, 0 blobs

        with pytest.raises(ValueError, match="Unsupported format version"):
            encoder.decode_multi(bad_blob, cache)


# Test decode_multi with unknown blob type token ID.
def test_decode_multi_unknown_blob_type_token() -> None:
    """Decode raises error when blob type token ID is unknown."""
    import struct

    with TokenCache(":memory:") as cache:
        encoder = GraphEntryEncoder()

        # Create a blob with valid format but invalid token ID (65535)
        # Header: version 1, 1 blob
        # Blob: type token ID 65535 (unlikely to exist), length 4, data
        header = b"\x01" + struct.pack(">H", 1)  # Version 1, 1 blob
        blob_header = struct.pack(">H", 65535)  # Invalid token ID
        blob_header += struct.pack(">I", 4)  # Length 4
        blob_data = b"\x00\x00\x00\x00"  # 4 bytes of data
        # Metadata (empty dict encoded as JSON token)
        meta = encoder.encode({}, cache)

        bad_blob = header + blob_header + blob_data + meta

        with pytest.raises(ValueError, match="Unknown blob type token ID"):
            encoder.decode_multi(bad_blob, cache)


# -------------------------------------------------------------------------
# Confidences blob tests
# -------------------------------------------------------------------------


# Test confidences stored in separate blob.
def test_confidences_in_separate_blob() -> None:
    """Confidences are stored in a separate blob, not metadata."""
    with TokenCache(":memory:") as cache:
        encoder = GraphEntryEncoder()

        graph = GeneratedGraph(
            edges=[
                ProposedEdge(source="A", target="B", confidence=0.9),
                ProposedEdge(source="B", target="C", confidence=0.75),
            ],
            variables=["A", "B", "C"],
        )

        blob = encoder.encode_entry(graph, cache)

        # Decode raw blobs to verify confidences is separate
        blobs, metadata = encoder.decode_multi(blob, cache)

        assert BLOB_TYPE_CONFIDENCES in blobs
        assert "confidences" not in metadata  # Not in metadata


# Test encoding without confidences (structure learning scenario).
def test_encode_without_confidences() -> None:
    """Graphs can be encoded without confidences blob."""
    with TokenCache(":memory:") as cache:
        encoder = GraphEntryEncoder()

        # Structure learning result - no meaningful confidences
        graph = GeneratedGraph(
            edges=[
                ProposedEdge(source="A", target="B", confidence=0.5),
                ProposedEdge(source="B", target="C", confidence=0.5),
            ],
            variables=["A", "B", "C"],
        )

        # Encode with include_confidences=False
        blob = encoder.encode_entry(graph, cache, include_confidences=False)

        # Decode raw to verify no confidences blob
        blobs, _ = encoder.decode_multi(blob, cache)
        assert BLOB_TYPE_CONFIDENCES not in blobs

        # Full decode still works, edges get default confidence
        restored, _ = encoder.decode_entry(blob, cache)
        assert len(restored.edges) == 2
        assert restored.edges[0].confidence == 0.5


# Test default confidences omitted when all edges have default value.
def test_default_confidences_omitted() -> None:
    """Confidences blob omitted when edges have default (0.5) confidence."""
    with TokenCache(":memory:") as cache:
        encoder = GraphEntryEncoder()

        # All edges have default confidence
        graph = GeneratedGraph(
            edges=[
                ProposedEdge(source="A", target="B", confidence=0.5),
            ],
            variables=["A", "B"],
        )

        blob = encoder.encode_entry(graph, cache)

        # Confidences blob should be omitted (no non-default values)
        blobs, _ = encoder.decode_multi(blob, cache)
        assert BLOB_TYPE_CONFIDENCES not in blobs


# -------------------------------------------------------------------------
# Edge case and error handling tests
# -------------------------------------------------------------------------


# Test decode_entry raises error when no graph blob present.
def test_decode_entry_missing_graph_blob() -> None:
    """Decode raises error when graph blob is missing."""
    with TokenCache(":memory:") as cache:
        encoder = GraphEntryEncoder()

        # Encode with only metadata, no graph blob
        blobs: dict[str, bytes] = {"other": b"\x01\x02\x03"}
        metadata = {"info": "no graph here"}
        encoded = encoder.encode_multi(blobs, metadata, cache)

        with pytest.raises(ValueError, match="No graph blob found"):
            encoder.decode_entry(encoded, cache)


# Test encode with plain dict falls back to JSON encoding.
def test_encode_plain_dict_fallback() -> None:
    """Encode falls back to JSON for dicts without 'edges' key."""
    with TokenCache(":memory:") as cache:
        encoder = GraphEntryEncoder()

        # Plain dict without 'edges' - should use JSON encoding
        data = {"key": "value", "count": 42}
        blob = encoder.encode(data, cache)

        # Decode as JSON (not via decode_entry)
        from causaliq_core.cache.encoders import JsonEncoder

        json_encoder = JsonEncoder()
        decoded = json_encoder.decode(blob, cache)
        assert decoded == data


# Test _dict_to_graph with ProposedEdge objects in list.
def test_dict_to_graph_with_proposed_edge_objects() -> None:
    """Dict with ProposedEdge objects is handled correctly."""
    with TokenCache(":memory:") as cache:
        encoder = GraphEntryEncoder()

        # Dict with ProposedEdge objects (not dicts)
        data = {
            "edges": [
                ProposedEdge(source="A", target="B", confidence=0.9),
            ],
            "variables": ["A", "B"],
            "reasoning": "Test",
        }

        blob = encoder.encode(data, cache)
        restored, _ = encoder.decode(blob, cache)

        assert restored.edges[0].source == "A"
        assert restored.edges[0].confidence == 0.9


# Test _dict_to_graph with GenerationMetadata object.
def test_dict_to_graph_with_metadata_object() -> None:
    """Dict with GenerationMetadata object is handled correctly."""
    with TokenCache(":memory:") as cache:
        encoder = GraphEntryEncoder()

        meta = GenerationMetadata(
            model="test-model",
            provider="test",
            latency_ms=100,
        )

        data = {
            "edges": [{"source": "A", "target": "B", "confidence": 0.8}],
            "variables": ["A", "B"],
            "metadata": meta,
        }

        blob = encoder.encode(data, cache)
        restored, _ = encoder.decode(blob, cache)

        assert restored.metadata is not None
        assert restored.metadata.model == "test-model"


# Test export with dict input (not GeneratedGraph).
def test_export_with_dict_input(tmp_path) -> None:
    """Export handles dict input directly."""
    encoder = GraphEntryEncoder()

    # Pass dict directly to export
    data = {
        "edges": [{"source": "X", "target": "Y", "confidence": 0.7}],
        "variables": ["X", "Y"],
        "reasoning": "Dict export test",
    }

    output_path = tmp_path / "dict_export.json"
    encoder.export(data, output_path)

    assert output_path.exists()
    import json

    exported = json.loads(output_path.read_text())
    assert exported["edges"][0]["source"] == "X"
    assert exported["reasoning"] == "Dict export test"
