"""Unit tests for GraphCompressor.

Tests cover compress/decompress round-trips for generated graphs with
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
    GraphCompressor,
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
        compressor = GraphCompressor()

        graph = GeneratedGraph(
            edges=[
                ProposedEdge(source="A", target="B", confidence=0.9),
                ProposedEdge(source="B", target="C", confidence=0.8),
            ],
            variables=["A", "B", "C"],
            reasoning="A causes B, B causes C",
        )

        blob = compressor.compress_entry(graph, cache)
        restored, extra_blobs = compressor.decompress_entry(blob, cache)

        assert restored.variables == ["A", "B", "C"]
        assert len(restored.edges) == 2
        assert restored.reasoning == "A causes B, B causes C"
        assert extra_blobs == {}


# Test confidence values are preserved through round-trip.
def test_encode_decode_preserves_confidence() -> None:
    """Confidence values are preserved exactly."""
    with TokenCache(":memory:") as cache:
        compressor = GraphCompressor()

        graph = GeneratedGraph(
            edges=[
                ProposedEdge(source="X", target="Y", confidence=0.95),
                ProposedEdge(source="Y", target="Z", confidence=0.72),
            ],
            variables=["X", "Y", "Z"],
        )

        blob = compressor.compress_entry(graph, cache)
        restored, _ = compressor.decompress_entry(blob, cache)

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
        compressor = GraphCompressor()

        timestamp = datetime(2026, 2, 6, 10, 30, 0, tzinfo=timezone.utc)
        metadata = GenerationMetadata(
            model="groq/llama-3.1-8b-instant",
            provider="groq",
            timestamp=timestamp,
            llm_timestamp=timestamp,
            llm_latency_ms=1500,
            input_tokens=100,
            output_tokens=50,
            from_cache=False,
        )

        graph = GeneratedGraph(
            edges=[ProposedEdge(source="A", target="B", confidence=0.9)],
            variables=["A", "B"],
            metadata=metadata,
        )

        blob = compressor.compress_entry(graph, cache)
        restored, _ = compressor.decompress_entry(blob, cache)

        assert restored.metadata is not None
        assert restored.metadata.model == "groq/llama-3.1-8b-instant"
        assert restored.metadata.provider == "groq"
        assert restored.metadata.timestamp == timestamp
        assert restored.metadata.llm_latency_ms == 1500
        assert restored.metadata.input_tokens == 100
        assert restored.metadata.output_tokens == 50
        assert restored.metadata.from_cache is False


# Test edge reasoning is preserved.
def test_encode_decode_with_edge_reasoning() -> None:
    """Per-edge reasoning is preserved through round-trip."""
    with TokenCache(":memory:") as cache:
        compressor = GraphCompressor()

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

        blob = compressor.compress_entry(graph, cache)
        restored, _ = compressor.decompress_entry(blob, cache)

        # Build lookup by source
        reasoning_map = {e.source: e.reasoning for e in restored.edges}
        assert "epidemiology" in reasoning_map["smoking"]
        assert "Genetic" in reasoning_map["genetics"]


# Test empty graph with no edges.
def test_encode_decode_empty_graph() -> None:
    """Empty graph (no edges) encodes and decodes correctly."""
    with TokenCache(":memory:") as cache:
        compressor = GraphCompressor()

        graph = GeneratedGraph(
            edges=[],
            variables=["A", "B", "C"],
            reasoning="No causal relationships found",
        )

        blob = compressor.compress_entry(graph, cache)
        restored, _ = compressor.decompress_entry(blob, cache)

        assert restored.variables == ["A", "B", "C"]
        assert len(restored.edges) == 0
        assert restored.reasoning == "No causal relationships found"


# Test low confidence edges are excluded from SDG structure.
def test_low_confidence_edges_excluded_from_sdg() -> None:
    """Edges with confidence < 0.5 are excluded from SDG structure."""
    with TokenCache(":memory:") as cache:
        compressor = GraphCompressor()

        graph = GeneratedGraph(
            edges=[
                ProposedEdge(source="A", target="B", confidence=0.9),
                ProposedEdge(source="B", target="C", confidence=0.3),  # Low
            ],
            variables=["A", "B", "C"],
        )

        # After decode, only high-confidence edges appear (SDG only stores
        # edges >= 0.5). The low-confidence edge is lost.
        blob = compressor.compress_entry(graph, cache)
        restored, _ = compressor.decompress_entry(blob, cache)

        assert len(restored.edges) == 1
        assert restored.edges[0].source == "A"
        assert restored.edges[0].target == "B"


# Test generic encode method with GeneratedGraph.
def test_encode_method_with_generated_graph() -> None:
    """Generic encode() method handles GeneratedGraph objects."""
    with TokenCache(":memory:") as cache:
        compressor = GraphCompressor()

        graph = GeneratedGraph(
            edges=[ProposedEdge(source="A", target="B", confidence=0.9)],
            variables=["A", "B"],
        )

        blob = compressor.compress(graph, cache)
        restored, _ = compressor.decompress(blob, cache)

        assert isinstance(restored, GeneratedGraph)
        assert len(restored.edges) == 1


# Test generic encode method with dict input.
def test_encode_method_with_dict() -> None:
    """Generic encode() method handles dict with edges key."""
    with TokenCache(":memory:") as cache:
        compressor = GraphCompressor()

        data = {
            "edges": [
                {"source": "A", "target": "B", "confidence": 0.9},
            ],
            "variables": ["A", "B"],
            "reasoning": "Test reasoning",
        }

        blob = compressor.compress(data, cache)
        restored, _ = compressor.decompress(blob, cache)

        assert isinstance(restored, GeneratedGraph)
        assert len(restored.edges) == 1
        assert restored.reasoning == "Test reasoning"


# Test export to JSON file.
def test_export_to_json(tmp_path) -> None:
    """Export writes valid JSON file."""
    compressor = GraphCompressor()

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
    compressor.export(graph, output_path)

    assert output_path.exists()
    import json

    data = json.loads(output_path.read_text())
    assert "edges" in data
    assert "variables" in data
    assert data["edges"][0]["source"] == "A"


# Test import from JSON file.
def test_import_from_json(tmp_path) -> None:
    """Import reads JSON file and creates GeneratedGraph."""
    compressor = GraphCompressor()

    json_data = {
        "edges": [
            {"source": "X", "target": "Y", "confidence": 0.85},
        ],
        "variables": ["X", "Y"],
        "reasoning": "Imported graph",
    }

    input_path = tmp_path / "input.json"
    input_path.write_text(__import__("json").dumps(json_data))

    graph = compressor.import_(input_path)

    assert isinstance(graph, GeneratedGraph)
    assert len(graph.edges) == 1
    assert graph.edges[0].source == "X"
    assert graph.edges[0].confidence == 0.85
    assert graph.reasoning == "Imported graph"


# Test round-trip through export and import.
def test_export_import_round_trip(tmp_path) -> None:
    """Export then import preserves graph data."""
    compressor = GraphCompressor()

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
            llm_latency_ms=500,
        ),
    )

    path = tmp_path / "roundtrip.json"
    compressor.export(original, path)
    restored = compressor.import_(path)

    assert restored.variables == original.variables
    assert restored.reasoning == original.reasoning
    assert len(restored.edges) == 1
    assert restored.edges[0].source == "cause"
    assert restored.edges[0].confidence == 0.88


# Test default export format is json.
def test_default_export_format() -> None:
    """Default export format is json."""
    compressor = GraphCompressor()
    assert compressor.default_export_format == "json"


# Test graph with many edges.
def test_encode_decode_many_edges() -> None:
    """Graphs with many edges encode and decode correctly."""
    with TokenCache(":memory:") as cache:
        compressor = GraphCompressor()

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

        blob = compressor.compress_entry(graph, cache)
        restored, _ = compressor.decompress_entry(blob, cache)

        assert len(restored.variables) == 20
        # All edges should be restored (all confidence >= 0.5)
        assert len(restored.edges) == len(edges)


# -------------------------------------------------------------------------
# Multi-blob format tests
# -------------------------------------------------------------------------


# Test compress_multi and decompress_multi round-trip.
def test_encode_decompress_multi_basic() -> None:
    """Multi-blob encode/decode round-trip works."""
    with TokenCache(":memory:") as cache:
        compressor = GraphCompressor()

        blobs = {
            "graph": b"\x01\x02\x03\x04",
            "trace": b"\x05\x06\x07\x08\x09\x0a",
        }
        metadata = {"model": "test", "variables": ["A", "B"]}

        encoded = compressor.compress_multi(blobs, metadata, cache)
        decoded_blobs, decoded_meta = compressor.decompress_multi(
            encoded, cache
        )

        assert decoded_blobs == blobs
        assert decoded_meta == metadata


# Test extra blobs stored alongside graph.
def test_compress_entry_with_extra_blobs() -> None:
    """Extra blobs can be stored alongside graph."""
    with TokenCache(":memory:") as cache:
        compressor = GraphCompressor()

        graph = GeneratedGraph(
            edges=[ProposedEdge(source="A", target="B", confidence=0.9)],
            variables=["A", "B"],
        )

        trace_data = b"execution trace binary data here"
        blob = compressor.compress_entry(
            graph, cache, extra_blobs={BLOB_TYPE_TRACE: trace_data}
        )

        restored, extra_blobs = compressor.decompress_entry(blob, cache)

        assert restored.edges[0].source == "A"
        assert BLOB_TYPE_TRACE in extra_blobs
        assert extra_blobs[BLOB_TYPE_TRACE] == trace_data


# Test multi-blob with empty blobs dict.
def test_compress_multi_empty_blobs() -> None:
    """Multi-blob format works with no blobs (metadata only)."""
    with TokenCache(":memory:") as cache:
        compressor = GraphCompressor()

        blobs: dict[str, bytes] = {}
        metadata = {"info": "metadata only"}

        encoded = compressor.compress_multi(blobs, metadata, cache)
        decoded_blobs, decoded_meta = compressor.decompress_multi(
            encoded, cache
        )

        assert decoded_blobs == {}
        assert decoded_meta == metadata


# Test blob type constants are correct.
def test_blob_type_constants() -> None:
    """Blob type constants have expected values."""
    assert BLOB_TYPE_GRAPH == "graph"
    assert BLOB_TYPE_CONFIDENCES == "confidences"
    assert BLOB_TYPE_TRACE == "trace"


# Test unsupported format version raises error.
def test_decompress_multi_unsupported_version() -> None:
    """Decode raises error for unsupported format version."""
    with TokenCache(":memory:") as cache:
        compressor = GraphCompressor()

        # Create blob with wrong version byte
        bad_blob = b"\xff\x00\x00"  # Version 255, 0 blobs

        with pytest.raises(ValueError, match="Unsupported format version"):
            compressor.decompress_multi(bad_blob, cache)


# Test decompress_multi with unknown blob type token ID.
def test_decompress_multi_unknown_blob_type_token() -> None:
    """Decode raises error when blob type token ID is unknown."""
    import struct

    with TokenCache(":memory:") as cache:
        compressor = GraphCompressor()

        # Create a blob with valid format but invalid token ID (65535)
        # Header: version 1, 1 blob
        # Blob: type token ID 65535 (unlikely to exist), length 4, data
        header = b"\x01" + struct.pack(">H", 1)  # Version 1, 1 blob
        blob_header = struct.pack(">H", 65535)  # Invalid token ID
        blob_header += struct.pack(">I", 4)  # Length 4
        blob_data = b"\x00\x00\x00\x00"  # 4 bytes of data
        # Metadata (empty dict encoded as JSON token)
        meta = compressor.compress({}, cache)

        bad_blob = header + blob_header + blob_data + meta

        with pytest.raises(ValueError, match="Unknown blob type token ID"):
            compressor.decompress_multi(bad_blob, cache)


# -------------------------------------------------------------------------
# Confidences blob tests
# -------------------------------------------------------------------------


# Test confidences stored in separate blob.
def test_confidences_in_separate_blob() -> None:
    """Confidences are stored in a separate blob, not metadata."""
    with TokenCache(":memory:") as cache:
        compressor = GraphCompressor()

        graph = GeneratedGraph(
            edges=[
                ProposedEdge(source="A", target="B", confidence=0.9),
                ProposedEdge(source="B", target="C", confidence=0.75),
            ],
            variables=["A", "B", "C"],
        )

        blob = compressor.compress_entry(graph, cache)

        # Decode raw blobs to verify confidences is separate
        blobs, metadata = compressor.decompress_multi(blob, cache)

        assert BLOB_TYPE_CONFIDENCES in blobs
        assert "confidences" not in metadata  # Not in metadata


# Test encoding without confidences (structure learning scenario).
def test_encode_without_confidences() -> None:
    """Graphs can be encoded without confidences blob."""
    with TokenCache(":memory:") as cache:
        compressor = GraphCompressor()

        # Structure learning result - no meaningful confidences
        graph = GeneratedGraph(
            edges=[
                ProposedEdge(source="A", target="B", confidence=0.5),
                ProposedEdge(source="B", target="C", confidence=0.5),
            ],
            variables=["A", "B", "C"],
        )

        # Compress with include_confidences=False
        blob = compressor.compress_entry(
            graph, cache, include_confidences=False
        )

        # Decompress raw to verify no confidences blob
        blobs, _ = compressor.decompress_multi(blob, cache)
        assert BLOB_TYPE_CONFIDENCES not in blobs

        # Full decompress still works, edges get default confidence
        restored, _ = compressor.decompress_entry(blob, cache)
        assert len(restored.edges) == 2
        assert restored.edges[0].confidence == 0.5


# Test default confidences omitted when all edges have default value.
def test_default_confidences_omitted() -> None:
    """Confidences blob omitted when edges have default (0.5) confidence."""
    with TokenCache(":memory:") as cache:
        compressor = GraphCompressor()

        # All edges have default confidence
        graph = GeneratedGraph(
            edges=[
                ProposedEdge(source="A", target="B", confidence=0.5),
            ],
            variables=["A", "B"],
        )

        blob = compressor.compress_entry(graph, cache)

        # Confidences blob should be omitted (no non-default values)
        blobs, _ = compressor.decompress_multi(blob, cache)
        assert BLOB_TYPE_CONFIDENCES not in blobs


# -------------------------------------------------------------------------
# Edge case and error handling tests
# -------------------------------------------------------------------------


# Test decompress_entry raises error when no graph blob present.
def test_decompress_entry_missing_graph_blob() -> None:
    """Decode raises error when graph blob is missing."""
    with TokenCache(":memory:") as cache:
        compressor = GraphCompressor()

        # Encode with only metadata, no graph blob
        blobs: dict[str, bytes] = {"other": b"\x01\x02\x03"}
        metadata = {"info": "no graph here"}
        encoded = compressor.compress_multi(blobs, metadata, cache)

        with pytest.raises(ValueError, match="No graph blob found"):
            compressor.decompress_entry(encoded, cache)


# Test encode with plain dict falls back to JSON encoding.
def test_encode_plain_dict_fallback() -> None:
    """Encode falls back to JSON for dicts without 'edges' key."""
    with TokenCache(":memory:") as cache:
        compressor = GraphCompressor()

        # Plain dict without 'edges' - should use JSON encoding
        data = {"key": "value", "count": 42}
        blob = compressor.compress(data, cache)

        # Decode as JSON (not via decompress_entry)
        from causaliq_core.cache.compressors import JsonCompressor

        json_compressor = JsonCompressor()
        decoded = json_compressor.decompress(blob, cache)
        assert decoded == data


# Test _dict_to_graph with ProposedEdge objects in list.
def test_dict_to_graph_with_proposed_edge_objects() -> None:
    """Dict with ProposedEdge objects is handled correctly."""
    with TokenCache(":memory:") as cache:
        compressor = GraphCompressor()

        # Dict with ProposedEdge objects (not dicts)
        data = {
            "edges": [
                ProposedEdge(source="A", target="B", confidence=0.9),
            ],
            "variables": ["A", "B"],
            "reasoning": "Test",
        }

        blob = compressor.compress(data, cache)
        restored, _ = compressor.decompress(blob, cache)

        assert restored.edges[0].source == "A"
        assert restored.edges[0].confidence == 0.9


# Test _dict_to_graph with GenerationMetadata object.
def test_dict_to_graph_with_metadata_object() -> None:
    """Dict with GenerationMetadata object is handled correctly."""
    with TokenCache(":memory:") as cache:
        compressor = GraphCompressor()

        meta = GenerationMetadata(
            model="test-model",
            provider="test",
            llm_latency_ms=100,
        )

        data = {
            "edges": [{"source": "A", "target": "B", "confidence": 0.8}],
            "variables": ["A", "B"],
            "metadata": meta,
        }

        blob = compressor.compress(data, cache)
        restored, _ = compressor.decompress(blob, cache)

        assert restored.metadata is not None
        assert restored.metadata.model == "test-model"


# Test export with dict input (not GeneratedGraph).
def test_export_with_dict_input(tmp_path) -> None:
    """Export handles dict input directly."""
    compressor = GraphCompressor()

    # Pass dict directly to export
    data = {
        "edges": [{"source": "X", "target": "Y", "confidence": 0.7}],
        "variables": ["X", "Y"],
        "llm_reasoning": "Dict export test",
    }

    output_path = tmp_path / "dict_export.json"
    compressor.export(data, output_path)

    assert output_path.exists()
    import json

    exported = json.loads(output_path.read_text())
    assert exported["edges"][0]["source"] == "X"
    assert exported["llm_reasoning"] == "Dict export test"


# Test metadata with llm_cost_usd field.
def test_encode_decode_with_llm_cost() -> None:
    """llm_cost_usd is preserved through round-trip."""
    with TokenCache(":memory:") as cache:
        compressor = GraphCompressor()

        llm_time = datetime(2026, 2, 9, 10, 0, 0, tzinfo=timezone.utc)

        metadata = GenerationMetadata(
            model="test-model",
            provider="test-provider",
            timestamp=llm_time,
            llm_timestamp=llm_time,
            llm_latency_ms=1000,
            input_tokens=100,
            output_tokens=50,
            llm_cost_usd=0.005,
        )

        graph = GeneratedGraph(
            edges=[ProposedEdge(source="A", target="B", confidence=0.9)],
            variables=["A", "B"],
            metadata=metadata,
        )

        blob = compressor.compress_entry(graph, cache)
        restored, _ = compressor.decompress_entry(blob, cache)

        assert restored.metadata is not None
        assert restored.metadata.timestamp == llm_time
        assert restored.metadata.llm_cost_usd == 0.005


# Test export includes llm_timestamp and llm_cost_usd in JSON.
def test_export_includes_llm_fields(tmp_path) -> None:
    """Export dict includes llm_timestamp and llm_cost_usd."""
    compressor = GraphCompressor()

    llm_time = datetime(2026, 2, 9, 10, 0, 0, tzinfo=timezone.utc)

    metadata = GenerationMetadata(
        model="test-model",
        provider="test",
        timestamp=llm_time,
        llm_timestamp=llm_time,
        llm_latency_ms=2000,
        llm_cost_usd=0.01,
    )

    graph = GeneratedGraph(
        edges=[ProposedEdge(source="X", target="Y", confidence=0.85)],
        variables=["X", "Y"],
        metadata=metadata,
    )

    path = tmp_path / "llm_fields_test.json"
    compressor.export(graph, path)

    import json

    exported = json.loads(path.read_text())

    assert "metadata" in exported
    assert exported["metadata"]["llm_timestamp"] == "2026-02-09T10:00:00+00:00"
    assert exported["metadata"]["llm_cost_usd"] == 0.01


# Test export with dict input creates GraphML file.
def test_export_dict_creates_graphml(tmp_path) -> None:
    """Export with dict input creates both JSON and GraphML files."""
    compressor = GraphCompressor()

    data = {
        "edges": [
            {"source": "A", "target": "B", "confidence": 0.9},
            {"source": "B", "target": "C", "confidence": 0.8},
        ],
        "variables": ["A", "B", "C"],
        "reasoning": "Dict export GraphML test",
    }

    path = tmp_path / "dict_graphml_test"
    compressor.export(data, path)

    # Both files should exist
    json_path = path.with_suffix(".json")
    graphml_path = path.with_suffix(".graphml")

    assert json_path.exists()
    assert graphml_path.exists()

    # GraphML file should contain graph structure
    graphml_content = graphml_path.read_text()
    assert "<graphml" in graphml_content
    assert "A" in graphml_content
    assert "B" in graphml_content


# Test export with tuple input (from decode()).
def test_export_tuple_from_decode(tmp_path) -> None:
    """Export handles tuple from decode() which returns
    (graph, extra_blobs)."""
    compressor = GraphCompressor()

    graph = GeneratedGraph(
        edges=[ProposedEdge(source="P", target="Q", confidence=0.75)],
        variables=["P", "Q"],
        reasoning="Tuple export test",
    )

    # Simulate decode() return value: (graph, extra_blobs)
    data_tuple = (graph, {"trace": b"some trace data"})

    path = tmp_path / "tuple_export"
    compressor.export(data_tuple, path)

    # Both files should exist
    json_path = path.with_suffix(".json")
    graphml_path = path.with_suffix(".graphml")

    assert json_path.exists()
    assert graphml_path.exists()

    import json

    exported = json.loads(json_path.read_text())
    assert exported["edges"][0]["source"] == "P"
    assert exported["llm_reasoning"] == "Tuple export test"
