"""Tests for network context loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from causaliq_knowledge.graph.models import (
    NetworkContext,
    NetworkLoadError,
    VariableSpec,
    VariableType,
)

# --- NetworkContext.load() tests ---


# Test loading a valid JSON file.
def test_load_valid_json(tmp_path: Path) -> None:
    data = {
        "schema_version": "2.0",
        "network": "test",
        "domain": "test_domain",
        "variables": [
            {"name": "a", "type": "binary", "states": ["no", "yes"]},
            {"name": "b", "type": "binary", "states": ["no", "yes"]},
        ],
    }
    json_file = tmp_path / "test.json"
    json_file.write_text(json.dumps(data))

    spec = NetworkContext.load(json_file)

    assert isinstance(spec, NetworkContext)
    assert spec.network == "test"
    assert spec.domain == "test_domain"
    assert len(spec.variables) == 2


# Test loading non-existent file raises error.
def test_load_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(NetworkLoadError) as exc_info:
        NetworkContext.load(tmp_path / "nonexistent.json")

    assert "not found" in str(exc_info.value).lower()


# Test loading non-JSON file raises error.
def test_load_wrong_extension(tmp_path: Path) -> None:
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("not json")

    with pytest.raises(NetworkLoadError) as exc_info:
        NetworkContext.load(txt_file)

    assert "JSON file" in str(exc_info.value)


# Test loading invalid JSON raises error.
def test_load_invalid_json(tmp_path: Path) -> None:
    json_file = tmp_path / "invalid.json"
    json_file.write_text("{invalid json}")

    with pytest.raises(NetworkLoadError) as exc_info:
        NetworkContext.load(json_file)

    assert "Invalid JSON" in str(exc_info.value)


# Test loading file with read permission error raises OSError.
def test_load_os_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    json_file = tmp_path / "test.json"
    json_file.write_text('{"network": "test", "domain": "test"}')

    def mock_open(*args, **kwargs):
        raise OSError("Permission denied")

    monkeypatch.setattr("builtins.open", mock_open)

    with pytest.raises(NetworkLoadError) as exc_info:
        NetworkContext.load(json_file)

    assert "Failed to read" in str(exc_info.value)


# Test loading JSON missing required fields raises error.
def test_load_missing_required_fields(tmp_path: Path) -> None:
    data = {"schema_version": "2.0"}  # Missing network and domain
    json_file = tmp_path / "incomplete.json"
    json_file.write_text(json.dumps(data))

    with pytest.raises(NetworkLoadError) as exc_info:
        NetworkContext.load(json_file)

    assert "Missing required fields" in str(exc_info.value)


# --- NetworkContext.from_dict() tests ---


# Test creating spec from minimal dict.
def test_from_dict_minimal() -> None:
    data = {
        "network": "test",
        "domain": "test_domain",
    }
    spec = NetworkContext.from_dict(data)

    assert spec.network == "test"
    assert spec.domain == "test_domain"


# Test creating spec with variables from dict.
def test_from_dict_with_variables() -> None:
    data = {
        "network": "test",
        "domain": "test_domain",
        "variables": [
            {
                "name": "smoking",
                "type": "binary",
                "states": ["no", "yes"],
                "role": "exogenous",
            },
        ],
    }
    spec = NetworkContext.from_dict(data)

    assert len(spec.variables) == 1
    assert spec.variables[0].name == "smoking"
    assert spec.variables[0].type == VariableType.BINARY


# Test from_dict raises error for missing fields.
def test_from_dict_missing_required() -> None:
    data = {"schema_version": "2.0"}

    with pytest.raises(NetworkLoadError) as exc_info:
        NetworkContext.from_dict(data)

    assert "Missing required fields" in str(exc_info.value)


# Test from_dict raises error for invalid variable.
def test_from_dict_invalid_variable() -> None:
    data = {
        "network": "test",
        "domain": "test_domain",
        "variables": [{"name": "a"}],  # Missing type
    }

    with pytest.raises(NetworkLoadError):
        NetworkContext.from_dict(data)


# --- NetworkContext.validate_variables() tests ---


# Test validation fails for empty variables.
def test_validate_empty_variables() -> None:
    spec = NetworkContext(network="test", domain="test")

    with pytest.raises(NetworkLoadError) as exc_info:
        spec.validate_variables()

    assert "no variables" in str(exc_info.value).lower()


# Test validation fails for duplicate variable names.
def test_validate_duplicate_names() -> None:
    spec = NetworkContext(
        network="test",
        domain="test",
        variables=[
            VariableSpec(name="a", type="binary"),
            VariableSpec(name="a", type="binary"),  # Duplicate
        ],
    )

    with pytest.raises(NetworkLoadError) as exc_info:
        spec.validate_variables()

    assert "Duplicate" in str(exc_info.value)


# Test validation warns about missing states for discrete variables.
def test_validate_warns_missing_states() -> None:
    spec = NetworkContext(
        network="test",
        domain="test",
        variables=[
            VariableSpec(name="a", type="binary"),  # No states
        ],
    )

    warnings = spec.validate_variables()

    assert len(warnings) == 1
    assert "no states" in warnings[0].lower()


# Test validation warns about binary with wrong number of states.
def test_validate_warns_binary_wrong_state_count() -> None:
    spec = NetworkContext(
        network="test",
        domain="test",
        variables=[
            VariableSpec(name="a", type="binary", states=["a", "b", "c"]),
        ],
    )

    warnings = spec.validate_variables()

    assert any("binary" in w and "3 states" in w for w in warnings)


# Test validation returns no warnings for valid spec.
def test_validate_no_warnings_for_valid_spec() -> None:
    spec = NetworkContext(
        network="test",
        domain="test",
        variables=[
            VariableSpec(name="a", type="binary", states=["no", "yes"]),
            VariableSpec(name="b", type="continuous"),  # No states needed
        ],
    )

    warnings = spec.validate_variables()

    assert warnings == []


# --- NetworkContext.load_and_validate() tests ---


# Test load_and_validate with valid file.
def test_load_and_validate_valid(tmp_path: Path) -> None:
    data = {
        "network": "test",
        "domain": "test_domain",
        "variables": [
            {"name": "a", "type": "binary", "states": ["no", "yes"]},
            {"name": "b", "type": "binary", "states": ["no", "yes"]},
        ],
    }
    json_file = tmp_path / "test.json"
    json_file.write_text(json.dumps(data))

    spec, warnings = NetworkContext.load_and_validate(json_file)

    assert spec.network == "test"
    assert warnings == []


# Test load_and_validate returns warnings.
def test_load_and_validate_with_warnings(tmp_path: Path) -> None:
    data = {
        "network": "test",
        "domain": "test_domain",
        "variables": [
            {"name": "a", "type": "binary"},  # Missing states
        ],
    }
    json_file = tmp_path / "test.json"
    json_file.write_text(json.dumps(data))

    spec, warnings = NetworkContext.load_and_validate(json_file)

    assert spec is not None
    assert len(warnings) == 1


# --- NetworkLoadError tests ---


# Test basic error message.
def test_error_basic() -> None:
    error = NetworkLoadError("Test error")
    assert str(error) == "Test error"
    assert error.message == "Test error"
    assert error.path is None
    assert error.details is None


# Test error with path.
def test_error_with_path() -> None:
    error = NetworkLoadError("Test error", path="/path/to/file.json")
    assert "/path/to/file.json" in str(error)
    assert error.path == "/path/to/file.json"


# Test error with details.
def test_error_with_details() -> None:
    error = NetworkLoadError("Test error", details="Extra info")
    assert "Extra info" in str(error)
    assert error.details == "Extra info"


# Test error with all fields.
def test_error_with_all_fields() -> None:
    error = NetworkLoadError(
        "Test error",
        path="/path/to/file.json",
        details="Extra info",
    )
    message = str(error)
    assert "Test error" in message
    assert "/path/to/file.json" in message
    assert "Extra info" in message
