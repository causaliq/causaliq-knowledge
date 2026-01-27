"""Unit tests for VariableDisguiser class."""

from __future__ import annotations

import pytest

from causaliq_knowledge.graph.disguiser import VariableDisguiser
from causaliq_knowledge.graph.models import (
    ModelSpec,
    VariableSpec,
    VariableType,
)


def _create_test_spec() -> ModelSpec:
    """Create a test model specification."""
    return ModelSpec(
        dataset_id="test",
        domain="epidemiology",
        variables=[
            VariableSpec(
                name="smoking",
                description="Smoking status",
                type=VariableType.BINARY,
            ),
            VariableSpec(
                name="cancer",
                description="Cancer diagnosis",
                type=VariableType.BINARY,
            ),
            VariableSpec(
                name="age",
                description="Patient age",
                type=VariableType.CONTINUOUS,
            ),
        ],
    )


# Test VariableDisguiser initialisation stores seed.
def test_disguiser_init_stores_seed() -> None:
    spec = _create_test_spec()
    disguiser = VariableDisguiser(spec, seed=42)
    assert disguiser.seed == 42


# Test VariableDisguiser with no seed.
def test_disguiser_init_no_seed() -> None:
    spec = _create_test_spec()
    disguiser = VariableDisguiser(spec)
    assert disguiser.seed is None


# Test VariableDisguiser stores custom prefix.
def test_disguiser_init_stores_prefix() -> None:
    spec = _create_test_spec()
    disguiser = VariableDisguiser(spec, prefix="VAR")
    assert disguiser.prefix == "VAR"


# Test default prefix is V.
def test_disguiser_default_prefix() -> None:
    spec = _create_test_spec()
    disguiser = VariableDisguiser(spec)
    assert disguiser.prefix == "V"


# Test original_to_disguised mapping has all variables.
def test_original_to_disguised_has_all_vars() -> None:
    spec = _create_test_spec()
    disguiser = VariableDisguiser(spec, seed=42)
    mapping = disguiser.original_to_disguised
    assert len(mapping) == 3
    assert "smoking" in mapping
    assert "cancer" in mapping
    assert "age" in mapping


# Test disguised_to_original mapping has all variables.
def test_disguised_to_original_has_all_vars() -> None:
    spec = _create_test_spec()
    disguiser = VariableDisguiser(spec, seed=42)
    mapping = disguiser.disguised_to_original
    assert len(mapping) == 3


# Test mappings are bidirectional.
def test_mappings_are_bidirectional() -> None:
    spec = _create_test_spec()
    disguiser = VariableDisguiser(spec, seed=42)
    for orig, disg in disguiser.original_to_disguised.items():
        assert disguiser.disguised_to_original[disg] == orig


# Test disguise_name returns disguised name.
def test_disguise_name() -> None:
    spec = _create_test_spec()
    disguiser = VariableDisguiser(spec, seed=42)
    disguised = disguiser.disguise_name("smoking")
    assert disguised.startswith("V")
    assert disguised[1:].isdigit()


# Test reveal_name returns original name.
def test_reveal_name() -> None:
    spec = _create_test_spec()
    disguiser = VariableDisguiser(spec, seed=42)
    disguised = disguiser.disguise_name("smoking")
    revealed = disguiser.reveal_name(disguised)
    assert revealed == "smoking"


# Test disguise_name raises KeyError for unknown name.
def test_disguise_name_unknown_raises() -> None:
    spec = _create_test_spec()
    disguiser = VariableDisguiser(spec, seed=42)
    with pytest.raises(KeyError, match="Unknown variable"):
        disguiser.disguise_name("unknown")


# Test reveal_name raises KeyError for unknown disguised name.
def test_reveal_name_unknown_raises() -> None:
    spec = _create_test_spec()
    disguiser = VariableDisguiser(spec, seed=42)
    with pytest.raises(KeyError, match="Unknown disguised name"):
        disguiser.reveal_name("V999")


# Test seed produces reproducible mapping.
def test_seed_reproducibility() -> None:
    spec = _create_test_spec()
    disguiser1 = VariableDisguiser(spec, seed=42)
    disguiser2 = VariableDisguiser(spec, seed=42)
    assert disguiser1.original_to_disguised == disguiser2.original_to_disguised


# Test different seeds produce different mappings.
def test_different_seeds_different_mappings() -> None:
    spec = _create_test_spec()
    disguiser1 = VariableDisguiser(spec, seed=42)
    disguiser2 = VariableDisguiser(spec, seed=99)
    assert disguiser1.original_to_disguised != disguiser2.original_to_disguised


# Test disguise_text replaces all variable names.
def test_disguise_text() -> None:
    spec = _create_test_spec()
    disguiser = VariableDisguiser(spec, seed=42)
    text = "smoking causes cancer"
    disguised = disguiser.disguise_text(text)
    assert "smoking" not in disguised
    assert "cancer" not in disguised


# Test reveal_text restores original names.
def test_reveal_text() -> None:
    spec = _create_test_spec()
    disguiser = VariableDisguiser(spec, seed=42)
    text = "smoking causes cancer"
    disguised = disguiser.disguise_text(text)
    revealed = disguiser.reveal_text(disguised)
    assert "smoking" in revealed
    assert "cancer" in revealed


# Test disguise_text is case insensitive.
def test_disguise_text_case_insensitive() -> None:
    spec = _create_test_spec()
    disguiser = VariableDisguiser(spec, seed=42)
    text = "SMOKING causes Cancer"
    disguised = disguiser.disguise_text(text)
    assert "SMOKING" not in disguised
    assert "Cancer" not in disguised


# Test reveal_text is case insensitive.
def test_reveal_text_case_insensitive() -> None:
    spec = _create_test_spec()
    disguiser = VariableDisguiser(spec, seed=42)
    # Get the disguised forms
    v1 = disguiser.disguise_name("smoking")
    # Create text with uppercase disguised name
    text = f"{v1.upper()} affects health"
    revealed = disguiser.reveal_text(text)
    assert "smoking" in revealed


# Test disguise_names_list converts all names.
def test_disguise_names_list() -> None:
    spec = _create_test_spec()
    disguiser = VariableDisguiser(spec, seed=42)
    names = ["smoking", "cancer"]
    disguised = disguiser.disguise_names_list(names)
    assert len(disguised) == 2
    assert all(n.startswith("V") for n in disguised)


# Test reveal_names_list converts all names.
def test_reveal_names_list() -> None:
    spec = _create_test_spec()
    disguiser = VariableDisguiser(spec, seed=42)
    names = ["smoking", "cancer"]
    disguised = disguiser.disguise_names_list(names)
    revealed = disguiser.reveal_names_list(disguised)
    assert revealed == names


# Test custom prefix is used in disguised names.
def test_custom_prefix() -> None:
    spec = _create_test_spec()
    disguiser = VariableDisguiser(spec, seed=42, prefix="VAR")
    disguised = disguiser.disguise_name("smoking")
    assert disguised.startswith("VAR")


# Test mapping properties return copies.
def test_mapping_properties_return_copies() -> None:
    spec = _create_test_spec()
    disguiser = VariableDisguiser(spec, seed=42)
    mapping1 = disguiser.original_to_disguised
    mapping2 = disguiser.original_to_disguised
    assert mapping1 is not mapping2
    assert mapping1 == mapping2


# Test reveal_names_list raises for unknown name.
def test_reveal_names_list_unknown_raises() -> None:
    spec = _create_test_spec()
    disguiser = VariableDisguiser(spec, seed=42)
    with pytest.raises(KeyError, match="Unknown disguised name"):
        disguiser.reveal_names_list(["V999"])


# Test disguise_names_list raises for unknown name.
def test_disguise_names_list_unknown_raises() -> None:
    spec = _create_test_spec()
    disguiser = VariableDisguiser(spec, seed=42)
    with pytest.raises(KeyError, match="Unknown variable"):
        disguiser.disguise_names_list(["unknown"])
