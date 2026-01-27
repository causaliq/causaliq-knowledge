"""Functional tests for VariableDisguiser using tracked test files."""

from __future__ import annotations

from pathlib import Path

from causaliq_knowledge.graph import (
    ModelLoader,
    VariableDisguiser,
)

# Path to test model files.
TEST_MODELS_DIR = (
    Path(__file__).parent.parent / "data" / "functional" / "models"
)


# Test VariableDisguiser creates mapping for all variables.
def test_disguiser_creates_mapping_for_all_variables() -> None:
    spec = ModelLoader.load(TEST_MODELS_DIR / "simple_chain.json")
    disguiser = VariableDisguiser(spec, seed=42)
    mapping = disguiser.original_to_disguised
    assert len(mapping) == 3
    assert "cause" in mapping
    assert "mediator" in mapping
    assert "effect" in mapping


# Test VariableDisguiser seed produces reproducible mapping.
def test_disguiser_seed_reproducibility() -> None:
    spec = ModelLoader.load(TEST_MODELS_DIR / "collider.json")
    disguiser1 = VariableDisguiser(spec, seed=123)
    disguiser2 = VariableDisguiser(spec, seed=123)
    assert disguiser1.original_to_disguised == disguiser2.original_to_disguised


# Test VariableDisguiser different seeds produce different mappings.
def test_disguiser_different_seeds() -> None:
    spec = ModelLoader.load(TEST_MODELS_DIR / "collider.json")
    disguiser1 = VariableDisguiser(spec, seed=42)
    disguiser2 = VariableDisguiser(spec, seed=99)
    assert disguiser1.original_to_disguised != disguiser2.original_to_disguised


# Test VariableDisguiser disguise and reveal are inverses.
def test_disguiser_round_trip() -> None:
    spec = ModelLoader.load(TEST_MODELS_DIR / "simple_chain.json")
    disguiser = VariableDisguiser(spec, seed=42)
    for name in ["cause", "mediator", "effect"]:
        disguised = disguiser.disguise_name(name)
        revealed = disguiser.reveal_name(disguised)
        assert revealed == name


# Test VariableDisguiser disguise_text replaces all variable names.
def test_disguiser_text_replacement() -> None:
    spec = ModelLoader.load(TEST_MODELS_DIR / "simple_chain.json")
    disguiser = VariableDisguiser(spec, seed=42)
    text = "The cause affects the mediator which influences the effect."
    disguised = disguiser.disguise_text(text)
    # Original names should not appear
    assert "cause" not in disguised
    assert "mediator" not in disguised
    assert "effect" not in disguised
    # Disguised names should appear
    assert "V" in disguised


# Test VariableDisguiser reveal_text restores original names.
def test_disguiser_text_reveal() -> None:
    spec = ModelLoader.load(TEST_MODELS_DIR / "simple_chain.json")
    disguiser = VariableDisguiser(spec, seed=42)
    original = "cause -> mediator -> effect"
    disguised = disguiser.disguise_text(original)
    revealed = disguiser.reveal_text(disguised)
    assert "cause" in revealed
    assert "mediator" in revealed
    assert "effect" in revealed


# Test VariableDisguiser with collider model variable names.
def test_disguiser_collider_variables() -> None:
    spec = ModelLoader.load(TEST_MODELS_DIR / "collider.json")
    disguiser = VariableDisguiser(spec, seed=42)
    names = ["genetic_factor", "environmental_exposure", "disease_status"]
    disguised = disguiser.disguise_names_list(names)
    revealed = disguiser.reveal_names_list(disguised)
    assert revealed == names


# Test VariableDisguiser custom prefix.
def test_disguiser_custom_prefix() -> None:
    spec = ModelLoader.load(TEST_MODELS_DIR / "minimal.json")
    disguiser = VariableDisguiser(spec, seed=42, prefix="VAR")
    mapping = disguiser.original_to_disguised
    for disguised_name in mapping.values():
        assert disguised_name.startswith("VAR")


# Test VariableDisguiser preserves non-variable text.
def test_disguiser_preserves_other_text() -> None:
    spec = ModelLoader.load(TEST_MODELS_DIR / "simple_chain.json")
    disguiser = VariableDisguiser(spec, seed=42)
    text = "The relationship between cause and effect is mediated."
    disguised = disguiser.disguise_text(text)
    # Non-variable words should remain
    assert "relationship" in disguised
    assert "between" in disguised
    assert "mediated" in disguised


# Test VariableDisguiser case insensitive replacement.
def test_disguiser_case_insensitive() -> None:
    spec = ModelLoader.load(TEST_MODELS_DIR / "simple_chain.json")
    disguiser = VariableDisguiser(spec, seed=42)
    text = "CAUSE affects Effect through Mediator"
    disguised = disguiser.disguise_text(text)
    # All variations should be replaced
    assert "CAUSE" not in disguised
    assert "Effect" not in disguised
    assert "Mediator" not in disguised


# Test VariableDisguiser bidirectional mapping is consistent.
def test_disguiser_bidirectional_mapping() -> None:
    spec = ModelLoader.load(TEST_MODELS_DIR / "collider.json")
    disguiser = VariableDisguiser(spec, seed=42)
    orig_to_disg = disguiser.original_to_disguised
    disg_to_orig = disguiser.disguised_to_original
    # Every original maps to a unique disguised name
    assert len(set(orig_to_disg.values())) == len(orig_to_disg)
    # Mappings are inverses
    for orig, disg in orig_to_disg.items():
        assert disg_to_orig[disg] == orig
