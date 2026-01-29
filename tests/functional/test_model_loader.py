"""Functional tests for model specification loading from files."""

from __future__ import annotations

from pathlib import Path

from causaliq_knowledge.graph.loader import ModelLoader
from causaliq_knowledge.graph.models import VariableRole, VariableType

# Path to test model files.
MODELS_DIR = Path(__file__).parent.parent / "data" / "functional" / "models"


# --- Simple chain model tests ---


# Test loading simple_chain.json returns valid ModelSpec.
def test_load_simple_chain() -> None:
    spec = ModelLoader.load(MODELS_DIR / "simple_chain.json")

    assert spec.dataset_id == "simple_chain"
    assert spec.domain == "test_domain"
    assert len(spec.variables) == 3


# Test simple_chain variable names are correct.
def test_simple_chain_variable_names() -> None:
    spec = ModelLoader.load(MODELS_DIR / "simple_chain.json")

    names = spec.get_variable_names()
    assert names == ["cause", "mediator", "effect"]


# Test simple_chain variable roles are correctly parsed.
def test_simple_chain_variable_roles() -> None:
    spec = ModelLoader.load(MODELS_DIR / "simple_chain.json")

    cause = spec.get_variable("cause")
    assert cause is not None
    assert cause.role == VariableRole.EXOGENOUS

    mediator = spec.get_variable("mediator")
    assert mediator is not None
    assert mediator.role == VariableRole.ENDOGENOUS


# Test simple_chain ground truth edges are loaded.
def test_simple_chain_ground_truth() -> None:
    spec = ModelLoader.load(MODELS_DIR / "simple_chain.json")

    assert spec.ground_truth is not None
    assert len(spec.ground_truth.edges) == 2
    assert ["cause", "mediator"] in spec.ground_truth.edges


# --- Collider model tests ---


# Test loading collider.json returns valid ModelSpec.
def test_load_collider() -> None:
    spec = ModelLoader.load(MODELS_DIR / "collider.json")

    assert spec.dataset_id == "collider"
    assert spec.domain == "epidemiology"
    assert len(spec.variables) == 3


# Test collider provenance information is loaded.
def test_collider_provenance() -> None:
    spec = ModelLoader.load(MODELS_DIR / "collider.json")

    assert spec.provenance is not None
    assert spec.provenance.source_network == "synthetic"


# Test collider LLM guidance is loaded.
def test_collider_llm_guidance() -> None:
    spec = ModelLoader.load(MODELS_DIR / "collider.json")

    assert spec.llm_guidance is not None
    assert len(spec.llm_guidance.usage_notes) == 2
    assert len(spec.llm_guidance.do_not_provide) == 1


# Test collider views are correctly configured.
def test_collider_views() -> None:
    spec = ModelLoader.load(MODELS_DIR / "collider.json")

    assert "name" in spec.views.minimal.include_fields
    assert "short_description" in spec.views.standard.include_fields
    assert "extended_description" in spec.views.rich.include_fields


# Test collider constraints are loaded.
def test_collider_constraints() -> None:
    spec = ModelLoader.load(MODELS_DIR / "collider.json")

    assert spec.constraints is not None
    assert len(spec.constraints.forbidden_edges) == 2
    assert len(spec.constraints.partial_order) == 2
    assert "causes" in spec.constraints.tiers


# Test collider causal principles are loaded.
def test_collider_causal_principles() -> None:
    spec = ModelLoader.load(MODELS_DIR / "collider.json")

    assert len(spec.causal_principles) == 1
    assert spec.causal_principles[0].id == "collider_structure"


# Test collider llm_name to name mapping.
def test_collider_llm_to_name_mapping() -> None:
    spec = ModelLoader.load(MODELS_DIR / "collider.json")

    mapping = spec.get_llm_to_name_mapping()
    assert mapping["genetic_factor"] == "Gene"
    assert mapping["environmental_exposure"] == "Environment"
    assert mapping["disease_status"] == "Disease"


# Test collider v-structures in ground truth.
def test_collider_v_structures() -> None:
    spec = ModelLoader.load(MODELS_DIR / "collider.json")

    assert spec.ground_truth is not None
    assert len(spec.ground_truth.v_structures) == 1


# --- Minimal model tests ---


# Test loading minimal.json with only required fields.
def test_load_minimal() -> None:
    spec = ModelLoader.load(MODELS_DIR / "minimal.json")

    assert spec.dataset_id == "minimal"
    assert spec.domain == "testing"
    assert len(spec.variables) == 2


# Test minimal model variables are continuous type.
def test_minimal_continuous_variables() -> None:
    spec = ModelLoader.load(MODELS_DIR / "minimal.json")

    for var in spec.variables:
        assert var.type == VariableType.CONTINUOUS


# Test minimal model has no optional fields set.
def test_minimal_optional_fields_absent() -> None:
    spec = ModelLoader.load(MODELS_DIR / "minimal.json")

    assert spec.provenance is None
    assert spec.llm_guidance is None
    assert spec.constraints is None
    assert spec.ground_truth is None


# --- Validation tests ---


# Test load_and_validate returns no warnings for complete models.
def test_validate_collider_no_warnings() -> None:
    spec, warnings = ModelLoader.load_and_validate(
        MODELS_DIR / "collider.json"
    )

    assert spec is not None
    assert warnings == []


# Test load_and_validate warns about missing states in minimal model.
def test_validate_minimal_warns_no_states() -> None:
    spec, warnings = ModelLoader.load_and_validate(MODELS_DIR / "minimal.json")

    assert spec is not None
    # Continuous variables don't need states, so no warnings
    assert warnings == []
