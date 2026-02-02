"""Tests for graph model specifications."""

from __future__ import annotations

from causaliq_knowledge.graph.models import (
    CausalPrinciple,
    Constraints,
    GroundTruth,
    LLMGuidance,
    ModelSpec,
    PromptDetails,
    Provenance,
    VariableRole,
    VariableSpec,
    VariableType,
    ViewDefinition,
)

# --- VariableType enum tests ---


# Test binary type enum value.
def test_variable_type_binary_value() -> None:
    assert VariableType.BINARY == "binary"
    assert VariableType.BINARY.value == "binary"


# Test categorical type enum value.
def test_variable_type_categorical_value() -> None:
    assert VariableType.CATEGORICAL == "categorical"


# Test ordinal type enum value.
def test_variable_type_ordinal_value() -> None:
    assert VariableType.ORDINAL == "ordinal"


# Test continuous type enum value.
def test_variable_type_continuous_value() -> None:
    assert VariableType.CONTINUOUS == "continuous"


# --- VariableRole enum tests ---


# Test exogenous role enum value.
def test_variable_role_exogenous_value() -> None:
    assert VariableRole.EXOGENOUS == "exogenous"


# Test endogenous role enum value.
def test_variable_role_endogenous_value() -> None:
    assert VariableRole.ENDOGENOUS == "endogenous"


# Test latent role enum value.
def test_variable_role_latent_value() -> None:
    assert VariableRole.LATENT == "latent"


# --- VariableSpec model tests ---


# Test creating variable with minimal required fields.
def test_variable_spec_minimal() -> None:
    var = VariableSpec(name="test_var", type="binary")
    assert var.name == "test_var"
    assert var.type == VariableType.BINARY
    assert var.states == []
    assert var.role is None
    assert var.short_description is None


# Test creating variable with all fields populated.
def test_variable_spec_full() -> None:
    var = VariableSpec(
        name="Smoker",
        llm_name="tobacco_history",
        display_name="Tobacco Use History",
        aliases=["smoking", "cigarette_use"],
        type="binary",
        states=["never", "ever"],
        role="exogenous",
        category="lifestyle_exposure",
        short_description="Patient has history of tobacco smoking.",
        extended_description="Detailed description here.",
        base_rate={"never": 0.70, "ever": 0.30},
        sensitivity_hints="Important causal hints.",
        related_domain_knowledge=["Fact 1", "Fact 2"],
        references=["Ref1", "Ref2"],
    )
    assert var.name == "Smoker"
    assert var.llm_name == "tobacco_history"
    assert var.type == VariableType.BINARY
    assert var.role == VariableRole.EXOGENOUS
    assert var.states == ["never", "ever"]
    assert var.base_rate == {"never": 0.70, "ever": 0.30}
    assert len(var.aliases) == 2
    assert len(var.related_domain_knowledge) == 2


# Test type enum validation from string input.
def test_variable_spec_type_validation_from_string() -> None:
    var = VariableSpec(name="test", type="categorical")
    assert var.type == VariableType.CATEGORICAL


# Test type validation is case insensitive.
def test_variable_spec_type_validation_case_insensitive() -> None:
    var = VariableSpec(name="test", type="BINARY")
    assert var.type == VariableType.BINARY


# Test type validation when already a VariableType enum.
def test_variable_spec_type_validation_from_enum() -> None:
    var = VariableSpec(name="test", type=VariableType.ORDINAL)
    assert var.type == VariableType.ORDINAL


# Test role enum validation from string input.
def test_variable_spec_role_validation_from_string() -> None:
    var = VariableSpec(name="test", type="binary", role="endogenous")
    assert var.role == VariableRole.ENDOGENOUS


# Test role validation when already a VariableRole enum.
def test_variable_spec_role_validation_from_enum() -> None:
    var = VariableSpec(name="test", type="binary", role=VariableRole.LATENT)
    assert var.role == VariableRole.LATENT


# Test role can be None.
def test_variable_spec_role_none_allowed() -> None:
    var = VariableSpec(name="test", type="binary", role=None)
    assert var.role is None


# --- Provenance model tests ---


# Test creating provenance with no fields (all defaults).
def test_provenance_empty() -> None:
    prov = Provenance()
    assert prov.source_network is None
    assert prov.source_url is None


# Test creating provenance with all fields populated.
def test_provenance_full() -> None:
    prov = Provenance(
        source_network="cancer",
        source_reference="Korb2010BayesianAI",
        source_url="https://example.com",
        disguise_strategy="semantic_rename",
        memorization_risk="moderate",
        notes="Test notes",
    )
    assert prov.source_network == "cancer"
    assert prov.memorization_risk == "moderate"


# --- LLMGuidance model tests ---


# Test creating LLM guidance with defaults.
def test_llm_guidance_empty() -> None:
    guidance = LLMGuidance()
    assert guidance.usage_notes == []
    assert guidance.do_not_provide == []


# Test creating LLM guidance with all fields.
def test_llm_guidance_full() -> None:
    guidance = LLMGuidance(
        usage_notes=["Note 1", "Note 2"],
        do_not_provide=["Secret 1"],
    )
    assert len(guidance.usage_notes) == 2
    assert len(guidance.do_not_provide) == 1


# --- ViewDefinition model tests ---


# Test creating view definition with defaults.
def test_view_definition_empty() -> None:
    view = ViewDefinition()
    assert view.description is None
    assert view.include_fields == []


# Test creating view definition with all fields.
def test_view_definition_full() -> None:
    view = ViewDefinition(
        description="Minimal view",
        include_fields=["name", "type"],
    )
    assert view.description == "Minimal view"
    assert view.include_fields == ["name", "type"]


# --- PromptDetails model tests ---


# Test PromptDetails has sensible default field selections.
def test_prompt_details_default() -> None:
    prompt_details = PromptDetails()
    assert "name" in prompt_details.minimal.include_fields
    assert "type" in prompt_details.standard.include_fields
    assert "extended_description" in prompt_details.rich.include_fields


# Test creating PromptDetails with custom view definitions.
def test_prompt_details_custom() -> None:
    prompt_details = PromptDetails(
        minimal=ViewDefinition(include_fields=["name"]),
        standard=ViewDefinition(include_fields=["name", "type"]),
        rich=ViewDefinition(include_fields=["name", "type", "description"]),
    )
    assert prompt_details.minimal.include_fields == ["name"]
    assert len(prompt_details.standard.include_fields) == 2


# --- Constraints model tests ---


# Test creating constraints with defaults.
def test_constraints_empty() -> None:
    constraints = Constraints()
    assert constraints.forbidden_edges == []
    assert constraints.partial_order == []
    assert constraints.tiers == {}


# Test creating constraints with all fields populated.
def test_constraints_full() -> None:
    constraints = Constraints(
        forbidden_edges=[["A", "B"], ["C", "D"]],
        partial_order=[["A", "C"]],
        tiers={"exposure": ["A", "B"], "outcome": ["C"]},
        notes="Test constraints",
    )
    assert len(constraints.forbidden_edges) == 2
    assert constraints.tiers["exposure"] == ["A", "B"]


# --- CausalPrinciple model tests ---


# Test creating causal principle with required fields only.
def test_causal_principle_minimal() -> None:
    principle = CausalPrinciple(
        id="test_principle",
        statement="A causes B.",
    )
    assert principle.id == "test_principle"
    assert principle.statement == "A causes B."
    assert principle.references == []


# Test creating causal principle with all fields.
def test_causal_principle_full() -> None:
    principle = CausalPrinciple(
        id="test_principle",
        statement="A causes B.",
        references=["Pearl2009"],
    )
    assert len(principle.references) == 1


# --- GroundTruth model tests ---


# Test creating ground truth with defaults.
def test_ground_truth_empty() -> None:
    gt = GroundTruth()
    assert gt.edges == []
    assert gt.v_structures == []


# Test creating ground truth with all fields populated.
def test_ground_truth_full() -> None:
    gt = GroundTruth(
        edges=[["A", "B"], ["B", "C"]],
        v_structures=[{"parents": ["A", "B"], "child": "C"}],
    )
    assert len(gt.edges) == 2
    assert len(gt.v_structures) == 1


# --- ModelSpec model tests ---


# Test creating ModelSpec with minimal required fields.
def test_model_spec_minimal() -> None:
    spec = ModelSpec(
        dataset_id="test",
        domain="test_domain",
    )
    assert spec.dataset_id == "test"
    assert spec.domain == "test_domain"
    assert spec.schema_version == "2.0"
    assert spec.variables == []


# Test creating ModelSpec with all fields populated.
def test_model_spec_full() -> None:
    spec = ModelSpec(
        schema_version="2.0",
        dataset_id="cancer",
        domain="oncology",
        purpose="Testing",
        provenance=Provenance(source_network="cancer"),
        llm_guidance=LLMGuidance(usage_notes=["Note"]),
        prompt_details=PromptDetails(),
        variables=[
            VariableSpec(name="smoking", type="binary", states=["no", "yes"]),
            VariableSpec(name="cancer", type="binary", states=["no", "yes"]),
        ],
        constraints=Constraints(),
        causal_principles=[
            CausalPrinciple(id="p1", statement="Smoking causes cancer")
        ],
        ground_truth=GroundTruth(edges_experiment=[["smoking", "cancer"]]),
    )
    assert spec.dataset_id == "cancer"
    assert len(spec.variables) == 2
    assert len(spec.causal_principles) == 1


# Test get_variable returns variable when found.
def test_model_spec_get_variable_found() -> None:
    spec = ModelSpec(
        dataset_id="test",
        domain="test",
        variables=[
            VariableSpec(name="a", type="binary"),
            VariableSpec(name="b", type="binary"),
        ],
    )
    var = spec.get_variable("a")
    assert var is not None
    assert var.name == "a"


# Test get_variable returns None when not found.
def test_model_spec_get_variable_not_found() -> None:
    spec = ModelSpec(
        dataset_id="test",
        domain="test",
        variables=[VariableSpec(name="a", type="binary")],
    )
    var = spec.get_variable("nonexistent")
    assert var is None


# Test get_variable_names returns all names.
def test_model_spec_get_variable_names() -> None:
    spec = ModelSpec(
        dataset_id="test",
        domain="test",
        variables=[
            VariableSpec(name="a", type="binary"),
            VariableSpec(name="b", type="categorical"),
            VariableSpec(name="c", type="continuous"),
        ],
    )
    names = spec.get_variable_names()
    assert names == ["a", "b", "c"]


# Test get_llm_names returns list of llm_name values.
def test_model_spec_get_llm_names() -> None:
    spec = ModelSpec(
        dataset_id="test",
        domain="test",
        variables=[
            VariableSpec(name="smoke", llm_name="tobacco_use", type="binary"),
            VariableSpec(name="lung", llm_name="cancer_status", type="binary"),
            VariableSpec(name="age", type="continuous"),  # llm_name defaults
        ],
    )
    llm_names = spec.get_llm_names()
    assert llm_names == ["tobacco_use", "cancer_status", "age"]


# Test get_llm_to_name_mapping returns correct mapping.
def test_model_spec_get_llm_to_name_mapping() -> None:
    spec = ModelSpec(
        dataset_id="test",
        domain="test",
        variables=[
            VariableSpec(name="smoke", llm_name="tobacco_use", type="binary"),
            VariableSpec(name="lung", llm_name="cancer_status", type="binary"),
        ],
    )
    mapping = spec.get_llm_to_name_mapping()
    assert mapping == {"tobacco_use": "smoke", "cancer_status": "lung"}


# Test get_name_to_llm_mapping returns correct mapping.
def test_model_spec_get_name_to_llm_mapping() -> None:
    spec = ModelSpec(
        dataset_id="test",
        domain="test",
        variables=[
            VariableSpec(name="smoke", llm_name="tobacco_use", type="binary"),
            VariableSpec(name="lung", llm_name="cancer_status", type="binary"),
        ],
    )
    mapping = spec.get_name_to_llm_mapping()
    assert mapping == {"smoke": "tobacco_use", "lung": "cancer_status"}


# Test uses_distinct_llm_names returns True when names differ.
def test_model_spec_uses_distinct_llm_names_true() -> None:
    spec = ModelSpec(
        dataset_id="test",
        domain="test",
        variables=[
            VariableSpec(name="smoke", llm_name="tobacco_use", type="binary"),
            VariableSpec(name="lung", type="binary"),  # llm_name = name
        ],
    )
    assert spec.uses_distinct_llm_names() is True


# Test uses_distinct_llm_names returns False when all names same.
def test_model_spec_uses_distinct_llm_names_false() -> None:
    spec = ModelSpec(
        dataset_id="test",
        domain="test",
        variables=[
            VariableSpec(name="a", type="binary"),  # llm_name defaults to "a"
            VariableSpec(name="b", type="binary"),  # llm_name defaults to "b"
        ],
    )
    assert spec.uses_distinct_llm_names() is False


# Test llm_name defaults to name when not specified.
def test_variable_spec_llm_name_defaults_to_name() -> None:
    var = VariableSpec(name="test_var", type="binary")
    assert var.name == "test_var"
    assert var.llm_name == "test_var"  # Should default to name
