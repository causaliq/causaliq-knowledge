"""Functional tests for graph generation prompts.

These tests use actual model specification files to verify prompt
generation works correctly with real data.
"""

from pathlib import Path

import pytest

from causaliq_knowledge.graph.loader import ModelLoader
from causaliq_knowledge.graph.prompts import (
    GraphQueryPrompt,
    OutputFormat,
)
from causaliq_knowledge.graph.view_filter import PromptDetail

# Path to test model files.
MODELS_DIR = Path(__file__).parent.parent / "data" / "functional" / "models"


@pytest.fixture
def simple_chain_spec():
    """Load the simple_chain model specification."""
    return ModelLoader.load(MODELS_DIR / "simple_chain.json")


@pytest.fixture
def collider_spec():
    """Load the collider model specification."""
    return ModelLoader.load(MODELS_DIR / "collider.json")


# Test prompt generation from simple_chain model at minimal level.
def test_simple_chain_minimal_prompt(simple_chain_spec) -> None:
    prompt = GraphQueryPrompt.from_model_spec(
        simple_chain_spec,
        level=PromptDetail.MINIMAL,
    )
    system, user = prompt.build()

    # System prompt should request edge list format
    assert "edges" in system
    assert "source" in system
    assert "target" in system

    # User prompt should contain variable names
    assert "cause" in user
    assert "mediator" in user
    assert "effect" in user

    # Domain should be included
    assert "test_domain" in user


# Test prompt generation from simple_chain model at standard level.
def test_simple_chain_standard_prompt(simple_chain_spec) -> None:
    prompt = GraphQueryPrompt.from_model_spec(
        simple_chain_spec,
        level=PromptDetail.STANDARD,
    )
    _, user = prompt.build()

    # Should include variable descriptions
    assert "root cause variable" in user
    assert "Intermediate variable" in user
    assert "outcome variable" in user

    # Should include types
    assert "Type: binary" in user


# Test prompt generation from simple_chain model at rich level.
def test_simple_chain_rich_prompt(simple_chain_spec) -> None:
    prompt = GraphQueryPrompt.from_model_spec(
        simple_chain_spec,
        level=PromptDetail.RICH,
    )
    _, user = prompt.build()

    # Should include roles
    assert "Role: exogenous" in user
    assert "Role: endogenous" in user

    # Rich prompt should mention role considerations
    assert "exogenous variables have no parents" in user


# Test prompt generation from collider model at minimal level.
def test_collider_minimal_prompt(collider_spec) -> None:
    prompt = GraphQueryPrompt.from_model_spec(
        collider_spec,
        level=PromptDetail.MINIMAL,
    )
    _, user = prompt.build()

    # Should contain LLM names (not benchmark names)
    assert "genetic_factor" in user
    assert "environmental_exposure" in user
    assert "disease_status" in user

    # Should include domain
    assert "epidemiology" in user


# Test prompt generation from collider model at standard level.
def test_collider_standard_prompt(collider_spec) -> None:
    prompt = GraphQueryPrompt.from_model_spec(
        collider_spec,
        level=PromptDetail.STANDARD,
    )
    _, user = prompt.build()

    # Should include descriptions
    assert "Genetic predisposition" in user
    assert "Environmental exposure" in user
    assert "Disease diagnosis" in user

    # Should include states
    assert "wild_type" in user
    assert "variant" in user


# Test prompt generation from collider model at rich level.
def test_collider_rich_prompt(collider_spec) -> None:
    prompt = GraphQueryPrompt.from_model_spec(
        collider_spec,
        level=PromptDetail.RICH,
    )
    _, user = prompt.build()

    # Should include extended descriptions
    assert "genetic variant that increases risk" in user
    assert "environmental risk factors" in user


# Test adjacency matrix format with simple_chain.
def test_simple_chain_adjacency_format(simple_chain_spec) -> None:
    prompt = GraphQueryPrompt.from_model_spec(
        simple_chain_spec,
        output_format=OutputFormat.ADJACENCY_MATRIX,
    )
    system, _ = prompt.build()

    # System prompt should request adjacency matrix format
    assert "adjacency_matrix" in system
    assert "variables" in system


# Test adjacency matrix format with collider.
def test_collider_adjacency_format(collider_spec) -> None:
    prompt = GraphQueryPrompt.from_model_spec(
        collider_spec,
        output_format=OutputFormat.ADJACENCY_MATRIX,
    )
    system, _ = prompt.build()

    assert "adjacency_matrix" in system


# Test get_variable_names returns correct names from simple_chain.
def test_simple_chain_variable_names(simple_chain_spec) -> None:
    prompt = GraphQueryPrompt.from_model_spec(
        simple_chain_spec,
        level=PromptDetail.MINIMAL,
    )
    names = prompt.get_variable_names()

    assert len(names) == 3
    assert "cause" in names
    assert "mediator" in names
    assert "effect" in names


# Test get_variable_names returns llm_names from collider.
def test_collider_variable_names(collider_spec) -> None:
    prompt = GraphQueryPrompt.from_model_spec(
        collider_spec,
        level=PromptDetail.MINIMAL,
    )
    names = prompt.get_variable_names()

    assert len(names) == 3
    # Should return llm_name field (default), not benchmark name
    assert "genetic_factor" in names
    assert "environmental_exposure" in names
    assert "disease_status" in names
    # Should NOT contain benchmark names
    assert "Gene" not in names
    assert "Environment" not in names
    assert "Disease" not in names


# Test custom system prompt overrides default for real model.
def test_custom_system_prompt_with_model(simple_chain_spec) -> None:
    custom = "You are a causal discovery expert. Return graph as JSON."
    prompt = GraphQueryPrompt.from_model_spec(
        simple_chain_spec,
        system_prompt=custom,
    )
    system, _ = prompt.build()

    assert system == custom
    assert "confidence" not in system  # Default prompt not used


# Test prompt contains domain from model spec.
def test_prompt_includes_model_domain(collider_spec) -> None:
    prompt = GraphQueryPrompt.from_model_spec(
        collider_spec,
        level=PromptDetail.MINIMAL,
    )

    assert prompt.domain == "epidemiology"
    _, user = prompt.build()
    assert "epidemiology" in user


# Test all view levels produce non-empty prompts.
@pytest.mark.parametrize("level", list(PromptDetail))
def test_all_view_levels_produce_prompts(
    simple_chain_spec,
    level: PromptDetail,
) -> None:
    prompt = GraphQueryPrompt.from_model_spec(simple_chain_spec, level=level)
    system, user = prompt.build()

    assert len(system) > 100  # Non-trivial system prompt
    assert len(user) > 50  # Non-trivial user prompt
    assert "cause" in user  # Contains variable names


# Test all view levels with collider model.
@pytest.mark.parametrize("level", list(PromptDetail))
def test_all_view_levels_collider(collider_spec, level: PromptDetail) -> None:
    prompt = GraphQueryPrompt.from_model_spec(collider_spec, level=level)
    system, user = prompt.build()

    assert len(system) > 100
    assert len(user) > 50
    assert "genetic_factor" in user


# Test both output formats produce valid prompts.
@pytest.mark.parametrize("output_format", list(OutputFormat))
def test_all_output_formats(
    simple_chain_spec,
    output_format: OutputFormat,
) -> None:
    prompt = GraphQueryPrompt.from_model_spec(
        simple_chain_spec,
        output_format=output_format,
    )
    system, user = prompt.build()

    assert len(system) > 100
    assert len(user) > 50

    if output_format == OutputFormat.EDGE_LIST:
        assert "edges" in system
    else:
        assert "adjacency_matrix" in system
