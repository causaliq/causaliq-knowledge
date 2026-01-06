"""Unit tests for llm.prompts module."""

from causaliq_knowledge.llm.prompts import (
    DEFAULT_SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    USER_PROMPT_WITH_DOMAIN_TEMPLATE,
    VARIABLE_DESCRIPTIONS_TEMPLATE,
    EdgeQueryPrompt,
    parse_edge_response,
)
from causaliq_knowledge.models import EdgeDirection

# --- EdgeQueryPrompt Tests ---


# Test basic prompt construction without domain or descriptions.
def test_edge_query_prompt_build_basic():
    prompt = EdgeQueryPrompt(node_a="X", node_b="Y")
    system, user = prompt.build()

    assert system == DEFAULT_SYSTEM_PROMPT
    assert '"X"' in user
    assert '"Y"' in user
    assert "X causes Y" in user
    assert "Y causes X" in user


# Test prompt construction with domain context.
def test_edge_query_prompt_build_with_domain():
    prompt = EdgeQueryPrompt(node_a="X", node_b="Y", domain="medicine")
    system, user = prompt.build()

    assert system == DEFAULT_SYSTEM_PROMPT
    assert "In the domain of medicine:" in user
    assert '"X"' in user
    assert '"Y"' in user


# Test prompt construction with variable descriptions.
def test_edge_query_prompt_build_with_descriptions():
    descriptions = {
        "X": "The treatment variable",
        "Y": "The outcome variable",
    }
    prompt = EdgeQueryPrompt(node_a="X", node_b="Y", descriptions=descriptions)
    system, user = prompt.build()

    assert "Variable descriptions:" in user
    assert "X: The treatment variable" in user
    assert "Y: The outcome variable" in user


# Test prompt construction with missing variable descriptions.
def test_edge_query_prompt_build_with_missing_descriptions():
    descriptions = {"X": "Only X described"}
    prompt = EdgeQueryPrompt(node_a="X", node_b="Y", descriptions=descriptions)
    system, user = prompt.build()

    assert "X: Only X described" in user
    assert "Y: No description" in user


# Test prompt construction with domain and descriptions.
def test_edge_query_prompt_build_with_domain_and_descriptions():
    descriptions = {"X": "Variable X desc", "Y": "Variable Y desc"}
    prompt = EdgeQueryPrompt(
        node_a="X", node_b="Y", domain="economics", descriptions=descriptions
    )
    system, user = prompt.build()

    assert "In the domain of economics:" in user
    assert "Variable descriptions:" in user
    assert "X: Variable X desc" in user
    assert "Y: Variable Y desc" in user


# Test prompt construction with custom system prompt.
def test_edge_query_prompt_build_with_custom_system():
    custom_system = "You are a custom assistant."
    prompt = EdgeQueryPrompt(
        node_a="X", node_b="Y", system_prompt=custom_system
    )
    system, user = prompt.build()

    assert system == custom_system


# Test from_context factory with None context.
def test_edge_query_prompt_from_context_none():
    prompt = EdgeQueryPrompt.from_context("X", "Y", None)

    assert prompt.node_a == "X"
    assert prompt.node_b == "Y"
    assert prompt.domain is None
    assert prompt.descriptions is None


# Test from_context factory with empty context.
def test_edge_query_prompt_from_context_empty():
    prompt = EdgeQueryPrompt.from_context("X", "Y", {})

    assert prompt.node_a == "X"
    assert prompt.node_b == "Y"
    assert prompt.domain is None
    assert prompt.descriptions is None


# Test from_context factory with full context.
def test_edge_query_prompt_from_context_full():
    context = {
        "domain": "biology",
        "descriptions": {"X": "Gene X", "Y": "Protein Y"},
        "system_prompt": "Custom system prompt",
    }
    prompt = EdgeQueryPrompt.from_context("X", "Y", context)

    assert prompt.node_a == "X"
    assert prompt.node_b == "Y"
    assert prompt.domain == "biology"
    assert prompt.descriptions == {"X": "Gene X", "Y": "Protein Y"}
    assert prompt.system_prompt == "Custom system prompt"


# --- parse_edge_response Tests ---


# Test parse_edge_response with valid complete response.
def test_parse_edge_response_valid_complete():
    json_data = {
        "exists": True,
        "direction": "a_to_b",
        "confidence": 0.85,
        "reasoning": "Strong causal evidence",
    }
    result = parse_edge_response(json_data, model="gpt-4")

    assert result.exists is True
    assert result.direction == EdgeDirection.A_TO_B
    assert result.confidence == 0.85
    assert result.reasoning == "Strong causal evidence"
    assert result.model == "gpt-4"


# Test parse_edge_response with None json_data.
def test_parse_edge_response_none():
    result = parse_edge_response(None, model="gpt-4")

    assert result.exists is None
    assert result.direction is None
    assert result.confidence == 0.0
    assert "Failed to parse" in result.reasoning
    assert result.model == "gpt-4"


# Test parse_edge_response with minimal response.
def test_parse_edge_response_minimal():
    json_data = {}
    result = parse_edge_response(json_data)

    assert result.exists is None
    assert result.direction is None
    assert result.confidence == 0.0
    assert result.reasoning == ""
    assert result.model is None


# Test parse_edge_response with exists=false.
def test_parse_edge_response_no_edge():
    json_data = {
        "exists": False,
        "direction": None,
        "confidence": 0.9,
        "reasoning": "No causal link",
    }
    result = parse_edge_response(json_data)

    assert result.exists is False
    assert result.direction is None
    assert result.confidence == 0.9


# Test parse_edge_response with b_to_a direction.
def test_parse_edge_response_b_to_a():
    json_data = {
        "exists": True,
        "direction": "b_to_a",
        "confidence": 0.7,
        "reasoning": "Reverse causation",
    }
    result = parse_edge_response(json_data)

    assert result.direction == EdgeDirection.B_TO_A


# Test parse_edge_response with undirected edge.
def test_parse_edge_response_undirected():
    json_data = {
        "exists": True,
        "direction": "undirected",
        "confidence": 0.5,
        "reasoning": "Bidirectional relationship",
    }
    result = parse_edge_response(json_data)

    assert result.direction == EdgeDirection.UNDIRECTED


# Test parse_edge_response with uppercase direction.
def test_parse_edge_response_uppercase_direction():
    json_data = {
        "exists": True,
        "direction": "A_TO_B",
        "confidence": 0.8,
        "reasoning": "Test uppercase",
    }
    result = parse_edge_response(json_data)

    assert result.direction == EdgeDirection.A_TO_B


# Test parse_edge_response with invalid direction string.
def test_parse_edge_response_invalid_direction():
    json_data = {
        "exists": True,
        "direction": "invalid_direction",
        "confidence": 0.8,
        "reasoning": "Test invalid direction",
    }
    result = parse_edge_response(json_data)

    assert result.exists is True
    assert result.direction is None
    assert result.confidence == 0.8


# Test parse_edge_response with non-numeric confidence.
def test_parse_edge_response_invalid_confidence():
    json_data = {
        "exists": True,
        "direction": "a_to_b",
        "confidence": "high",
        "reasoning": "Non-numeric confidence",
    }
    result = parse_edge_response(json_data)

    assert result.confidence == 0.0


# Test parse_edge_response clamps confidence to 0-1 range (high).
def test_parse_edge_response_confidence_clamp_high():
    json_data = {
        "exists": True,
        "direction": "a_to_b",
        "confidence": 1.5,
        "reasoning": "Over 1.0 confidence",
    }
    result = parse_edge_response(json_data)

    assert result.confidence == 1.0


# Test parse_edge_response clamps confidence to 0-1 range (low).
def test_parse_edge_response_confidence_clamp_low():
    json_data = {
        "exists": True,
        "direction": "a_to_b",
        "confidence": -0.5,
        "reasoning": "Negative confidence",
    }
    result = parse_edge_response(json_data)

    assert result.confidence == 0.0


# Test parse_edge_response with None confidence.
def test_parse_edge_response_none_confidence():
    json_data = {
        "exists": True,
        "direction": "a_to_b",
        "confidence": None,
        "reasoning": "None confidence",
    }
    result = parse_edge_response(json_data)

    assert result.confidence == 0.0


# Test parse_edge_response converts reasoning to string.
def test_parse_edge_response_reasoning_to_string():
    json_data = {
        "exists": True,
        "direction": "a_to_b",
        "confidence": 0.8,
        "reasoning": 12345,  # Non-string reasoning
    }
    result = parse_edge_response(json_data)

    assert result.reasoning == "12345"


# --- Template Constants Tests ---


# Test DEFAULT_SYSTEM_PROMPT contains required instructions.
def test_default_system_prompt_content():
    assert "causal reasoning" in DEFAULT_SYSTEM_PROMPT.lower()
    assert "JSON" in DEFAULT_SYSTEM_PROMPT
    assert "exists" in DEFAULT_SYSTEM_PROMPT
    assert "direction" in DEFAULT_SYSTEM_PROMPT
    assert "confidence" in DEFAULT_SYSTEM_PROMPT
    assert "reasoning" in DEFAULT_SYSTEM_PROMPT


# Test USER_PROMPT_TEMPLATE contains placeholders.
def test_user_prompt_template_placeholders():
    assert "{node_a}" in USER_PROMPT_TEMPLATE
    assert "{node_b}" in USER_PROMPT_TEMPLATE


# Test USER_PROMPT_WITH_DOMAIN_TEMPLATE contains placeholders.
def test_user_prompt_with_domain_template_placeholders():
    assert "{domain}" in USER_PROMPT_WITH_DOMAIN_TEMPLATE
    assert "{node_a}" in USER_PROMPT_WITH_DOMAIN_TEMPLATE
    assert "{node_b}" in USER_PROMPT_WITH_DOMAIN_TEMPLATE


# Test VARIABLE_DESCRIPTIONS_TEMPLATE contains placeholders.
def test_variable_descriptions_template_placeholders():
    assert "{node_a}" in VARIABLE_DESCRIPTIONS_TEMPLATE
    assert "{node_b}" in VARIABLE_DESCRIPTIONS_TEMPLATE
    assert "{desc_a}" in VARIABLE_DESCRIPTIONS_TEMPLATE
    assert "{desc_b}" in VARIABLE_DESCRIPTIONS_TEMPLATE
