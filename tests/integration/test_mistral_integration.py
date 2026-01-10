"""Integration tests for Mistral AI LLM client.

These tests make real API calls to Mistral and are marked as slow.
They will be skipped if MISTRAL_API_KEY is not set.

Run with: pytest -m slow tests/integration/test_mistral_integration.py -v
"""

import pytest
from conftest import skip_no_mistral

pytestmark = [pytest.mark.slow, pytest.mark.integration, skip_no_mistral]


# Test simple completion request to Mistral.
def test_mistral_simple_completion():
    from causaliq_knowledge.llm import MistralClient, MistralConfig

    config = MistralConfig(
        model="mistral-small-latest",
        temperature=0.1,
        max_tokens=100,
    )
    client = MistralClient(config)
    messages = [
        {"role": "user", "content": "What is 2 + 2? Reply with the number."}
    ]

    response = client.completion(messages)

    assert response.content is not None
    assert len(response.content) > 0
    assert response.model is not None
    assert response.input_tokens > 0
    assert response.output_tokens > 0
    assert client.call_count == 1
    assert "4" in response.content


# Test completion with a system message.
def test_mistral_system_message():
    from causaliq_knowledge.llm import MistralClient, MistralConfig

    config = MistralConfig(
        model="mistral-small-latest",
        temperature=0.1,
        max_tokens=50,
    )
    client = MistralClient(config)
    messages = [
        {
            "role": "system",
            "content": "You are an assistant that responds in uppercase.",
        },
        {"role": "user", "content": "Say hello."},
    ]

    response = client.completion(messages)

    assert response.content is not None
    upper_count = sum(1 for c in response.content if c.isupper())
    alpha_count = sum(1 for c in response.content if c.isalpha())
    if alpha_count > 0:
        assert upper_count / alpha_count >= 0.5


# Test getting a JSON response from Mistral.
def test_mistral_json_response():
    from causaliq_knowledge.llm import MistralClient, MistralConfig

    config = MistralConfig(
        model="mistral-small-latest",
        temperature=0.0,
        max_tokens=200,
    )
    client = MistralClient(config)
    messages = [
        {
            "role": "system",
            "content": "Always respond in valid JSON format.",
        },
        {
            "role": "user",
            "content": 'Return JSON with "name" and "age" for Grace, 22.',
        },
    ]

    parsed, response = client.complete_json(messages)

    assert response.content is not None
    if parsed is not None:
        assert "name" in parsed or "Name" in parsed
        assert "age" in parsed or "Age" in parsed


# Test the is_available method returns True with valid API key.
def test_mistral_is_available():
    from causaliq_knowledge.llm import MistralClient, MistralConfig

    config = MistralConfig()
    client = MistralClient(config)

    assert client.is_available() is True


# Test listing available models from Mistral.
def test_mistral_list_models():
    from causaliq_knowledge.llm import MistralClient, MistralConfig

    config = MistralConfig()
    client = MistralClient(config)

    models = client.list_models()

    assert isinstance(models, list)
    assert len(models) > 0
    model_str = " ".join(models).lower()
    assert "mistral" in model_str or "codestral" in model_str


# Test the provider_name property returns 'mistral'.
def test_mistral_provider_name():
    from causaliq_knowledge.llm import MistralClient, MistralConfig

    config = MistralConfig()
    client = MistralClient(config)

    assert client.provider_name == "mistral"


# Test that cost is calculated for responses.
def test_mistral_cost_calculation():
    from causaliq_knowledge.llm import MistralClient, MistralConfig

    config = MistralConfig(
        model="mistral-small-latest",
        temperature=0.1,
        max_tokens=50,
    )
    client = MistralClient(config)
    messages = [{"role": "user", "content": "Hi"}]

    response = client.completion(messages)

    assert response.cost >= 0
