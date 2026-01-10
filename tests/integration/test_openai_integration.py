"""Integration tests for OpenAI LLM client.

These tests make real API calls to OpenAI and are marked as slow.
They will be skipped if OPENAI_API_KEY is not set.

Run with: pytest -m slow tests/integration/test_openai_integration.py -v
"""

import pytest
from conftest import skip_no_openai

pytestmark = [pytest.mark.slow, pytest.mark.integration, skip_no_openai]


# Test simple completion request to OpenAI.
def test_openai_simple_completion():
    from causaliq_knowledge.llm import OpenAIClient, OpenAIConfig

    config = OpenAIConfig(
        model="gpt-4o-mini",
        temperature=0.1,
        max_tokens=100,
    )
    client = OpenAIClient(config)
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
def test_openai_system_message():
    from causaliq_knowledge.llm import OpenAIClient, OpenAIConfig

    config = OpenAIConfig(
        model="gpt-4o-mini",
        temperature=0.1,
        max_tokens=50,
    )
    client = OpenAIClient(config)
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


# Test getting a JSON response from OpenAI.
def test_openai_json_response():
    from causaliq_knowledge.llm import OpenAIClient, OpenAIConfig

    config = OpenAIConfig(
        model="gpt-4o-mini",
        temperature=0.0,
        max_tokens=200,
    )
    client = OpenAIClient(config)
    messages = [
        {
            "role": "system",
            "content": "Always respond in valid JSON format.",
        },
        {
            "role": "user",
            "content": 'Return JSON with "name" and "age" for Charlie, 35.',
        },
    ]

    parsed, response = client.complete_json(messages)

    assert response.content is not None
    if parsed is not None:
        assert "name" in parsed or "Name" in parsed
        assert "age" in parsed or "Age" in parsed


# Test the is_available method returns True with valid API key.
def test_openai_is_available():
    from causaliq_knowledge.llm import OpenAIClient, OpenAIConfig

    config = OpenAIConfig()
    client = OpenAIClient(config)

    assert client.is_available() is True


# Test listing available models from OpenAI.
def test_openai_list_models():
    from causaliq_knowledge.llm import OpenAIClient, OpenAIConfig

    config = OpenAIConfig()
    client = OpenAIClient(config)

    models = client.list_models()

    assert isinstance(models, list)
    assert len(models) > 0
    model_str = " ".join(models).lower()
    assert "gpt" in model_str


# Test the provider_name property returns 'openai'.
def test_openai_provider_name():
    from causaliq_knowledge.llm import OpenAIClient, OpenAIConfig

    config = OpenAIConfig()
    client = OpenAIClient(config)

    assert client.provider_name == "openai"


# Test that cost is calculated for responses.
def test_openai_cost_calculation():
    from causaliq_knowledge.llm import OpenAIClient, OpenAIConfig

    config = OpenAIConfig(
        model="gpt-4o-mini",
        temperature=0.1,
        max_tokens=50,
    )
    client = OpenAIClient(config)
    messages = [{"role": "user", "content": "Hi"}]

    response = client.completion(messages)

    assert response.cost >= 0
