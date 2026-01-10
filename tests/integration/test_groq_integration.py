"""Integration tests for Groq LLM client.

These tests make real API calls to Groq and are marked as slow.
They will be skipped if GROQ_API_KEY is not set.

Run with: pytest -m slow tests/integration/test_groq_integration.py -v
"""

import pytest
from conftest import skip_no_groq

pytestmark = [pytest.mark.slow, pytest.mark.integration, skip_no_groq]


# Test simple completion request to Groq.
def test_groq_simple_completion():
    from causaliq_knowledge.llm import GroqClient, GroqConfig

    config = GroqConfig(
        model="llama-3.1-8b-instant",
        temperature=0.1,
        max_tokens=100,
    )
    client = GroqClient(config)
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
def test_groq_system_message():
    from causaliq_knowledge.llm import GroqClient, GroqConfig

    config = GroqConfig(
        model="llama-3.1-8b-instant",
        temperature=0.1,
        max_tokens=50,
    )
    client = GroqClient(config)
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


# Test getting a JSON response from Groq.
def test_groq_json_response():
    from causaliq_knowledge.llm import GroqClient, GroqConfig

    config = GroqConfig(
        model="llama-3.1-8b-instant",
        temperature=0.0,
        max_tokens=200,
    )
    client = GroqClient(config)
    messages = [
        {
            "role": "system",
            "content": "Always respond in valid JSON format.",
        },
        {
            "role": "user",
            "content": 'Return JSON with "name" and "age" for Alice, 30.',
        },
    ]

    parsed, response = client.complete_json(messages)

    assert response.content is not None
    if parsed is not None:
        assert "name" in parsed or "Name" in parsed
        assert "age" in parsed or "Age" in parsed


# Test the is_available method returns True with valid API key.
def test_groq_is_available():
    from causaliq_knowledge.llm import GroqClient, GroqConfig

    config = GroqConfig()
    client = GroqClient(config)

    assert client.is_available() is True


# Test listing available models from Groq.
def test_groq_list_models():
    from causaliq_knowledge.llm import GroqClient, GroqConfig

    config = GroqConfig()
    client = GroqClient(config)

    models = client.list_models()

    assert isinstance(models, list)
    assert len(models) > 0
    model_str = " ".join(models).lower()
    assert "llama" in model_str or "mixtral" in model_str


# Test the provider_name property returns 'groq'.
def test_groq_provider_name():
    from causaliq_knowledge.llm import GroqClient, GroqConfig

    config = GroqConfig()
    client = GroqClient(config)

    assert client.provider_name == "groq"


# Test that call count increments with each API call.
def test_groq_call_count_increments():
    from causaliq_knowledge.llm import GroqClient, GroqConfig

    config = GroqConfig(
        model="llama-3.1-8b-instant",
        temperature=0.1,
        max_tokens=10,
    )
    client = GroqClient(config)
    messages = [{"role": "user", "content": "Hi"}]

    assert client.call_count == 0
    client.completion(messages)
    assert client.call_count == 1
    client.completion(messages)
    assert client.call_count == 2
