"""Integration tests for Ollama LLM client.

These tests make real API calls to a local Ollama server (marked as slow).
They will be skipped if Ollama is not running on localhost:11434.

Run with: pytest -m slow tests/integration/test_ollama_integration.py -v

Prerequisites:
    1. Install Ollama: https://ollama.ai/
    2. Pull a model: ollama pull llama3.2:1b
    3. Start Ollama server (usually runs automatically)
"""

import pytest
from conftest import skip_no_ollama

pytestmark = [pytest.mark.slow, pytest.mark.integration, skip_no_ollama]


# Test simple completion request to Ollama.
def test_ollama_simple_completion():
    from causaliq_knowledge.llm import OllamaClient, OllamaConfig

    config = OllamaConfig(
        model="llama3.2:1b",
        temperature=0.1,
        max_tokens=100,
        timeout=120.0,
    )
    client = OllamaClient(config)
    messages = [
        {"role": "user", "content": "What is 2 + 2? Reply with the number."}
    ]

    response = client.completion(messages)

    assert response.content is not None
    assert len(response.content) > 0
    assert response.model is not None
    assert client.call_count == 1
    assert "4" in response.content


# Test completion with a system message.
def test_ollama_system_message():
    from causaliq_knowledge.llm import OllamaClient, OllamaConfig

    config = OllamaConfig(
        model="llama3.2:1b",
        temperature=0.1,
        max_tokens=50,
        timeout=120.0,
    )
    client = OllamaClient(config)
    messages = [
        {"role": "system", "content": "Respond in exactly one word."},
        {"role": "user", "content": "What color is the sky?"},
    ]

    response = client.completion(messages)

    assert response.content is not None
    words = response.content.strip().split()
    assert len(words) <= 10


# Test getting a JSON response from Ollama.
def test_ollama_json_response():
    from causaliq_knowledge.llm import OllamaClient, OllamaConfig

    config = OllamaConfig(
        model="llama3.2:1b",
        temperature=0.0,
        max_tokens=200,
        timeout=120.0,
    )
    client = OllamaClient(config)
    messages = [
        {
            "role": "system",
            "content": "Always respond in valid JSON format only.",
        },
        {
            "role": "user",
            "content": 'Return a JSON object: {"greeting": "hello"}',
        },
    ]

    parsed, response = client.complete_json(messages)

    assert response.content is not None


# Test the is_available method returns True when Ollama is running.
def test_ollama_is_available():
    from causaliq_knowledge.llm import OllamaClient, OllamaConfig

    config = OllamaConfig()
    client = OllamaClient(config)

    assert client.is_available() is True


# Test listing available models from Ollama.
def test_ollama_list_models():
    from causaliq_knowledge.llm import OllamaClient, OllamaConfig

    config = OllamaConfig()
    client = OllamaClient(config)

    models = client.list_models()

    assert isinstance(models, list)
    assert len(models) >= 0


# Test the provider_name property returns 'ollama'.
def test_ollama_provider_name():
    from causaliq_knowledge.llm import OllamaClient, OllamaConfig

    config = OllamaConfig()
    client = OllamaClient(config)

    assert client.provider_name == "ollama"


# Test configuring a custom base URL for Ollama.
def test_ollama_custom_base_url():
    from causaliq_knowledge.llm import OllamaClient, OllamaConfig

    config = OllamaConfig(base_url="http://localhost:11434")
    client = OllamaClient(config)

    assert client.is_available() is True


# Test making multiple calls to Ollama increments counter.
def test_ollama_multiple_calls():
    from causaliq_knowledge.llm import OllamaClient, OllamaConfig

    config = OllamaConfig(
        model="llama3.2:1b",
        temperature=0.1,
        max_tokens=20,
        timeout=120.0,
    )
    client = OllamaClient(config)
    messages = [{"role": "user", "content": "Say hi"}]

    assert client.call_count == 0
    response1 = client.completion(messages)
    assert client.call_count == 1
    response2 = client.completion(messages)
    assert client.call_count == 2

    assert response1.content is not None
    assert response2.content is not None
