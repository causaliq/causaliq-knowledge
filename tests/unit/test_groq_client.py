"""Tests for GroqClient direct API implementation."""

import httpx
import pytest

from causaliq_knowledge.llm.base_client import LLMResponse
from causaliq_knowledge.llm.groq_client import GroqClient, GroqConfig


# Test default configuration values
def test_groq_config_defaults(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    config = GroqConfig()

    assert config.model == "llama-3.1-8b-instant"
    assert config.temperature == 0.1
    assert config.max_tokens == 500
    assert config.timeout == 30.0
    assert config.api_key == "test-key"


# Test custom configuration values
def test_groq_config_custom():
    config = GroqConfig(
        model="custom-model",
        temperature=0.5,
        max_tokens=1000,
        timeout=60.0,
        api_key="custom-key",
    )

    assert config.model == "custom-model"
    assert config.temperature == 0.5
    assert config.max_tokens == 1000
    assert config.timeout == 60.0
    assert config.api_key == "custom-key"


# Test missing API key raises error
def test_groq_config_missing_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(
        ValueError,
        match="GROQ_API_KEY environment variable is required",
    ):
        GroqConfig()


# Test provider_name property returns correct value
def test_groq_client_provider_name(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    client = GroqClient()
    assert client.provider_name == "groq"


# Test is_available returns True when API key is set
def test_groq_client_is_available_with_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    client = GroqClient()
    assert client.is_available() is True


# Test is_available returns True with explicit API key
def test_groq_client_is_available_with_explicit_key():
    config = GroqConfig(api_key="explicit-key")
    client = GroqClient(config)
    assert client.is_available() is True


# Test basic response creation
def test_groq_response_basic():
    response = LLMResponse(
        content='{"test": "data"}',
        model="llama-3.1-8b-instant",
        input_tokens=10,
        output_tokens=5,
        cost=0.0,
    )

    assert response.content == '{"test": "data"}'
    assert response.model == "llama-3.1-8b-instant"
    assert response.input_tokens == 10
    assert response.output_tokens == 5
    assert response.cost == 0.0


# Test successful JSON parsing
def test_groq_response_parse_json_success():
    response = LLMResponse(
        content='{"exists": true, "confidence": 0.8}', model="test"
    )

    parsed = response.parse_json()
    assert parsed == {"exists": True, "confidence": 0.8}


# Test JSON parsing with markdown code blocks
def test_groq_response_parse_json_with_markdown():
    response = LLMResponse(
        content='```json\n{"exists": true}\n```', model="test"
    )

    parsed = response.parse_json()
    assert parsed == {"exists": True}


# Test JSON parsing with trailing markdown only
def test_groq_response_parse_json_trailing_markdown():
    response = LLMResponse(content='{"exists": false}```', model="test")

    parsed = response.parse_json()
    assert parsed == {"exists": False}


# Test JSON parsing with both leading and trailing markdown
def test_groq_response_parse_json_both_markdown():
    response = LLMResponse(content='```\n{"test": "value"}```', model="test")

    parsed = response.parse_json()
    assert parsed == {"test": "value"}


# Test JSON parsing failure
def test_groq_response_parse_json_failure():
    response = LLMResponse(content="invalid json", model="test")

    parsed = response.parse_json()
    assert parsed is None


# Test client initialization
def test_groq_client_initialization(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    client = GroqClient()

    assert client.config.model == "llama-3.1-8b-instant"
    assert client.call_count == 0


# Test client with custom configuration
def test_groq_client_with_custom_config():
    config = GroqConfig(model="custom-model", api_key="test-key")
    client = GroqClient(config)

    assert client.config.model == "custom-model"


# Test successful completion request
def test_groq_client_completion_success(monkeypatch):
    # Mock HTTP response
    mock_response_data = {
        "choices": [{"message": {"content": "Test response"}}],
        "model": "llama-3.1-8b-instant",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }

    class MockResponse:
        def json(self):
            return mock_response_data

        def raise_for_status(self):
            pass

    class MockClient:
        def post(self, *args, **kwargs):
            return MockResponse()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr("httpx.Client", lambda *args, **kwargs: MockClient())

    # Create client and test
    config = GroqConfig(api_key="test-key")
    client = GroqClient(config)

    messages = [{"role": "user", "content": "Test message"}]
    response = client.completion(messages)

    assert isinstance(response, LLMResponse)
    assert response.content == "Test response"
    assert response.input_tokens == 10
    assert response.output_tokens == 5
    assert client.call_count == 1


# Test completion with HTTP error
def test_groq_client_completion_http_error(monkeypatch):
    class MockResponse:
        status_code = 429
        text = "Rate limit exceeded"

    class MockClient:
        def post(self, *args, **kwargs):
            class MockRequest:
                pass

            raise httpx.HTTPStatusError(
                "Too Many Requests",
                request=MockRequest(),
                response=MockResponse(),
            )

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr("httpx.Client", lambda *args, **kwargs: MockClient())

    # Test error handling
    config = GroqConfig(api_key="test-key")
    client = GroqClient(config)

    messages = [{"role": "user", "content": "Test"}]
    with pytest.raises(ValueError, match="Groq API error: 429"):
        client.completion(messages)


# Test completion with timeout
def test_groq_client_completion_timeout(monkeypatch):
    class MockClient:
        def post(self, *args, **kwargs):
            raise httpx.TimeoutException("Request timeout")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr("httpx.Client", lambda *args, **kwargs: MockClient())

    config = GroqConfig(api_key="test-key")
    client = GroqClient(config)

    messages = [{"role": "user", "content": "Test"}]
    with pytest.raises(ValueError, match="Groq API request timed out"):
        client.completion(messages)


# Test completion with generic exception
def test_groq_client_completion_generic_error(monkeypatch):
    class MockClient:
        def post(self, *args, **kwargs):
            raise RuntimeError("Unexpected error")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr("httpx.Client", lambda *args, **kwargs: MockClient())

    config = GroqConfig(api_key="test-key")
    client = GroqClient(config)

    messages = [{"role": "user", "content": "Test"}]
    with pytest.raises(ValueError, match="Groq API error: Unexpected error"):
        client.completion(messages)


def test_groq_client_complete_json(monkeypatch):
    # Mock successful JSON response
    mock_response_data = {
        "choices": [
            {"message": {"content": '{"exists": true, "confidence": 0.8}'}}
        ],
        "model": "llama-3.1-8b-instant",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }

    class MockResponse:
        def json(self):
            return mock_response_data

        def raise_for_status(self):
            pass

    class MockClient:
        def post(self, *args, **kwargs):
            return MockResponse()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr("httpx.Client", lambda *args, **kwargs: MockClient())

    config = GroqConfig(api_key="test-key")
    client = GroqClient(config)

    messages = [{"role": "user", "content": "Test"}]
    parsed_json, response = client.complete_json(messages)

    assert parsed_json == {"exists": True, "confidence": 0.8}
    assert isinstance(response, LLMResponse)


# Test list_models returns available models
def test_groq_client_list_models_success(monkeypatch):
    mock_response_data = {
        "data": [
            {"id": "llama-3.1-8b-instant"},
            {"id": "llama-3.3-70b-versatile"},
            {"id": "whisper-large-v3"},  # Should be filtered
            {"id": "llama-guard-4"},  # Should be filtered
        ]
    }

    class MockResponse:
        def json(self):
            return mock_response_data

        def raise_for_status(self):
            pass

    class MockClient:
        def get(self, *args, **kwargs):
            return MockResponse()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr("httpx.Client", lambda *args, **kwargs: MockClient())

    config = GroqConfig(api_key="test-key")
    client = GroqClient(config)

    models = client.list_models()

    # Should filter out whisper and guard models
    assert "llama-3.1-8b-instant" in models
    assert "llama-3.3-70b-versatile" in models
    assert "whisper-large-v3" not in models
    assert "llama-guard-4" not in models


# Test list_models handles API error
def test_groq_client_list_models_error(monkeypatch):
    import httpx

    class MockResponse:
        status_code = 401
        text = "Unauthorized"

        def raise_for_status(self):
            raise httpx.HTTPStatusError(
                "Unauthorized",
                request=httpx.Request("GET", "http://test"),
                response=self,
            )

    class MockClient:
        def get(self, *args, **kwargs):
            return MockResponse()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr("httpx.Client", lambda *args, **kwargs: MockClient())

    config = GroqConfig(api_key="test-key")
    client = GroqClient(config)

    with pytest.raises(ValueError, match="Groq API error"):
        client.list_models()


# Test list_models handles generic exception
def test_groq_client_list_models_generic_exception(monkeypatch):
    class MockClient:
        def get(self, *args, **kwargs):
            raise RuntimeError("Network failure")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr("httpx.Client", lambda *args, **kwargs: MockClient())

    config = GroqConfig(api_key="test-key")
    client = GroqClient(config)

    with pytest.raises(ValueError, match="Failed to list Groq models"):
        client.list_models()
