"""Tests for OllamaClient local API implementation."""

import httpx
import pytest

from causaliq_knowledge.llm.base_client import BaseLLMClient, LLMResponse
from causaliq_knowledge.llm.ollama_client import OllamaClient, OllamaConfig

# --- OllamaConfig Tests ---


# Test default configuration values
def test_ollama_config_defaults():
    config = OllamaConfig()

    assert config.model == "llama3.2:1b"
    assert config.temperature == 0.1
    assert config.max_tokens == 500
    assert config.timeout == 120.0
    assert config.api_key is None
    assert config.base_url == "http://localhost:11434"


# Test custom configuration values
def test_ollama_config_custom():
    config = OllamaConfig(
        model="llama3.1:8b",
        temperature=0.5,
        max_tokens=1000,
        timeout=60.0,
        base_url="http://192.168.1.100:11434",
    )

    assert config.model == "llama3.1:8b"
    assert config.temperature == 0.5
    assert config.max_tokens == 1000
    assert config.timeout == 60.0
    assert config.base_url == "http://192.168.1.100:11434"


# --- OllamaClient Tests ---


# Test client initialization with default config
def test_ollama_client_default_config():
    client = OllamaClient()

    assert client.config.model == "llama3.2:1b"
    assert client.config.base_url == "http://localhost:11434"
    assert client.call_count == 0


# Test client initialization with custom config
def test_ollama_client_custom_config():
    config = OllamaConfig(model="llama3.1:8b")
    client = OllamaClient(config=config)

    assert client.config.model == "llama3.1:8b"


# Test provider_name property returns correct value
def test_ollama_client_provider_name():
    client = OllamaClient()
    assert client.provider_name == "ollama"


# Test OllamaClient inherits from BaseLLMClient
def test_ollama_client_inherits_from_base():
    assert issubclass(OllamaClient, BaseLLMClient)


# Test OllamaConfig inherits from LLMConfig
def test_ollama_config_inherits_from_base():
    from causaliq_knowledge.llm.base_client import LLMConfig

    assert issubclass(OllamaConfig, LLMConfig)


# --- API Response Tests ---


# Test successful completion with mocked response
def test_ollama_completion_success(mocker):
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "model": "llama3.2:1b",
        "message": {"role": "assistant", "content": "Hello! How can I help?"},
        "prompt_eval_count": 15,
        "eval_count": 10,
    }
    mock_response.raise_for_status = mocker.Mock()

    mock_client = mocker.Mock()
    mock_client.post.return_value = mock_response
    mock_client.__enter__ = mocker.Mock(return_value=mock_client)
    mock_client.__exit__ = mocker.Mock(return_value=False)

    mocker.patch("httpx.Client", return_value=mock_client)

    client = OllamaClient()
    response = client.completion([{"role": "user", "content": "Hello"}])

    assert isinstance(response, LLMResponse)
    assert response.content == "Hello! How can I help?"
    assert response.model == "llama3.2:1b"
    assert response.input_tokens == 15
    assert response.output_tokens == 10
    assert response.cost == 0.0
    assert client.call_count == 1


# Test completion with custom options
def test_ollama_completion_with_options(mocker):
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "model": "llama3.2:1b",
        "message": {"role": "assistant", "content": "Response"},
        "prompt_eval_count": 10,
        "eval_count": 5,
    }
    mock_response.raise_for_status = mocker.Mock()

    mock_client = mocker.Mock()
    mock_client.post.return_value = mock_response
    mock_client.__enter__ = mocker.Mock(return_value=mock_client)
    mock_client.__exit__ = mocker.Mock(return_value=False)

    mocker.patch("httpx.Client", return_value=mock_client)

    client = OllamaClient()
    client.completion(
        [{"role": "user", "content": "Test"}],
        temperature=0.7,
        max_tokens=100,
    )

    # Verify the payload included custom options
    call_args = mock_client.post.call_args
    payload = call_args.kwargs["json"]
    assert payload["options"]["temperature"] == 0.7
    assert payload["options"]["num_predict"] == 100


# Test connection error when Ollama not running
def test_ollama_completion_connection_error(mocker):
    mock_client = mocker.Mock()
    mock_client.post.side_effect = httpx.ConnectError("Connection refused")
    mock_client.__enter__ = mocker.Mock(return_value=mock_client)
    mock_client.__exit__ = mocker.Mock(return_value=False)

    mocker.patch("httpx.Client", return_value=mock_client)

    client = OllamaClient()

    with pytest.raises(ValueError, match="Could not connect to Ollama"):
        client.completion([{"role": "user", "content": "Hello"}])


# Test model not found error
def test_ollama_completion_model_not_found(mocker):
    mock_response = mocker.Mock()
    mock_response.status_code = 404
    mock_response.text = "model not found"
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Not Found", request=mocker.Mock(), response=mock_response
    )

    mock_client = mocker.Mock()
    mock_client.post.return_value = mock_response
    mock_client.__enter__ = mocker.Mock(return_value=mock_client)
    mock_client.__exit__ = mocker.Mock(return_value=False)

    mocker.patch("httpx.Client", return_value=mock_client)

    client = OllamaClient()

    with pytest.raises(ValueError, match="Model 'llama3.2:1b' not found"):
        client.completion([{"role": "user", "content": "Hello"}])


# Test timeout error
def test_ollama_completion_timeout(mocker):
    mock_client = mocker.Mock()
    mock_client.post.side_effect = httpx.TimeoutException("Request timed out")
    mock_client.__enter__ = mocker.Mock(return_value=mock_client)
    mock_client.__exit__ = mocker.Mock(return_value=False)

    mocker.patch("httpx.Client", return_value=mock_client)

    client = OllamaClient()

    with pytest.raises(ValueError, match="timed out"):
        client.completion([{"role": "user", "content": "Hello"}])


# Test generic HTTP error
def test_ollama_completion_http_error(mocker):
    mock_response = mocker.Mock()
    mock_response.status_code = 500
    mock_response.text = "Internal server error"
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Server Error", request=mocker.Mock(), response=mock_response
    )

    mock_client = mocker.Mock()
    mock_client.post.return_value = mock_response
    mock_client.__enter__ = mocker.Mock(return_value=mock_client)
    mock_client.__exit__ = mocker.Mock(return_value=False)

    mocker.patch("httpx.Client", return_value=mock_client)

    client = OllamaClient()

    with pytest.raises(ValueError, match="Ollama API error: 500"):
        client.completion([{"role": "user", "content": "Hello"}])


# Test generic exception handling
def test_ollama_completion_generic_error(mocker):
    mock_client = mocker.Mock()
    mock_client.post.side_effect = Exception("Unexpected error")
    mock_client.__enter__ = mocker.Mock(return_value=mock_client)
    mock_client.__exit__ = mocker.Mock(return_value=False)

    mocker.patch("httpx.Client", return_value=mock_client)

    client = OllamaClient()

    with pytest.raises(ValueError, match="Ollama API error"):
        client.completion([{"role": "user", "content": "Hello"}])


# --- complete_json Tests ---


# Test complete_json parses JSON response
def test_ollama_complete_json_success(mocker):
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "model": "llama3.2:1b",
        "message": {
            "role": "assistant",
            "content": '{"exists": true, "confidence": 0.8}',
        },
        "prompt_eval_count": 10,
        "eval_count": 5,
    }
    mock_response.raise_for_status = mocker.Mock()

    mock_client = mocker.Mock()
    mock_client.post.return_value = mock_response
    mock_client.__enter__ = mocker.Mock(return_value=mock_client)
    mock_client.__exit__ = mocker.Mock(return_value=False)

    mocker.patch("httpx.Client", return_value=mock_client)

    client = OllamaClient()
    parsed, response = client.complete_json(
        [{"role": "user", "content": "Test"}]
    )

    assert parsed == {"exists": True, "confidence": 0.8}
    assert isinstance(response, LLMResponse)


# Test complete_json with invalid JSON returns None
def test_ollama_complete_json_invalid(mocker):
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "model": "llama3.2:1b",
        "message": {"role": "assistant", "content": "Not valid JSON"},
        "prompt_eval_count": 10,
        "eval_count": 5,
    }
    mock_response.raise_for_status = mocker.Mock()

    mock_client = mocker.Mock()
    mock_client.post.return_value = mock_response
    mock_client.__enter__ = mocker.Mock(return_value=mock_client)
    mock_client.__exit__ = mocker.Mock(return_value=False)

    mocker.patch("httpx.Client", return_value=mock_client)

    client = OllamaClient()
    parsed, response = client.complete_json(
        [{"role": "user", "content": "Test"}]
    )

    assert parsed is None
    assert response.content == "Not valid JSON"


# --- is_available Tests ---


# Test is_available returns True when server running and model exists
def test_ollama_is_available_true(mocker):
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "models": [
            {"name": "llama3.2:1b"},
            {"name": "llama3.1:8b"},
        ]
    }

    mock_client = mocker.Mock()
    mock_client.get.return_value = mock_response
    mock_client.__enter__ = mocker.Mock(return_value=mock_client)
    mock_client.__exit__ = mocker.Mock(return_value=False)

    mocker.patch("httpx.Client", return_value=mock_client)

    client = OllamaClient()
    assert client.is_available() is True


# Test is_available returns False when model not found
def test_ollama_is_available_model_not_found(mocker):
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "models": [{"name": "other-model:latest"}]
    }

    mock_client = mocker.Mock()
    mock_client.get.return_value = mock_response
    mock_client.__enter__ = mocker.Mock(return_value=mock_client)
    mock_client.__exit__ = mocker.Mock(return_value=False)

    mocker.patch("httpx.Client", return_value=mock_client)

    client = OllamaClient()
    assert client.is_available() is False


# Test is_available returns False when server not running
def test_ollama_is_available_server_down(mocker):
    mock_client = mocker.Mock()
    mock_client.get.side_effect = httpx.ConnectError("Connection refused")
    mock_client.__enter__ = mocker.Mock(return_value=mock_client)
    mock_client.__exit__ = mocker.Mock(return_value=False)

    mocker.patch("httpx.Client", return_value=mock_client)

    client = OllamaClient()
    assert client.is_available() is False


# Test is_available returns false on non-200 status code
def test_ollama_is_available_non_200_status(mocker):
    mock_response = mocker.Mock()
    mock_response.status_code = 500

    mock_client = mocker.Mock()
    mock_client.get.return_value = mock_response
    mock_client.__enter__ = mocker.Mock(return_value=mock_client)
    mock_client.__exit__ = mocker.Mock(return_value=False)

    mocker.patch("httpx.Client", return_value=mock_client)

    client = OllamaClient()
    assert client.is_available() is False


# Test is_available with model name that has :latest suffix
def test_ollama_is_available_with_latest_suffix(mocker):
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "models": [{"name": "llama3.2:1b:latest"}]
    }

    mock_client = mocker.Mock()
    mock_client.get.return_value = mock_response
    mock_client.__enter__ = mocker.Mock(return_value=mock_client)
    mock_client.__exit__ = mocker.Mock(return_value=False)

    mocker.patch("httpx.Client", return_value=mock_client)

    client = OllamaClient()
    assert client.is_available() is True


# --- list_models Tests ---


# Test list_models returns installed models
def test_ollama_list_models_success(mocker):
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "models": [
            {"name": "llama3.2:1b"},
            {"name": "mistral:7b"},
        ]
    }
    mock_response.raise_for_status = mocker.Mock()

    mock_client = mocker.Mock()
    mock_client.get.return_value = mock_response
    mock_client.__enter__ = mocker.Mock(return_value=mock_client)
    mock_client.__exit__ = mocker.Mock(return_value=False)

    mocker.patch("httpx.Client", return_value=mock_client)

    client = OllamaClient()
    models = client.list_models()

    assert "llama3.2:1b" in models
    assert "mistral:7b" in models


# Test list_models returns empty list when no models installed
def test_ollama_list_models_empty(mocker):
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"models": []}
    mock_response.raise_for_status = mocker.Mock()

    mock_client = mocker.Mock()
    mock_client.get.return_value = mock_response
    mock_client.__enter__ = mocker.Mock(return_value=mock_client)
    mock_client.__exit__ = mocker.Mock(return_value=False)

    mocker.patch("httpx.Client", return_value=mock_client)

    client = OllamaClient()
    models = client.list_models()

    assert models == []


# Test list_models raises error when server not running
def test_ollama_list_models_server_down(mocker):
    import httpx

    mock_client = mocker.Mock()
    mock_client.get.side_effect = httpx.ConnectError("Connection refused")
    mock_client.__enter__ = mocker.Mock(return_value=mock_client)
    mock_client.__exit__ = mocker.Mock(return_value=False)

    mocker.patch("httpx.Client", return_value=mock_client)

    client = OllamaClient()

    with pytest.raises(ValueError, match="Ollama server not running"):
        client.list_models()


# Test list_models handles generic exception
def test_ollama_list_models_generic_exception(mocker):
    mock_client = mocker.Mock()
    mock_client.get.side_effect = RuntimeError("Network failure")
    mock_client.__enter__ = mocker.Mock(return_value=mock_client)
    mock_client.__exit__ = mocker.Mock(return_value=False)

    mocker.patch("httpx.Client", return_value=mock_client)

    client = OllamaClient()

    with pytest.raises(ValueError, match="Failed to list Ollama models"):
        client.list_models()
