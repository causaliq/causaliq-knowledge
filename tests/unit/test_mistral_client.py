"""Unit tests for Mistral client."""

import httpx
import pytest

from causaliq_knowledge.llm.mistral_client import MistralClient, MistralConfig

# --- MistralConfig Tests ---


# Test default configuration
def test_mistral_config_defaults(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    config = MistralConfig()

    assert config.model == "mistral-small-latest"
    assert config.temperature == 0.1
    assert config.max_tokens == 500
    assert config.timeout == 30.0
    assert config.api_key == "test-key"


# Test custom configuration
def test_mistral_config_custom(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "default-key")
    config = MistralConfig(
        model="mistral-large-latest",
        temperature=0.5,
        max_tokens=1000,
        timeout=60.0,
        api_key="custom-key",
    )

    assert config.model == "mistral-large-latest"
    assert config.temperature == 0.5
    assert config.max_tokens == 1000
    assert config.timeout == 60.0
    assert config.api_key == "custom-key"


# Test config requires API key
def test_mistral_config_requires_api_key(monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)

    with pytest.raises(ValueError, match="MISTRAL_API_KEY"):
        MistralConfig()


# Test config reads from env var
def test_mistral_config_env_var(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "env-api-key")
    config = MistralConfig()

    assert config.api_key == "env-api-key"


# --- MistralClient Tests ---


# Test client initialization
def test_mistral_client_init(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    client = MistralClient()

    assert client.config.model == "mistral-small-latest"
    assert client.provider_name == "mistral"
    assert client.call_count == 0


# Test client with custom config
def test_mistral_client_custom_config(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    config = MistralConfig(model="mistral-large-latest", api_key="custom-key")
    client = MistralClient(config)

    assert client.config.model == "mistral-large-latest"
    assert client.config.api_key == "custom-key"


# Test is_available with key
def test_mistral_is_available_with_key(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    client = MistralClient()

    assert client.is_available() is True


# Test is_available without key
def test_mistral_is_available_without_key(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    config = MistralConfig()
    config.api_key = None  # Clear after init
    client = MistralClient(config)

    assert client.is_available() is False


# Test _default_config returns MistralConfig
def test_mistral_default_config(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    client = MistralClient()
    default = client._default_config()

    assert isinstance(default, MistralConfig)
    assert default.model == "mistral-small-latest"
    assert default.api_key == "test-key"


# --- Completion Tests ---


# Test successful completion
def test_mistral_completion_success(mocker, monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")

    mock_response = mocker.Mock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Hello, world!"}}],
        "model": "mistral-small-latest",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    mock_response.raise_for_status = mocker.Mock()

    mock_client = mocker.Mock()
    mock_client.post.return_value = mock_response
    mock_client.__enter__ = mocker.Mock(return_value=mock_client)
    mock_client.__exit__ = mocker.Mock(return_value=False)

    mocker.patch("httpx.Client", return_value=mock_client)

    client = MistralClient()
    messages = [{"role": "user", "content": "Hello"}]
    response = client.completion(messages)

    assert response.content == "Hello, world!"
    assert response.model == "mistral-small-latest"
    assert response.input_tokens == 10
    assert response.output_tokens == 5
    assert client.call_count == 1


# Test completion with custom params
def test_mistral_completion_custom_params(mocker, monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")

    mock_response = mocker.Mock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Response"}}],
        "model": "mistral-large-latest",
        "usage": {"prompt_tokens": 20, "completion_tokens": 10},
    }
    mock_response.raise_for_status = mocker.Mock()

    mock_client = mocker.Mock()
    mock_client.post.return_value = mock_response
    mock_client.__enter__ = mocker.Mock(return_value=mock_client)
    mock_client.__exit__ = mocker.Mock(return_value=False)

    mocker.patch("httpx.Client", return_value=mock_client)

    config = MistralConfig(model="mistral-large-latest", api_key="test-key")
    client = MistralClient(config)
    messages = [{"role": "user", "content": "Think step by step"}]
    response = client.completion(messages, temperature=0.7, max_tokens=2000)

    assert response.content == "Response"

    # Check that custom params were passed
    call_args = mock_client.post.call_args
    payload = call_args.kwargs["json"]
    assert payload["temperature"] == 0.7
    assert payload["max_tokens"] == 2000


# Test completion HTTP error
def test_mistral_completion_http_error(mocker, monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "bad-key")

    mock_response = mocker.Mock()
    mock_response.status_code = 401
    mock_response.text = "Unauthorized"

    def raise_for_status():
        raise httpx.HTTPStatusError(
            "Error",
            request=httpx.Request("POST", "http://test"),
            response=mock_response,
        )

    mock_response.raise_for_status = raise_for_status

    mock_client = mocker.Mock()
    mock_client.post.return_value = mock_response
    mock_client.__enter__ = mocker.Mock(return_value=mock_client)
    mock_client.__exit__ = mocker.Mock(return_value=False)

    mocker.patch("httpx.Client", return_value=mock_client)

    client = MistralClient()
    messages = [{"role": "user", "content": "Hello"}]

    with pytest.raises(ValueError, match="(?i)mistral API error"):
        client.completion(messages)


# Test completion timeout
def test_mistral_completion_timeout(mocker, monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")

    mock_client = mocker.Mock()
    mock_client.post.side_effect = httpx.TimeoutException("Timeout")
    mock_client.__enter__ = mocker.Mock(return_value=mock_client)
    mock_client.__exit__ = mocker.Mock(return_value=False)

    mocker.patch("httpx.Client", return_value=mock_client)

    client = MistralClient()
    messages = [{"role": "user", "content": "Hello"}]

    with pytest.raises(ValueError, match="(?i)mistral API request timed out"):
        client.completion(messages)


# --- List Models Tests ---


# Test list_models success
def test_mistral_list_models_success(mocker, monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")

    mock_response = mocker.Mock()
    mock_response.json.return_value = {
        "data": [
            {"id": "mistral-small-latest"},
            {"id": "mistral-large-latest"},
            {"id": "codestral-latest"},
            {"id": "mistral-embed"},  # Should be filtered out
        ]
    }
    mock_response.raise_for_status = mocker.Mock()

    mock_client = mocker.Mock()
    mock_client.get.return_value = mock_response
    mock_client.__enter__ = mocker.Mock(return_value=mock_client)
    mock_client.__exit__ = mocker.Mock(return_value=False)

    mocker.patch("httpx.Client", return_value=mock_client)

    client = MistralClient()
    models = client.list_models()

    # Should filter out embedding models
    assert "mistral-small-latest" in models
    assert "mistral-large-latest" in models
    assert "codestral-latest" in models
    assert "mistral-embed" not in models


# Test list_models API error
def test_mistral_list_models_api_error(mocker, monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")

    mock_response = mocker.Mock()
    mock_response.status_code = 500
    mock_response.text = "Server Error"

    mock_client = mocker.Mock()
    mock_client.get.side_effect = httpx.HTTPStatusError(
        "Error",
        request=httpx.Request("GET", "http://test"),
        response=mock_response,
    )
    mock_client.__enter__ = mocker.Mock(return_value=mock_client)
    mock_client.__exit__ = mocker.Mock(return_value=False)

    mocker.patch("httpx.Client", return_value=mock_client)

    client = MistralClient()

    with pytest.raises(ValueError, match="(?i)mistral API error"):
        client.list_models()


# --- Cost Calculation Tests ---


# Test cost calculation for mistral-small
def test_mistral_cost_calculation_small(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    config = MistralConfig(model="mistral-small-latest")
    client = MistralClient(config)

    # mistral-small: $0.20/1M input, $0.60/1M output
    cost = client._calculate_cost("mistral-small-latest", 1_000_000, 1_000_000)

    assert cost == pytest.approx(0.80, rel=0.01)  # 0.20 + 0.60


# Test cost calculation for mistral-large
def test_mistral_cost_calculation_large(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    config = MistralConfig(model="mistral-large-latest")
    client = MistralClient(config)

    # mistral-large: $2.00/1M input, $6.00/1M output
    cost = client._calculate_cost("mistral-large-latest", 1_000_000, 1_000_000)

    assert cost == pytest.approx(8.00, rel=0.01)  # 2.00 + 6.00


# Test cost calculation for codestral
def test_mistral_cost_calculation_codestral(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    config = MistralConfig(model="codestral-latest")
    client = MistralClient(config)

    # codestral: $0.20/1M input, $0.60/1M output
    cost = client._calculate_cost("codestral-latest", 1_000_000, 1_000_000)

    assert cost == pytest.approx(0.80, rel=0.01)


# Test cost calculation for unknown model
def test_mistral_cost_calculation_unknown(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    config = MistralConfig(model="unknown-model", api_key="test-key")
    client = MistralClient(config)

    cost = client._calculate_cost("unknown-model", 1000, 1000)

    assert cost == 0.0


# --- complete_json Tests ---


# Test complete_json success
def test_mistral_complete_json_success(mocker, monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")

    mock_response = mocker.Mock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": '{"result": "success"}'}}],
        "model": "mistral-small-latest",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    mock_response.raise_for_status = mocker.Mock()

    mock_client = mocker.Mock()
    mock_client.post.return_value = mock_response
    mock_client.__enter__ = mocker.Mock(return_value=mock_client)
    mock_client.__exit__ = mocker.Mock(return_value=False)

    mocker.patch("httpx.Client", return_value=mock_client)

    client = MistralClient()
    messages = [{"role": "user", "content": "Return JSON"}]
    parsed, response = client.complete_json(messages)

    assert parsed == {"result": "success"}
    assert response.content == '{"result": "success"}'
