"""Tests for GeminiClient direct API implementation."""

import httpx
import pytest

from causaliq_knowledge.llm.base_client import LLMResponse
from causaliq_knowledge.llm.gemini_client import GeminiClient, GeminiConfig


# Test default configuration values
def test_gemini_config_defaults(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    config = GeminiConfig()

    assert config.model == "gemini-2.5-flash"
    assert config.temperature == 0.1
    assert config.max_tokens == 500
    assert config.timeout == 30.0
    assert config.api_key == "test-key"


# Test custom configuration values
def test_gemini_config_custom():
    config = GeminiConfig(
        model="gemini-1.5-pro",
        temperature=0.5,
        max_tokens=1000,
        timeout=60.0,
        api_key="custom-key",
    )

    assert config.model == "gemini-1.5-pro"
    assert config.temperature == 0.5
    assert config.max_tokens == 1000
    assert config.timeout == 60.0
    assert config.api_key == "custom-key"


# Test missing API key raises error
def test_gemini_config_missing_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(
        ValueError,
        match="GEMINI_API_KEY environment variable is required",
    ):
        GeminiConfig()


# Test provider_name property returns correct value
def test_gemini_client_provider_name(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    client = GeminiClient()
    assert client.provider_name == "gemini"


# Test basic response creation
def test_gemini_response_basic():
    response = LLMResponse(
        content='{"test": "data"}',
        model="gemini-2.5-flash",
        input_tokens=10,
        output_tokens=5,
        cost=0.0,
    )

    assert response.content == '{"test": "data"}'
    assert response.model == "gemini-2.5-flash"
    assert response.input_tokens == 10
    assert response.output_tokens == 5
    assert response.cost == 0.0


# Test successful JSON parsing
def test_gemini_response_parse_json_success():
    response = LLMResponse(
        content='{"exists": true, "confidence": 0.8}', model="test"
    )

    parsed = response.parse_json()
    assert parsed == {"exists": True, "confidence": 0.8}


# Test JSON parsing with markdown code blocks
def test_gemini_response_parse_json_with_markdown():
    response = LLMResponse(
        content='```json\n{"exists": false}\n```', model="test"
    )

    parsed = response.parse_json()
    assert parsed == {"exists": False}


# Test JSON parsing failure
def test_gemini_response_parse_json_failure():
    response = LLMResponse(content="invalid json content", model="test")

    parsed = response.parse_json()
    assert parsed is None


# Test client initialization
def test_gemini_client_initialization(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    client = GeminiClient()

    assert client.config.model == "gemini-2.5-flash"
    assert client.call_count == 0


# Test client with custom configuration
def test_gemini_client_with_custom_config():
    config = GeminiConfig(model="gemini-1.5-pro", api_key="test-key")
    client = GeminiClient(config)

    assert client.config.model == "gemini-1.5-pro"


# Test successful completion request
def test_gemini_client_completion_success(monkeypatch):
    # Mock HTTP response
    mock_response_data = {
        "candidates": [
            {
                "content": {"parts": [{"text": "Test response from Gemini"}]},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 15,
            "candidatesTokenCount": 8,
        },
    }

    # Mock httpx.Client
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
    config = GeminiConfig(api_key="test-key")
    client = GeminiClient(config)

    messages = [{"role": "user", "content": "Test message"}]
    response = client.completion(messages)

    assert isinstance(response, LLMResponse)
    assert response.content == "Test response from Gemini"
    assert response.input_tokens == 15
    assert response.output_tokens == 8
    assert client.call_count == 1


# Test completion with system message
def test_gemini_client_completion_with_system_message(monkeypatch):
    mock_response_data = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "Response with system context"}]
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 20,
            "candidatesTokenCount": 10,
        },
    }

    posted_payload = None

    class MockResponse:
        def json(self):
            return mock_response_data

        def raise_for_status(self):
            pass

    class MockClient:
        def post(self, *args, **kwargs):
            nonlocal posted_payload
            posted_payload = kwargs.get("json")
            return MockResponse()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr("httpx.Client", lambda *args, **kwargs: MockClient())

    config = GeminiConfig(api_key="test-key")
    client = GeminiClient(config)

    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Test message"},
    ]
    response = client.completion(messages)

    assert response.content == "Response with system context"
    # Verify the POST call included systemInstruction
    assert posted_payload is not None
    assert "systemInstruction" in posted_payload


# Test completion with API error in response
def test_gemini_client_completion_api_error_response(monkeypatch):
    mock_response_data = {"error": {"message": "Invalid API key"}}

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

    config = GeminiConfig(api_key="test-key")
    client = GeminiClient(config)

    messages = [{"role": "user", "content": "Test"}]
    with pytest.raises(ValueError, match="Gemini API error: Invalid API key"):
        client.completion(messages)


# Test completion blocked by safety filters
def test_gemini_client_completion_safety_filter(monkeypatch):
    mock_response_data = {"candidates": [{"finishReason": "SAFETY"}]}

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

    config = GeminiConfig(api_key="test-key")
    client = GeminiClient(config)

    messages = [{"role": "user", "content": "Test"}]
    with pytest.raises(
        ValueError, match="Content was blocked by Gemini safety filters"
    ):
        client.completion(messages)


# Test completion with HTTP error
def test_gemini_client_completion_http_error(monkeypatch):
    class MockResponse:
        status_code = 503
        text = "Service unavailable"

        def json(self):
            return {"error": {"message": "Service unavailable"}}

    class MockClient:
        def post(self, *args, **kwargs):
            class MockRequest:
                pass

            raise httpx.HTTPStatusError(
                "Service Unavailable",
                request=MockRequest(),
                response=MockResponse(),
            )

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr("httpx.Client", lambda *args, **kwargs: MockClient())

    config = GeminiConfig(api_key="test-key")
    client = GeminiClient(config)

    messages = [{"role": "user", "content": "Test"}]
    with pytest.raises(ValueError, match="Gemini API error: 503"):
        client.completion(messages)


# Test complete_json method
def test_gemini_client_complete_json(monkeypatch):
    # Mock successful JSON response
    mock_response_data = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": (
                                '```json\n{"exists": false, '
                                '"confidence": 0.9}\n```'
                            )
                        }
                    ]
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 12,
            "candidatesTokenCount": 6,
        },
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

    config = GeminiConfig(api_key="test-key")
    client = GeminiClient(config)

    messages = [{"role": "user", "content": "Test"}]
    parsed_json, response = client.complete_json(messages)

    assert parsed_json == {"exists": False, "confidence": 0.9}
    assert isinstance(response, LLMResponse)


# Test LLMResponse JSON parsing with logging on JSONDecodeError
def test_gemini_response_json_parse_with_logging():
    """Test JSONDecodeError with logging (lines 58-60)."""
    # Note: This is unreachable code due to duplicate exception handling
    # The first JSONDecodeError handler at line 57 catches all cases
    # But we'll test normal JSON parsing error for coverage
    response = LLMResponse("invalid json", "test-model")
    result = response.parse_json()
    assert result is None


# Test LLMResponse JSON parsing with generic code block markers
def test_gemini_response_json_parse_generic_code_blocks():
    """Test parsing JSON with generic ``` code block markers (line 51)."""
    # Test the elif condition for generic ``` markers
    response = LLMResponse('```{"test": "value"}```', "test-model")
    result = response.parse_json()
    assert result == {"test": "value"}


# Test exact safety filter condition (line 135)
def test_gemini_safety_filter_exact_condition(monkeypatch):
    """Test the exact safety filter condition to cover line 135."""

    # Create a minimal response that will trigger the safety condition
    def mock_post(self, url, **kwargs):
        class MockResponse:
            def json(self):
                return {
                    "candidates": [
                        {"finishReason": "SAFETY"}
                    ]  # Minimal structure to trigger line 135
                }

            def raise_for_status(self):
                pass

        return MockResponse()

    class MockClient:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        post = mock_post

    monkeypatch.setattr("httpx.Client", lambda *args, **kwargs: MockClient())

    config = GeminiConfig(api_key="test-key")
    client = GeminiClient(config)

    # This must trigger: if candidate.get("finishReason") == "SAFETY":
    with pytest.raises(
        ValueError, match="Content was blocked by Gemini safety filters"
    ):
        client.completion([{"role": "user", "content": "test"}])


# Test no candidates returned error (line 135)
def test_gemini_no_candidates_returned(monkeypatch):
    """Test error when API returns no candidates (line 135)."""

    def mock_post(self, url, **kwargs):
        class MockResponse:
            def json(self):
                return {"candidates": []}  # Empty candidates list

            def raise_for_status(self):
                pass

        return MockResponse()

    class MockClient:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        post = mock_post

    monkeypatch.setattr("httpx.Client", lambda *args, **kwargs: MockClient())

    config = GeminiConfig(api_key="test-key")
    client = GeminiClient(config)

    # This should trigger: if not candidates:
    # raise ValueError("No candidates returned by Gemini API")
    with pytest.raises(
        ValueError, match="No candidates returned by Gemini API"
    ):
        client.completion([{"role": "user", "content": "test"}])


# Test complete_text with assistant role messages
def test_gemini_complete_text_with_assistant_role(monkeypatch):
    """Test assistant role message processing (lines 90-91)."""

    mock_response_data = {
        "candidates": [
            {
                "content": {"parts": [{"text": "Gemini response"}]},
                "finishReason": "STOP",
            }
        ]
    }

    class MockResponse:
        def json(self):
            return mock_response_data

        def raise_for_status(self):
            pass  # No error for successful response

    class MockClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

        def post(self, url, **kwargs):
            # Verify assistant role was converted to model role
            json_data = kwargs.get("json", {})
            contents = json_data.get("contents", [])
            assistant_msg = next(
                (msg for msg in contents if msg["role"] == "model"), None
            )
            assert assistant_msg is not None
            assert assistant_msg["parts"][0]["text"] == "Assistant response"

            return MockResponse()

    monkeypatch.setattr("httpx.Client", lambda *args, **kwargs: MockClient())

    config = GeminiConfig(api_key="test-key")
    client = GeminiClient(config)

    messages = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "User message"},
        {"role": "assistant", "content": "Assistant response"},
        {"role": "user", "content": "Follow up"},
    ]

    response = client.completion(messages)
    assert response.content == "Gemini response"


# Test complete_text with safety filter blocking
def test_gemini_complete_text_safety_blocked(monkeypatch):
    """Test safety filter blocking response (line 138)."""

    mock_response_data = {
        "candidates": [
            {"content": {"parts": [{"text": ""}]}, "finishReason": "SAFETY"}
        ]
    }

    class MockResponse:
        def json(self):
            return mock_response_data

        def raise_for_status(self):
            pass  # No HTTP error for successful response

    class MockClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

        def post(self, url, **kwargs):
            return MockResponse()

    monkeypatch.setattr("httpx.Client", lambda *args, **kwargs: MockClient())

    config = GeminiConfig(api_key="test-key")
    client = GeminiClient(config)

    messages = [{"role": "user", "content": "Test message"}]

    with pytest.raises(
        ValueError, match="Content was blocked by Gemini safety filters"
    ):
        client.completion(messages)


# Test HTTP error with invalid JSON response
def test_gemini_http_error_invalid_json(monkeypatch):
    """Test HTTP error with invalid JSON in error response (lines 181-182)."""

    class MockResponse:
        status_code = 400
        text = "Invalid request"

        def json(self):
            raise ValueError("Invalid JSON")

    class MockClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

        def post(self, url, **kwargs):
            response = MockResponse()
            raise httpx.HTTPStatusError(
                "400 Bad Request", request=None, response=response
            )

    monkeypatch.setattr("httpx.Client", lambda *args, **kwargs: MockClient())

    config = GeminiConfig(api_key="test-key")
    client = GeminiClient(config)

    messages = [{"role": "user", "content": "Test"}]

    with pytest.raises(
        ValueError, match="Gemini API error: 400 - Invalid request"
    ):
        client.completion(messages)


# Test timeout exception handling
def test_gemini_timeout_exception(monkeypatch):
    """Test timeout exception handling (line 192)."""

    class MockClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

        def post(self, url, **kwargs):
            raise httpx.TimeoutException("Request timed out")

    monkeypatch.setattr("httpx.Client", lambda *args, **kwargs: MockClient())

    config = GeminiConfig(api_key="test-key")
    client = GeminiClient(config)

    messages = [{"role": "user", "content": "Test"}]

    with pytest.raises(ValueError, match="Gemini API request timed out"):
        client.completion(messages)
