"""Tests for GraphGenerator class."""

from __future__ import annotations

import pytest

from causaliq_knowledge.graph.generator import (
    GraphGenerator,
    GraphGeneratorConfig,
)
from causaliq_knowledge.graph.prompts import OutputFormat
from causaliq_knowledge.graph.response import GeneratedGraph
from causaliq_knowledge.graph.view_filter import PromptDetail
from causaliq_knowledge.llm.base_client import LLMResponse


# Fixture to set fake API keys for all tests in this module.
@pytest.fixture(autouse=True)
def mock_api_keys(monkeypatch):
    """Set fake API keys to allow client instantiation without real keys."""
    monkeypatch.setenv("GROQ_API_KEY", "fake-groq-key-for-testing")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key-for-testing")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key-for-testing")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-anthropic-key-for-testing")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-deepseek-key-for-testing")
    monkeypatch.setenv("MISTRAL_API_KEY", "fake-mistral-key-for-testing")


# --- GraphGeneratorConfig tests ---


# Test GraphGeneratorConfig default values.
def test_generator_config_defaults() -> None:
    config = GraphGeneratorConfig()
    assert config.temperature == 0.1
    assert config.max_tokens == 2000
    assert config.timeout == 60.0
    assert config.output_format == OutputFormat.EDGE_LIST
    assert config.prompt_detail == PromptDetail.STANDARD


# Test GraphGeneratorConfig custom values.
def test_generator_config_custom_values() -> None:
    config = GraphGeneratorConfig(
        temperature=0.5,
        max_tokens=1000,
        timeout=30.0,
        output_format=OutputFormat.ADJACENCY_MATRIX,
        prompt_detail=PromptDetail.RICH,
    )
    assert config.temperature == 0.5
    assert config.max_tokens == 1000
    assert config.timeout == 30.0
    assert config.output_format == OutputFormat.ADJACENCY_MATRIX
    assert config.prompt_detail == PromptDetail.RICH


# --- GraphGenerator creation tests ---


# Test GraphGenerator raises on unsupported model prefix.
def test_generator_raises_on_unsupported_model() -> None:
    with pytest.raises(ValueError, match="not supported"):
        GraphGenerator(model="unsupported/model")


# Test GraphGenerator with groq model creates GroqClient.
def test_generator_creates_groq_client(mocker) -> None:
    mock_groq = mocker.patch("causaliq_knowledge.graph.generator.GroqClient")
    mock_client = mocker.MagicMock()
    mock_groq.return_value = mock_client
    generator = GraphGenerator(model="groq/llama-3.1-8b-instant")
    assert generator.model == "groq/llama-3.1-8b-instant"
    mock_groq.assert_called_once()


# Test GraphGenerator with gemini model creates GeminiClient.
def test_generator_creates_gemini_client(mocker) -> None:
    mock_gemini = mocker.patch(
        "causaliq_knowledge.graph.generator.GeminiClient"
    )
    mock_client = mocker.MagicMock()
    mock_gemini.return_value = mock_client
    GraphGenerator(model="gemini/gemini-2.5-flash")
    mock_gemini.assert_called_once()


# Test GraphGenerator with openai model creates OpenAIClient.
def test_generator_creates_openai_client(mocker) -> None:
    mock_openai = mocker.patch(
        "causaliq_knowledge.graph.generator.OpenAIClient"
    )
    mock_client = mocker.MagicMock()
    mock_openai.return_value = mock_client
    GraphGenerator(model="openai/gpt-4o")
    mock_openai.assert_called_once()


# Test GraphGenerator with anthropic model creates AnthropicClient.
def test_generator_creates_anthropic_client(mocker) -> None:
    mock_anthropic = mocker.patch(
        "causaliq_knowledge.graph.generator.AnthropicClient"
    )
    mock_client = mocker.MagicMock()
    mock_anthropic.return_value = mock_client
    GraphGenerator(model="anthropic/claude-3-5-sonnet-20241022")
    mock_anthropic.assert_called_once()


# Test GraphGenerator with deepseek model creates DeepSeekClient.
def test_generator_creates_deepseek_client(mocker) -> None:
    mock_deepseek = mocker.patch(
        "causaliq_knowledge.graph.generator.DeepSeekClient"
    )
    mock_client = mocker.MagicMock()
    mock_deepseek.return_value = mock_client
    GraphGenerator(model="deepseek/deepseek-chat")
    mock_deepseek.assert_called_once()


# Test GraphGenerator with mistral model creates MistralClient.
def test_generator_creates_mistral_client(mocker) -> None:
    mock_mistral = mocker.patch(
        "causaliq_knowledge.graph.generator.MistralClient"
    )
    mock_client = mocker.MagicMock()
    mock_mistral.return_value = mock_client
    GraphGenerator(model="mistral/mistral-small-latest")
    mock_mistral.assert_called_once()


# Test GraphGenerator with ollama model creates OllamaClient.
def test_generator_creates_ollama_client(mocker) -> None:
    mock_ollama = mocker.patch(
        "causaliq_knowledge.graph.generator.OllamaClient"
    )
    mock_client = mocker.MagicMock()
    mock_ollama.return_value = mock_client
    GraphGenerator(model="ollama/llama3.2:1b")
    mock_ollama.assert_called_once()


# --- GraphGenerator properties tests ---


# Test GraphGenerator model property.
def test_generator_model_property(mocker) -> None:
    mock_groq = mocker.patch("causaliq_knowledge.graph.generator.GroqClient")
    mock_groq.return_value = mocker.MagicMock()
    generator = GraphGenerator(model="groq/test-model")
    assert generator.model == "groq/test-model"


# Test GraphGenerator config property.
def test_generator_config_property(mocker) -> None:
    mock_groq = mocker.patch("causaliq_knowledge.graph.generator.GroqClient")
    mock_groq.return_value = mocker.MagicMock()
    config = GraphGeneratorConfig(temperature=0.5)
    generator = GraphGenerator(model="groq/test-model", config=config)
    assert generator.config.temperature == 0.5


# Test GraphGenerator call_count property starts at zero.
def test_generator_call_count_starts_zero(mocker) -> None:
    mock_groq = mocker.patch("causaliq_knowledge.graph.generator.GroqClient")
    mock_groq.return_value = mocker.MagicMock()
    generator = GraphGenerator(model="groq/test-model")
    assert generator.call_count == 0


# --- GraphGenerator generate_graph tests ---


# Test generate_graph returns GeneratedGraph.
def test_generate_graph_returns_graph(mocker) -> None:
    mock_groq = mocker.patch("causaliq_knowledge.graph.generator.GroqClient")
    mock_client = mocker.MagicMock()
    edge_json = (
        '{"edges": [{"source": "a", "target": "b", "confidence": 0.8}]}'
    )
    mock_client.completion.return_value = LLMResponse(
        content=edge_json,
        model="test-model",
        input_tokens=100,
        output_tokens=50,
    )
    mock_client.use_cache = False
    mock_groq.return_value = mock_client

    generator = GraphGenerator(model="groq/test-model")
    variables = [{"name": "a"}, {"name": "b"}]
    graph = generator.generate_graph(variables)

    assert isinstance(graph, GeneratedGraph)
    assert len(graph.edges) == 1
    assert graph.edges[0].source == "a"
    assert graph.edges[0].target == "b"


# Test generate_graph increments call_count.
def test_generate_graph_increments_call_count(mocker) -> None:
    mock_groq = mocker.patch("causaliq_knowledge.graph.generator.GroqClient")
    mock_client = mocker.MagicMock()
    mock_client.completion.return_value = LLMResponse(
        content='{"edges": []}',
        model="test-model",
    )
    mock_client.use_cache = False
    mock_groq.return_value = mock_client

    generator = GraphGenerator(model="groq/test-model")
    variables = [{"name": "a"}, {"name": "b"}]

    generator.generate_graph(variables)
    assert generator.call_count == 1

    generator.generate_graph(variables)
    assert generator.call_count == 2


# Test generate_graph with custom prompt_detail level.
def test_generate_graph_with_prompt_detail(mocker) -> None:
    mock_groq = mocker.patch("causaliq_knowledge.graph.generator.GroqClient")
    mock_client = mocker.MagicMock()
    mock_client.completion.return_value = LLMResponse(
        content='{"edges": []}',
        model="test-model",
    )
    mock_client.use_cache = False
    mock_groq.return_value = mock_client

    generator = GraphGenerator(model="groq/test-model")
    variables = [{"name": "a", "type": "binary", "short_description": "desc"}]
    graph = generator.generate_graph(variables, level=PromptDetail.RICH)

    assert isinstance(graph, GeneratedGraph)


# Test generate_graph with domain context.
def test_generate_graph_with_domain(mocker) -> None:
    mock_groq = mocker.patch("causaliq_knowledge.graph.generator.GroqClient")
    mock_client = mocker.MagicMock()
    mock_client.completion.return_value = LLMResponse(
        content='{"edges": []}',
        model="test-model",
    )
    mock_client.use_cache = False
    mock_groq.return_value = mock_client

    generator = GraphGenerator(model="groq/test-model")
    variables = [{"name": "a"}, {"name": "b"}]
    graph = generator.generate_graph(variables, domain="epidemiology")

    assert isinstance(graph, GeneratedGraph)
    # Verify completion was called (domain is passed in prompt)
    mock_client.completion.assert_called_once()


# Test generate_graph adds metadata.
def test_generate_graph_adds_metadata(mocker) -> None:
    mock_groq = mocker.patch("causaliq_knowledge.graph.generator.GroqClient")
    mock_client = mocker.MagicMock()
    mock_client.completion.return_value = LLMResponse(
        content='{"edges": []}',
        model="test-model",
        input_tokens=100,
        output_tokens=50,
        cost=0.001,
    )
    mock_client.use_cache = False
    mock_groq.return_value = mock_client

    generator = GraphGenerator(model="groq/test-model")
    variables = [{"name": "a"}, {"name": "b"}]
    graph = generator.generate_graph(variables)

    assert graph.metadata is not None
    assert graph.metadata.model == "test-model"
    assert graph.metadata.provider == "groq"
    assert graph.metadata.input_tokens == 100
    assert graph.metadata.output_tokens == 50


# Test generate_graph with adjacency matrix format.
def test_generate_graph_adjacency_format(mocker) -> None:
    mock_groq = mocker.patch("causaliq_knowledge.graph.generator.GroqClient")
    mock_client = mocker.MagicMock()
    mock_client.completion.return_value = LLMResponse(
        content="""{
            "variables": ["a", "b"],
            "adjacency_matrix": [[0.0, 0.8], [0.0, 0.0]]
        }""",
        model="test-model",
    )
    mock_client.use_cache = False
    mock_groq.return_value = mock_client

    config = GraphGeneratorConfig(output_format=OutputFormat.ADJACENCY_MATRIX)
    generator = GraphGenerator(model="groq/test-model", config=config)
    variables = [{"name": "a"}, {"name": "b"}]
    graph = generator.generate_graph(variables)

    assert len(graph.edges) == 1
    assert graph.edges[0].source == "a"
    assert graph.edges[0].target == "b"


# Test generate_graph raises on invalid JSON response.
def test_generate_graph_raises_on_invalid_json(mocker) -> None:
    mock_groq = mocker.patch("causaliq_knowledge.graph.generator.GroqClient")
    mock_client = mocker.MagicMock()
    mock_client.completion.return_value = LLMResponse(
        content="not valid json",
        model="test-model",
    )
    mock_client.use_cache = False
    mock_groq.return_value = mock_client

    generator = GraphGenerator(model="groq/test-model")
    variables = [{"name": "a"}, {"name": "b"}]

    with pytest.raises(ValueError, match="Failed to parse JSON"):
        generator.generate_graph(variables)


# --- GraphGenerator get_stats tests ---


# Test get_stats returns expected structure.
def test_get_stats_returns_structure(mocker) -> None:
    mock_groq = mocker.patch("causaliq_knowledge.graph.generator.GroqClient")
    mock_client = mocker.MagicMock()
    mock_client.call_count = 5
    mock_groq.return_value = mock_client

    generator = GraphGenerator(model="groq/test-model")
    stats = generator.get_stats()

    assert "model" in stats
    assert "call_count" in stats
    assert "client_call_count" in stats
    assert stats["model"] == "groq/test-model"
    assert stats["client_call_count"] == 5


# --- GraphGenerator set_cache tests ---


# Test set_cache configures client cache.
def test_set_cache_configures_client(mocker) -> None:
    mock_groq = mocker.patch("causaliq_knowledge.graph.generator.GroqClient")
    mock_client = mocker.MagicMock()
    mock_groq.return_value = mock_client

    generator = GraphGenerator(model="groq/test-model")
    mock_cache = mocker.MagicMock()
    generator.set_cache(mock_cache, use_cache=True)

    mock_client.set_cache.assert_called_with(mock_cache, True)


# Test cache is passed to client on init.
def test_cache_passed_on_init(mocker) -> None:
    mock_groq = mocker.patch("causaliq_knowledge.graph.generator.GroqClient")
    mock_client = mocker.MagicMock()
    mock_groq.return_value = mock_client
    mock_cache = mocker.MagicMock()

    GraphGenerator(model="groq/test-model", cache=mock_cache)

    mock_client.set_cache.assert_called_with(mock_cache, use_cache=True)


# --- GraphGenerator _build_cache_key tests ---


# Test cache key includes graph prefix.
def test_cache_key_has_graph_prefix(mocker) -> None:
    mock_groq = mocker.patch("causaliq_knowledge.graph.generator.GroqClient")
    mock_groq.return_value = mocker.MagicMock()
    generator = GraphGenerator(model="groq/test-model")

    from causaliq_knowledge.graph.prompts import GraphQueryPrompt

    prompt = GraphQueryPrompt(
        variables=[{"name": "a"}],
        level=PromptDetail.MINIMAL,
    )
    system, user = prompt.build()
    key = generator._build_cache_key(prompt, system, user)

    assert key.startswith("graph_")


# Test cache key is deterministic.
def test_cache_key_is_deterministic(mocker) -> None:
    mock_groq = mocker.patch("causaliq_knowledge.graph.generator.GroqClient")
    mock_groq.return_value = mocker.MagicMock()
    generator = GraphGenerator(model="groq/test-model")

    from causaliq_knowledge.graph.prompts import GraphQueryPrompt

    prompt = GraphQueryPrompt(
        variables=[{"name": "a"}],
        level=PromptDetail.MINIMAL,
    )
    system, user = prompt.build()

    key1 = generator._build_cache_key(prompt, system, user)
    key2 = generator._build_cache_key(prompt, system, user)

    assert key1 == key2


# Test different prompts produce different keys.
def test_cache_key_differs_for_different_prompts(mocker) -> None:
    mock_groq = mocker.patch("causaliq_knowledge.graph.generator.GroqClient")
    mock_groq.return_value = mocker.MagicMock()
    generator = GraphGenerator(model="groq/test-model")

    from causaliq_knowledge.graph.prompts import GraphQueryPrompt

    prompt1 = GraphQueryPrompt(
        variables=[{"name": "a"}],
        level=PromptDetail.MINIMAL,
    )
    prompt2 = GraphQueryPrompt(
        variables=[{"name": "b"}],
        level=PromptDetail.MINIMAL,
    )

    system1, user1 = prompt1.build()
    system2, user2 = prompt2.build()

    key1 = generator._build_cache_key(prompt1, system1, user1)
    key2 = generator._build_cache_key(prompt2, system2, user2)

    assert key1 != key2


# --- GraphGenerator generate_from_context tests ---


# Test generate_from_context returns GeneratedGraph.
def test_generate_from_context_returns_graph(mocker) -> None:
    from causaliq_knowledge.graph.models import (
        NetworkContext,
        VariableSpec,
        VariableType,
    )

    mock_groq = mocker.patch("causaliq_knowledge.graph.generator.GroqClient")
    mock_client = mocker.MagicMock()
    mock_client.completion.return_value = LLMResponse(
        content='{"edges": []}',
        model="test-model",
    )
    mock_client.use_cache = False
    mock_groq.return_value = mock_client

    spec = NetworkContext(
        network="test-dataset",
        domain="test",
        variables=[
            VariableSpec(
                name="a",
                type=VariableType.BINARY,
                short_description="Variable A",
            ),
            VariableSpec(
                name="b",
                type=VariableType.BINARY,
                short_description="Variable B",
            ),
        ],
    )

    generator = GraphGenerator(model="groq/test-model")
    graph = generator.generate_from_context(spec)

    assert isinstance(graph, GeneratedGraph)
    assert generator.call_count == 1


# Test generate_from_context with custom prompt_detail level.
def test_generate_from_context_with_prompt_detail(mocker) -> None:
    from causaliq_knowledge.graph.models import (
        NetworkContext,
        VariableSpec,
        VariableType,
    )

    mock_groq = mocker.patch("causaliq_knowledge.graph.generator.GroqClient")
    mock_client = mocker.MagicMock()
    mock_client.completion.return_value = LLMResponse(
        content='{"edges": []}',
        model="test-model",
    )
    mock_client.use_cache = False
    mock_groq.return_value = mock_client

    spec = NetworkContext(
        network="epi-dataset",
        domain="epidemiology",
        variables=[
            VariableSpec(
                name="smoking",
                type=VariableType.BINARY,
                short_description="Smoking status",
            ),
            VariableSpec(
                name="cancer",
                type=VariableType.BINARY,
                short_description="Cancer diagnosis",
            ),
        ],
    )

    generator = GraphGenerator(model="groq/test-model")
    graph = generator.generate_from_context(spec, level=PromptDetail.RICH)

    assert isinstance(graph, GeneratedGraph)


# Test generate_from_context with custom output format.
def test_generate_from_context_with_output_format(mocker) -> None:
    from causaliq_knowledge.graph.models import (
        NetworkContext,
        VariableSpec,
        VariableType,
    )

    mock_groq = mocker.patch("causaliq_knowledge.graph.generator.GroqClient")
    mock_client = mocker.MagicMock()
    mock_client.completion.return_value = LLMResponse(
        content="""{
            "variables": ["a", "b"],
            "adjacency_matrix": [[0.0, 0.5], [0.0, 0.0]]
        }""",
        model="test-model",
    )
    mock_client.use_cache = False
    mock_groq.return_value = mock_client

    spec = NetworkContext(
        network="test-adj",
        domain="test",
        variables=[
            VariableSpec(
                name="a",
                type=VariableType.BINARY,
                short_description="A",
            ),
            VariableSpec(
                name="b",
                type=VariableType.BINARY,
                short_description="B",
            ),
        ],
    )

    generator = GraphGenerator(model="groq/test-model")
    graph = generator.generate_from_context(
        spec, output_format=OutputFormat.ADJACENCY_MATRIX
    )

    assert isinstance(graph, GeneratedGraph)
    assert len(graph.edges) == 1


# --- GraphGenerator caching tests ---


# Test generate_graph uses cached_completion when cache is set.
def test_generate_graph_uses_cached_completion(mocker) -> None:
    mock_groq = mocker.patch("causaliq_knowledge.graph.generator.GroqClient")
    mock_client = mocker.MagicMock()
    mock_client.cached_completion.return_value = LLMResponse(
        content='{"edges": []}',
        model="test-model",
    )
    mock_client.use_cache = True
    mock_groq.return_value = mock_client

    mock_cache = mocker.MagicMock()

    generator = GraphGenerator(model="groq/test-model", cache=mock_cache)
    variables = [{"name": "a"}, {"name": "b"}]
    graph = generator.generate_graph(variables)

    assert isinstance(graph, GeneratedGraph)
    mock_client.cached_completion.assert_called_once()
    mock_client.completion.assert_not_called()


# Test generate_graph detects cache hit from low latency.
def test_generate_graph_detects_cache_hit(mocker) -> None:
    mock_groq = mocker.patch("causaliq_knowledge.graph.generator.GroqClient")
    mock_time = mocker.patch("causaliq_knowledge.graph.generator.time")
    mock_client = mocker.MagicMock()
    mock_client.cached_completion.return_value = LLMResponse(
        content='{"edges": []}',
        model="test-model",
    )
    mock_client.use_cache = True
    mock_groq.return_value = mock_client

    # Simulate very fast response (cache hit) - 10ms
    mock_time.perf_counter.side_effect = [0.0, 0.01, 0.01]

    mock_cache = mocker.MagicMock()
    generator = GraphGenerator(model="groq/test-model", cache=mock_cache)
    variables = [{"name": "a"}, {"name": "b"}]
    graph = generator.generate_graph(variables)

    assert graph.metadata is not None
    assert graph.metadata.from_cache is True


# --- GraphGenerator generate_pdg_from_context tests ---


# Test generate_pdg_from_context returns PDGGenerationResult.
def test_generate_pdg_from_context_returns_pdg(mocker) -> None:
    from causaliq_core.graph.pdg import PDG

    from causaliq_knowledge.graph.models import (
        NetworkContext,
        VariableSpec,
        VariableType,
    )

    mock_groq = mocker.patch("causaliq_knowledge.graph.generator.GroqClient")
    mock_client = mocker.MagicMock()
    mock_client.completion.return_value = LLMResponse(
        content='{"edges": [{"source": "a", "target": "b", '
        '"existence": 0.8, "orientation": 0.7}]}',
        model="test-model",
    )
    mock_client.use_cache = False
    mock_groq.return_value = mock_client

    spec = NetworkContext(
        network="test-dataset",
        domain="test",
        variables=[
            VariableSpec(
                name="a",
                type=VariableType.BINARY,
                short_description="Variable A",
            ),
            VariableSpec(
                name="b",
                type=VariableType.BINARY,
                short_description="Variable B",
            ),
        ],
    )

    generator = GraphGenerator(model="groq/test-model")
    result = generator.generate_pdg_from_context(spec)

    assert isinstance(result.pdg, PDG)
    assert len(result.pdg.nodes) == 2
    assert len(result.pdg.edges) == 1
    assert result.metadata is not None
    assert generator.call_count == 1


# Test generate_pdg_from_context with custom prompt_detail level.
def test_generate_pdg_from_context_with_level(mocker) -> None:
    from causaliq_core.graph.pdg import PDG

    from causaliq_knowledge.graph.models import (
        NetworkContext,
        VariableSpec,
        VariableType,
    )

    mock_groq = mocker.patch("causaliq_knowledge.graph.generator.GroqClient")
    mock_client = mocker.MagicMock()
    mock_client.completion.return_value = LLMResponse(
        content='{"edges": []}',
        model="test-model",
    )
    mock_client.use_cache = False
    mock_groq.return_value = mock_client

    spec = NetworkContext(
        network="test-rich",
        domain="epidemiology",
        variables=[
            VariableSpec(
                name="smoking",
                type=VariableType.BINARY,
                short_description="Smoking status",
            ),
            VariableSpec(
                name="cancer",
                type=VariableType.BINARY,
                short_description="Cancer diagnosis",
            ),
        ],
    )

    generator = GraphGenerator(model="groq/test-model")
    result = generator.generate_pdg_from_context(spec, level=PromptDetail.RICH)

    assert isinstance(result.pdg, PDG)
    assert result.metadata is not None


# Test generate_pdg_from_context uses cached_completion when cache enabled.
def test_generate_pdg_from_context_uses_cache(mocker) -> None:
    from causaliq_core.graph.pdg import PDG

    from causaliq_knowledge.graph.models import (
        NetworkContext,
        VariableSpec,
        VariableType,
    )

    mock_groq = mocker.patch("causaliq_knowledge.graph.generator.GroqClient")
    mock_client = mocker.MagicMock()
    mock_client.cached_completion.return_value = LLMResponse(
        content='{"edges": [{"source": "x", "target": "y", '
        '"existence": 0.9, "orientation": 0.5}]}',
        model="test-model",
    )
    mock_client.use_cache = True
    mock_groq.return_value = mock_client

    mock_cache = mocker.MagicMock()

    spec = NetworkContext(
        network="cached-test",
        domain="test",
        variables=[
            VariableSpec(
                name="x",
                type=VariableType.CONTINUOUS,
                short_description="X variable",
            ),
            VariableSpec(
                name="y",
                type=VariableType.CONTINUOUS,
                short_description="Y variable",
            ),
        ],
    )

    generator = GraphGenerator(model="groq/test-model", cache=mock_cache)
    result = generator.generate_pdg_from_context(spec)

    assert isinstance(result.pdg, PDG)
    assert result.metadata is not None
    mock_client.cached_completion.assert_called_once()
    mock_client.completion.assert_not_called()


# Test generate_pdg_from_context with use_llm_names parameter.
def test_generate_pdg_from_context_with_llm_names(mocker) -> None:
    from causaliq_core.graph.pdg import PDG

    from causaliq_knowledge.graph.models import (
        NetworkContext,
        VariableSpec,
        VariableType,
    )

    mock_groq = mocker.patch("causaliq_knowledge.graph.generator.GroqClient")
    mock_client = mocker.MagicMock()
    mock_client.completion.return_value = LLMResponse(
        content='{"edges": []}',
        model="test-model",
    )
    mock_client.use_cache = False
    mock_groq.return_value = mock_client

    spec = NetworkContext(
        network="llm-names-test",
        domain="test",
        variables=[
            VariableSpec(
                name="var_a",
                llm_name="Variable A",
                type=VariableType.BINARY,
                short_description="First variable",
            ),
            VariableSpec(
                name="var_b",
                llm_name="Variable B",
                type=VariableType.BINARY,
                short_description="Second variable",
            ),
        ],
    )

    generator = GraphGenerator(model="groq/test-model")
    result = generator.generate_pdg_from_context(spec, use_llm_names=True)

    assert isinstance(result.pdg, PDG)
    assert result.metadata is not None
