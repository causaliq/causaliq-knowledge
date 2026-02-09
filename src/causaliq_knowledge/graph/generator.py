"""Graph generator using LLM providers.

This module provides the GraphGenerator class for generating complete
causal graphs from variable specifications using LLM providers.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from causaliq_knowledge.graph.prompts import GraphQueryPrompt, OutputFormat
from causaliq_knowledge.graph.response import (
    GeneratedGraph,
    GenerationMetadata,
    parse_graph_response,
)
from causaliq_knowledge.graph.view_filter import PromptDetail
from causaliq_knowledge.llm.anthropic_client import (
    AnthropicClient,
    AnthropicConfig,
)
from causaliq_knowledge.llm.deepseek_client import (
    DeepSeekClient,
    DeepSeekConfig,
)
from causaliq_knowledge.llm.gemini_client import GeminiClient, GeminiConfig
from causaliq_knowledge.llm.groq_client import GroqClient, GroqConfig
from causaliq_knowledge.llm.mistral_client import MistralClient, MistralConfig
from causaliq_knowledge.llm.ollama_client import OllamaClient, OllamaConfig
from causaliq_knowledge.llm.openai_client import OpenAIClient, OpenAIConfig

if TYPE_CHECKING:  # pragma: no cover
    from causaliq_core.cache import TokenCache

    from causaliq_knowledge.graph.models import ModelSpec

logger = logging.getLogger(__name__)


# Type alias for supported clients
LLMClientType = Union[
    AnthropicClient,
    DeepSeekClient,
    GeminiClient,
    GroqClient,
    MistralClient,
    OllamaClient,
    OpenAIClient,
]


@dataclass
class GraphGeneratorConfig:
    """Configuration for the GraphGenerator.

    Attributes:
        temperature: LLM sampling temperature (lower = more deterministic).
        max_tokens: Maximum tokens in LLM response.
        timeout: Request timeout in seconds.
        output_format: Desired output format (edge_list or adjacency_matrix).
        prompt_detail: Detail level for variable information in prompts.
        use_llm_names: Use llm_name instead of benchmark name in prompts.
        request_id: Optional identifier for requests (stored in metadata).
    """

    temperature: float = 0.1
    max_tokens: int = 2000
    timeout: float = 60.0
    output_format: OutputFormat = OutputFormat.EDGE_LIST
    prompt_detail: PromptDetail = PromptDetail.STANDARD
    use_llm_names: bool = True
    request_id: str = ""


class GraphGenerator:
    """Generate causal graphs from variable specifications using LLMs.

    This class provides methods for generating complete causal graphs
    from ModelSpec objects or variable dictionaries. It supports all
    LLM providers available in causaliq-knowledge and integrates with
    the TokenCache for efficient caching of requests.

    Attributes:
        model: The LLM model identifier (e.g., "groq/llama-3.1-8b-instant").
        config: Configuration for generation parameters.

    Example:
        >>> from causaliq_knowledge.graph import ModelLoader
        >>> from causaliq_knowledge.graph.generator import GraphGenerator
        >>>
        >>> # Load model specification
        >>> spec = ModelLoader.load("model.json")
        >>>
        >>> # Create generator
        >>> generator = GraphGenerator(model="groq/llama-3.1-8b-instant")
        >>>
        >>> # Generate graph
        >>> graph = generator.generate_from_spec(spec)
        >>> print(f"Generated {len(graph.edges)} edges")
    """

    def __init__(
        self,
        model: str = "groq/llama-3.1-8b-instant",
        config: Optional[GraphGeneratorConfig] = None,
        cache: Optional["TokenCache"] = None,
    ) -> None:
        """Initialise the GraphGenerator.

        Args:
            model: LLM model identifier with provider prefix. Supported:
                - "groq/llama-3.1-8b-instant" (Groq API)
                - "gemini/gemini-2.5-flash" (Google Gemini)
                - "openai/gpt-4o" (OpenAI)
                - "anthropic/claude-3-5-sonnet-20241022" (Anthropic)
                - "deepseek/deepseek-chat" (DeepSeek)
                - "mistral/mistral-small-latest" (Mistral)
                - "ollama/llama3.2:1b" (Local Ollama)
            config: Generation configuration. Uses defaults if None.
            cache: TokenCache instance for caching. Disabled if None.

        Raises:
            ValueError: If the model prefix is not supported.
        """
        self._model = model
        self._config = config or GraphGeneratorConfig()
        self._cache = cache
        self._client = self._create_client(model)
        self._call_count = 0

        # Configure cache on client if provided
        if cache is not None:
            self._client.set_cache(cache, use_cache=True)

    def _create_client(self, model: str) -> LLMClientType:
        """Create the appropriate LLM client for the model.

        Args:
            model: Model identifier with provider prefix.

        Returns:
            Configured LLM client instance.

        Raises:
            ValueError: If the model prefix is not supported.
        """
        config = self._config

        if model.startswith("anthropic/"):
            model_name = model.split("/", 1)[1]
            return AnthropicClient(
                config=AnthropicConfig(
                    model=model_name,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    timeout=config.timeout,
                )
            )
        elif model.startswith("deepseek/"):
            model_name = model.split("/", 1)[1]
            return DeepSeekClient(
                config=DeepSeekConfig(
                    model=model_name,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    timeout=config.timeout,
                )
            )
        elif model.startswith("gemini/"):
            model_name = model.split("/", 1)[1]
            return GeminiClient(
                config=GeminiConfig(
                    model=model_name,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    timeout=config.timeout,
                )
            )
        elif model.startswith("groq/"):
            model_name = model.split("/", 1)[1]
            return GroqClient(
                config=GroqConfig(
                    model=model_name,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    timeout=config.timeout,
                )
            )
        elif model.startswith("mistral/"):
            model_name = model.split("/", 1)[1]
            return MistralClient(
                config=MistralConfig(
                    model=model_name,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    timeout=config.timeout,
                )
            )
        elif model.startswith("ollama/"):
            model_name = model.split("/", 1)[1]
            return OllamaClient(
                config=OllamaConfig(
                    model=model_name,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    timeout=config.timeout,
                )
            )
        elif model.startswith("openai/"):
            model_name = model.split("/", 1)[1]
            return OpenAIClient(
                config=OpenAIConfig(
                    model=model_name,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    timeout=config.timeout,
                )
            )
        else:
            supported = [
                "anthropic/",
                "deepseek/",
                "gemini/",
                "groq/",
                "mistral/",
                "ollama/",
                "openai/",
            ]
            raise ValueError(
                f"Model '{model}' not supported. "
                f"Supported prefixes: {supported}."
            )

    @property
    def model(self) -> str:
        """Return the model identifier."""
        return self._model

    @property
    def config(self) -> GraphGeneratorConfig:
        """Return the generator configuration."""
        return self._config

    @property
    def call_count(self) -> int:
        """Return the number of generation calls made."""
        return self._call_count

    def set_cache(
        self,
        cache: Optional["TokenCache"],
        use_cache: bool = True,
    ) -> None:
        """Configure caching for this generator.

        Args:
            cache: TokenCache instance for caching, or None to disable.
            use_cache: Whether to use the cache (default True).
        """
        self._cache = cache
        self._client.set_cache(cache, use_cache)

    def _build_cache_key(
        self,
        prompt: GraphQueryPrompt,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Build a deterministic cache key for graph generation.

        Uses a prefix to distinguish graph queries from edge queries.

        Args:
            prompt: The GraphQueryPrompt used.
            system_prompt: The system prompt string.
            user_prompt: The user prompt string.

        Returns:
            16-character hex string cache key with graph prefix.
        """
        key_data = {
            "type": "graph_generation",
            "model": self._model,
            "output_format": self._config.output_format.value,
            "prompt_detail": prompt.level.value,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "temperature": self._config.temperature,
        }
        key_json = json.dumps(key_data, sort_keys=True, separators=(",", ":"))
        return "graph_" + hashlib.sha256(key_json.encode()).hexdigest()[:12]

    def generate_graph(
        self,
        variables: List[Dict[str, Any]],
        level: Optional[PromptDetail] = None,
        domain: Optional[str] = None,
        output_format: Optional[OutputFormat] = None,
        system_prompt: Optional[str] = None,
    ) -> GeneratedGraph:
        """Generate a causal graph from variable dictionaries.

        Args:
            variables: List of variable dictionaries with at least "name".
            level: View level for context. Uses config default if None.
            domain: Optional domain context for the query.
            output_format: Output format. Uses config default if None.
            system_prompt: Custom system prompt (optional).

        Returns:
            GeneratedGraph with proposed edges and metadata.

        Raises:
            ValueError: If LLM response cannot be parsed.
        """
        level = level or self._config.prompt_detail
        output_format = output_format or self._config.output_format

        # Build the prompt
        prompt = GraphQueryPrompt(
            variables=variables,
            level=level,
            domain=domain,
            output_format=output_format,
            system_prompt=system_prompt,
        )

        return self._execute_query(prompt)

    def generate_from_spec(
        self,
        spec: "ModelSpec",
        level: Optional[PromptDetail] = None,
        output_format: Optional[OutputFormat] = None,
        system_prompt: Optional[str] = None,
        use_llm_names: Optional[bool] = None,
    ) -> GeneratedGraph:
        """Generate a causal graph from a ModelSpec.

        Convenience method that extracts variables and domain from the
        specification automatically.

        Args:
            spec: The model specification.
            level: View level for context. Uses config default if None.
            output_format: Output format. Uses config default if None.
            system_prompt: Custom system prompt (optional).
            use_llm_names: Use llm_name instead of benchmark name.
                Uses config default if None.

        Returns:
            GeneratedGraph with proposed edges and metadata.

        Raises:
            ValueError: If LLM response cannot be parsed.
        """
        level = level or self._config.prompt_detail
        output_format = output_format or self._config.output_format
        use_llm = (
            use_llm_names
            if use_llm_names is not None
            else self._config.use_llm_names
        )

        # Use the class method to create prompt from spec
        prompt = GraphQueryPrompt.from_model_spec(
            spec=spec,
            level=level,
            output_format=output_format,
            system_prompt=system_prompt,
            use_llm_names=use_llm,
        )

        return self._execute_query(prompt)

    def _execute_query(self, prompt: GraphQueryPrompt) -> GeneratedGraph:
        """Execute the LLM query and parse the response.

        Args:
            prompt: The configured GraphQueryPrompt.

        Returns:
            GeneratedGraph with parsed edges and metadata.

        Raises:
            ValueError: If LLM response cannot be parsed.
        """
        system_prompt, user_prompt = prompt.build()
        variable_names = prompt.get_variable_names()

        # Build messages for the LLM
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        # Record request timestamp before making the call
        request_timestamp = datetime.now(timezone.utc)

        # Make the request (using cached_completion if cache is set)
        start_time = time.perf_counter()
        from_cache = False

        if self._cache is not None and self._client.use_cache:
            response = self._client.cached_completion(
                messages, request_id=self._config.request_id
            )
            # Check if response was from cache by comparing timing
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            # If latency is very low, likely from cache
            from_cache = latency_ms < 50
        else:
            response = self._client.completion(messages)

        latency_ms = int((time.perf_counter() - start_time) * 1000)

        # Record completion timestamp after receiving response
        completion_timestamp = datetime.now(timezone.utc)

        self._call_count += 1

        # Parse the response
        output_format_str = prompt.output_format.value
        graph = parse_graph_response(
            response.content,
            variable_names,
            output_format_str,
        )

        # Add metadata
        provider = self._model.split("/")[0] if "/" in self._model else ""
        model_name = (
            self._model.split("/", 1)[1] if "/" in self._model else self._model
        )

        # Calculate initial cost (cost if request was not from cache)
        # If from_cache is True, initial_cost_usd represents what the request
        # would have cost if it had been made fresh
        initial_cost = response.cost if not from_cache else response.cost

        graph.metadata = GenerationMetadata(
            model=model_name,
            provider=provider,
            timestamp=completion_timestamp,
            latency_ms=latency_ms,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost_usd=response.cost if not from_cache else 0.0,
            from_cache=from_cache,
            messages=messages,
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens,
            finish_reason=response.finish_reason,
            request_timestamp=request_timestamp,
            completion_timestamp=completion_timestamp,
            initial_cost_usd=initial_cost,
        )

        return graph

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about generation calls.

        Returns:
            Dict with call_count, model, and client stats.
        """
        return {
            "model": self._model,
            "call_count": self._call_count,
            "client_call_count": self._client.call_count,
        }
