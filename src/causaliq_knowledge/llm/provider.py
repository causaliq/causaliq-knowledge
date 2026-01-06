"""LLM-based knowledge provider with multi-model consensus."""

from typing import Optional

from causaliq_knowledge.base import KnowledgeProvider
from causaliq_knowledge.llm.client import LLMClient, LLMConfig
from causaliq_knowledge.llm.prompts import EdgeQueryPrompt, parse_edge_response
from causaliq_knowledge.models import EdgeDirection, EdgeKnowledge


def weighted_vote(responses: list[EdgeKnowledge]) -> EdgeKnowledge:
    """Combine multiple responses using weighted voting.

    Strategy:
    - For existence: weighted vote by confidence
    - For direction: weighted majority among those agreeing on existence
    - Final confidence: average confidence of agreeing models
    - Reasoning: combine reasoning from all models

    Args:
        responses: List of EdgeKnowledge from different models.

    Returns:
        Combined EdgeKnowledge result.
    """
    if not responses:
        return EdgeKnowledge.uncertain(reasoning="No responses to combine")

    if len(responses) == 1:
        return responses[0]

    # Calculate weighted votes for existence
    exists_weight = 0.0
    not_exists_weight = 0.0
    uncertain_weight = 0.0

    for r in responses:
        if r.exists is True:
            exists_weight += r.confidence
        elif r.exists is False:
            not_exists_weight += r.confidence
        else:
            uncertain_weight += r.confidence

    # Determine existence by highest weighted vote
    max_weight = max(exists_weight, not_exists_weight, uncertain_weight)
    if max_weight == 0:
        return EdgeKnowledge.uncertain(
            reasoning="All models uncertain with zero confidence"
        )

    if exists_weight == max_weight:
        final_exists: Optional[bool] = True
        agreeing = [r for r in responses if r.exists is True]
    elif not_exists_weight == max_weight:
        final_exists = False
        agreeing = [r for r in responses if r.exists is False]
    else:
        final_exists = None
        agreeing = [r for r in responses if r.exists is None]

    # Calculate direction from agreeing responses (only if edge exists)
    final_direction: Optional[EdgeDirection] = None
    if final_exists is True and agreeing:
        direction_votes: dict[Optional[EdgeDirection], float] = {}
        for r in agreeing:
            direction_votes[r.direction] = (
                direction_votes.get(r.direction, 0.0) + r.confidence
            )
        if direction_votes:
            final_direction = max(
                direction_votes.keys(),
                key=lambda d: direction_votes.get(d, 0.0),
            )

    # Calculate average confidence of agreeing responses
    if agreeing:
        final_confidence = sum(r.confidence for r in agreeing) / len(agreeing)
    else:  # pragma: no cover - defensive: agreeing is never empty here
        final_confidence = 0.0

    # Combine reasoning from all models
    models_used = [r.model for r in responses if r.model]
    reasoning_parts = []
    for r in responses:
        model_name = r.model or "unknown"
        reasoning_parts.append(f"[{model_name}] {r.reasoning}")
    combined_reasoning = " | ".join(reasoning_parts)

    return EdgeKnowledge(
        exists=final_exists,
        direction=final_direction,
        confidence=final_confidence,
        reasoning=combined_reasoning,
        model=", ".join(models_used) if models_used else None,
    )


def highest_confidence(responses: list[EdgeKnowledge]) -> EdgeKnowledge:
    """Return the response with highest confidence.

    Args:
        responses: List of EdgeKnowledge from different models.

    Returns:
        EdgeKnowledge with highest confidence score.
    """
    if not responses:
        return EdgeKnowledge.uncertain(reasoning="No responses to combine")

    return max(responses, key=lambda r: r.confidence)


# Mapping of strategy names to functions
CONSENSUS_STRATEGIES = {
    "weighted_vote": weighted_vote,
    "highest_confidence": highest_confidence,
}


class LLMKnowledge(KnowledgeProvider):
    """LLM-based knowledge provider using LiteLLM.

    This provider queries one or more LLMs about causal relationships
    and combines their responses using a configurable consensus strategy.

    Attributes:
        models: List of LiteLLM model identifiers.
        consensus_strategy: Strategy for combining multi-model responses.
        clients: Dict mapping model names to LLMClient instances.

    Example:
        >>> provider = LLMKnowledge(models=["gpt-4o-mini"])
        >>> result = provider.query_edge("smoking", "lung_cancer")
        >>> print(f"Exists: {result.exists}, Confidence: {result.confidence}")

        # Multi-model consensus
        >>> provider = LLMKnowledge(
        ...     models=["gpt-4o-mini", "claude-3-haiku-20240307"],
        ...     consensus_strategy="weighted_vote"
        ... )
    """

    def __init__(
        self,
        models: Optional[list[str]] = None,
        consensus_strategy: str = "weighted_vote",
        temperature: float = 0.1,
        max_tokens: int = 500,
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        """Initialize LLM knowledge provider.

        Args:
            models: List of LiteLLM model identifiers. Defaults to
                ["gpt-4o-mini"].
            consensus_strategy: How to combine multi-model responses.
                Options: "weighted_vote", "highest_confidence".
            temperature: LLM temperature (lower = more deterministic).
            max_tokens: Maximum tokens in LLM response.
            timeout: Request timeout in seconds.
            max_retries: Number of retries on failure.

        Raises:
            ValueError: If consensus_strategy is not recognized.
        """
        if models is None:
            models = ["gpt-4o-mini"]

        if consensus_strategy not in CONSENSUS_STRATEGIES:
            raise ValueError(
                f"Unknown consensus strategy: {consensus_strategy}. "
                f"Options: {list(CONSENSUS_STRATEGIES.keys())}"
            )

        self._models = models
        self._consensus_strategy = consensus_strategy
        self._consensus_fn = CONSENSUS_STRATEGIES[consensus_strategy]

        # Create a client for each model
        self._clients: dict[str, LLMClient] = {}
        for model in models:
            config = LLMConfig(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                max_retries=max_retries,
            )
            self._clients[model] = LLMClient(config=config)

    @property
    def name(self) -> str:
        """Return provider name with model list."""
        return f"LLMKnowledge({', '.join(self._models)})"

    @property
    def models(self) -> list[str]:
        """Return list of model identifiers."""
        return self._models.copy()

    @property
    def consensus_strategy(self) -> str:
        """Return the consensus strategy name."""
        return self._consensus_strategy

    def query_edge(
        self,
        node_a: str,
        node_b: str,
        context: Optional[dict] = None,
    ) -> EdgeKnowledge:
        """Query LLMs about a potential causal edge.

        Args:
            node_a: Name of the first variable.
            node_b: Name of the second variable.
            context: Optional context dict with keys:
                - domain: str - Domain context (e.g., "medicine")
                - descriptions: dict[str, str] - Variable descriptions
                - system_prompt: str - Custom system prompt

        Returns:
            EdgeKnowledge with combined result from all models.
        """
        # Build prompts
        prompt = EdgeQueryPrompt.from_context(node_a, node_b, context)
        system_prompt, user_prompt = prompt.build()

        # Query each model
        responses: list[EdgeKnowledge] = []
        for model, client in self._clients.items():
            try:
                json_data, _ = client.complete_json(
                    system=system_prompt,
                    user=user_prompt,
                )
                knowledge = parse_edge_response(json_data, model=model)
                responses.append(knowledge)
            except Exception as e:
                # On error, add uncertain response
                responses.append(
                    EdgeKnowledge.uncertain(
                        reasoning=f"Error querying {model}: {str(e)}",
                        model=model,
                    )
                )

        # Combine responses using consensus strategy
        return self._consensus_fn(responses)

    def get_stats(self) -> dict:
        """Get combined statistics from all clients.

        Returns:
            Dict with total_calls, total_cost, and per-model stats.
        """
        total_calls = 0
        total_cost = 0.0
        per_model: dict[str, dict] = {}

        for model, client in self._clients.items():
            stats = client.get_stats()
            total_calls += stats["call_count"]
            total_cost += stats["total_cost"]
            per_model[model] = stats

        return {
            "total_calls": total_calls,
            "total_cost": total_cost,
            "per_model": per_model,
        }
