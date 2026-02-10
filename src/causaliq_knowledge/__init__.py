"""
causaliq-knowledge: LLM and human knowledge for causal discovery.
"""

from causaliq_knowledge.action import ActionProvider
from causaliq_knowledge.base import KnowledgeProvider
from causaliq_knowledge.models import EdgeDirection, EdgeKnowledge

__version__ = "0.4.0"
__author__ = "CausalIQ"
__email__ = "info@causaliq.com"

# Package metadata
__title__ = "causaliq-knowledge"
__description__ = "LLM and human knowledge for causal discovery"

__url__ = "https://github.com/causaliq/causaliq-knowledge"
__license__ = "MIT"

# Version tuple for programmatic access (major, minor, patch)
VERSION = (0, 4, 0)

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "VERSION",
    # Core models
    "EdgeKnowledge",
    "EdgeDirection",
    # Abstract interface
    "KnowledgeProvider",
    # Workflow action provider (auto-discovered by causaliq-workflow)
    "ActionProvider",
    # Note: Import LLMKnowledge from causaliq_knowledge.llm
]
