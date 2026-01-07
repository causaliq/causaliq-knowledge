"""
causaliq-knowledge: LLM and human knowledge for causal discovery.
"""

from causaliq_knowledge.base import KnowledgeProvider
from causaliq_knowledge.models import EdgeDirection, EdgeKnowledge

__version__ = "0.1.0"
__author__ = "CausalIQ"
__email__ = "info@causaliq.com"

# Package metadata
__title__ = "causaliq-knowledge"
__description__ = "LLM and human knowledge for causal discovery"

__url__ = "https://github.com/causaliq/causaliq-knowledge"
__license__ = "MIT"

# Version tuple for programmatic access
VERSION = tuple(map(int, __version__.split(".")))

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
    # Note: Import LLMKnowledge from causaliq_knowledge.llm
]
