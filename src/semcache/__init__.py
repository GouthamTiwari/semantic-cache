"""semcache: a semantic cache for LLM calls."""

from semcache.cache import SemanticCache
from semcache.config import CacheConfig, NamespaceConfig, Settings
from semcache.types import ChatRequest, ChatResponse, Decision, TTLClass

__version__ = "0.1.0"

__all__ = [
    "CacheConfig",
    "ChatRequest",
    "ChatResponse",
    "Decision",
    "NamespaceConfig",
    "SemanticCache",
    "Settings",
    "TTLClass",
]
