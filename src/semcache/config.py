"""Settings (from .env) and cache tuning (from code, with defaults from the eval).

Two kinds of configuration on purpose:
- `Settings` holds environment-specific values (credentials, model IDs, database URL).
- `CacheConfig` holds the behaviour knobs (thresholds, guards, top-k). Its defaults are the
  operating point chosen by the dev-split sweep (reports/operating_point.json); a test
  checks that the two stay in sync.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from semcache.types import TTLClass

SCHEMA_VERSION = 1  # bump when the stored format or meaning changes: old entries become unreachable

NOVA_2_LITE = "global.amazon.nova-2-lite-v1:0"
TITAN_V2 = "amazon.titan-embed-text-v2:0"
TITAN_V2_DIMENSIONS = (256, 512, 1024)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    aws_region: str = "ap-south-1"
    aws_bearer_token_bedrock: SecretStr = SecretStr("")
    aws_access_key_id: SecretStr = SecretStr("")
    aws_secret_access_key: SecretStr = SecretStr("")

    bedrock_embed_model_id: str = TITAN_V2
    bedrock_chat_model_id: str = NOVA_2_LITE
    bedrock_judge_model_id: str = NOVA_2_LITE
    embed_dimensions: int = 1024
    # Titan allows 60 requests/minute on a new account; 1/s stays under it. 0 = no pacing.
    embed_requests_per_second: float = 1.0

    database_url: str = ""  # empty = in-memory store
    admin_token: SecretStr = SecretStr("")
    live_budget_usd: float = 5.0
    cache_dir: str = ".embed_cache"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


@dataclass(frozen=True)
class NamespaceConfig:
    """Per-tenant policy (SPEC section 6.3)."""

    default_ttl: TTLClass = TTLClass.NORMAL
    # Time-sensitive questions are bypassed unless the namespace opts into short-lived entries.
    allow_volatile: bool = False


@dataclass(frozen=True)
class CacheConfig:
    """Behaviour knobs. Threshold defaults come from the dev sweep (see module docstring)."""

    t_high: float = 0.94  # = reports/operating_point.json (v2), checked by a test
    t_low: float = 0.84
    top_k: int = 3
    guards: tuple[str, ...] = ("numbers", "dates", "negation", "entities", "relation")
    entity_mode: str = "swap"  # "swap" or "strict" (SPEC section 7)
    verifier_enabled: bool = True
    singleflight_timeout_s: float = 30.0
    stream_chunk_chars: int = 40
    namespaces: dict[str, NamespaceConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.t_low <= self.t_high <= 1.0:
            raise ValueError("need 0 <= t_low <= t_high <= 1")
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")
        if self.entity_mode not in ("swap", "strict"):
            raise ValueError("entity_mode must be 'swap' or 'strict'")

    def namespace(self, name: str) -> NamespaceConfig:
        return self.namespaces.get(name, NamespaceConfig())
