"""The cache scope: which requests are allowed to share an entry (SPEC section 4).

This is a security property, not an optimisation. Every field that could change the right
answer is part of the scope, so a different tenant, model, system prompt, tool set,
temperature band, embedding model or conversation history can never receive an old answer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from semcache.config import SCHEMA_VERSION
from semcache.normalize import sha256
from semcache.types import ChatRequest


def temperature_bucket(t: float) -> str:
    if t == 0:
        return "t0"
    if t <= 0.5:
        return "low"
    if t <= 1.0:
        return "mid"
    return "high"


def history_hash(request: ChatRequest) -> str:
    """Hash of every turn before the last user message ('' for single-turn requests)."""
    if not request.history:
        return ""
    turns = [[m.role, m.content] for m in request.history]
    return sha256(json.dumps(turns, ensure_ascii=False))


@dataclass(frozen=True)
class Scope:
    namespace: str
    model_id: str
    system_hash: str
    tools_hash: str
    temperature_bucket: str
    embed_model_id: str
    embed_dim: int
    history_hash: str
    schema_version: int = SCHEMA_VERSION

    @property
    def key(self) -> str:
        parts = [
            self.namespace,
            self.model_id,
            self.system_hash,
            self.tools_hash,
            self.temperature_bucket,
            self.embed_model_id,
            str(self.embed_dim),
            self.history_hash,
            str(self.schema_version),
        ]
        # The separator cannot appear inside a hash, and the namespace is the only free
        # text field, so "a|b" + "c" can never collide with "a" + "b|c" in a hashed field.
        return sha256("\x1f".join(parts))


def build_scope(
    request: ChatRequest, *, model_id: str, embed_model_id: str, embed_dim: int
) -> Scope:
    tools = json.dumps(request.tools, sort_keys=True) if request.tools else ""
    return Scope(
        namespace=request.namespace,
        model_id=model_id,
        system_hash=sha256(request.system),
        tools_hash=sha256(tools),
        temperature_bucket=temperature_bucket(request.temperature),
        embed_model_id=embed_model_id,
        embed_dim=embed_dim,
        history_hash=history_hash(request),
    )
