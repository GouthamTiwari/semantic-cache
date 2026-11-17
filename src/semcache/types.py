"""Shared data shapes: requests, responses, cache entries and decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class Decision(StrEnum):
    """What the cache did with one request (SPEC section 3.2)."""

    BYPASS = "bypass"  # policy said: do not use the cache at all
    EXACT_HIT = "exact_hit"  # same normalized text in the same scope
    SEMANTIC_HIT = "semantic_hit"  # similar enough, guards passed
    VERIFIED_HIT = "verified_hit"  # grey-zone score, the judge said "same"
    MISS = "miss"  # nothing close enough, upstream was called
    GUARD_REJECT = "guard_reject"  # close enough by score, but a guard said no; upstream called

    @property
    def is_hit(self) -> bool:
        return self in (Decision.EXACT_HIT, Decision.SEMANTIC_HIT, Decision.VERIFIED_HIT)


class TTLClass(StrEnum):
    STATIC = "static"
    NORMAL = "normal"
    VOLATILE = "volatile"

    @property
    def ttl(self) -> timedelta:
        return TTL_DURATIONS[self]


TTL_DURATIONS: dict[TTLClass, timedelta] = {
    TTLClass.STATIC: timedelta(days=7),
    TTLClass.NORMAL: timedelta(hours=24),
    TTLClass.VOLATILE: timedelta(minutes=15),
}


class Message(BaseModel):
    role: Literal["user", "assistant", "tool"]
    content: str


class ChatRequest(BaseModel):
    """One chat call the cache sits in front of (SPEC section 3.1)."""

    namespace: str = Field(default="default", min_length=1, max_length=128)
    model: str | None = None  # None = the configured chat model
    system: str = ""
    messages: list[Message] = Field(min_length=1)
    tools: list[dict[str, Any]] | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1, le=8192)
    stream: bool = False
    no_cache: bool = False

    @field_validator("messages")
    @classmethod
    def _last_message_is_user(cls, messages: list[Message]) -> list[Message]:
        # The cache answers the *last user question*; anything else has nothing to look up.
        if messages[-1].role != "user":
            raise ValueError("the last message must have role 'user'")
        if not messages[-1].content.strip():
            raise ValueError("the last user message is empty")
        return messages

    @property
    def question(self) -> str:
        return self.messages[-1].content

    @property
    def history(self) -> list[Message]:
        return self.messages[:-1]


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class ChatResponse(BaseModel):
    text: str
    decision: Decision
    score: float | None = None
    entry_id: str | None = None
    scope_key: str | None = None
    latency_ms: float = 0.0
    input_tokens: int = 0  # upstream tokens for a miss, the stored call's tokens for a hit
    output_tokens: int = 0
    cost_usd: float = 0.0  # money spent on this request (embedding + judge + upstream)
    saved_usd: float = 0.0  # for hits: what the stored answer's original call cost
    stored: bool = False
    skip_reason: str | None = None
    guard: str | None = None  # which guard rejected the best candidate, if any
    coalesced: bool = False  # answered by waiting for an identical in-flight request
    overhead_usd: float = 0.0  # the part of cost_usd spent on the cache itself (embed + judge)
    # Component latencies in ms: embed, search, judge, upstream (used by the simulation).
    timings: dict[str, float] = Field(default_factory=dict)


@dataclass
class CacheEntry:
    """One stored answer. Token counts are kept so a later hit knows what it saved."""

    id: str
    scope_key: str
    namespace: str
    model_id: str
    exact_key: str
    query_text: str
    query_norm: str
    embedding: list[float]
    answer_text: str
    stop_reason: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    ttl_class: TTLClass
    created_at: datetime
    expires_at: datetime
    hit_count: int = 0
    last_hit_at: datetime | None = None
    extra: dict[str, Any] = field(default_factory=dict)  # free-form, e.g. eval intent ids

    def is_expired(self, now: datetime) -> bool:
        return self.expires_at <= now
