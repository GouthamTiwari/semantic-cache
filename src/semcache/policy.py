"""Should a request use the cache, and may an answer be stored? (SPEC section 6.1)

Rules P1-P3 run before lookup (`request_policy`); rules P4-P6 run before storing
(`store_policy`). Each rule returns a short machine-readable reason that ends up in logs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from semcache.config import NamespaceConfig
from semcache.types import ChatRequest, TTLClass

TIME_SENSITIVE_RE = re.compile(
    r"\b(today|now|tonight|tomorrow|yesterday|latest|right now|this week|live|at the moment|"
    # "current account" is a bank product, not a point in time
    r"currently|current(?!\s+accounts?\b))\b",
    re.IGNORECASE,
)

# PII-looking patterns (P5). Deliberately broad: a missed store costs a little money,
# a stored phone number in a shared cache is a privacy incident.
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?<!\d)(?:\+\d{1,3}[\s-]?)?(?:[6-9]\d{9}|\d{3}[\s-]\d{3}[\s-]\d{4})(?!\d)")
LONG_DIGITS_RE = re.compile(r"(?<!\d)\d(?:[\s-]?\d){8,}(?!\d)")

CLEAN_STOP_REASONS = {"end_turn", "stop_sequence"}


@dataclass(frozen=True)
class RequestPolicy:
    use_cache: bool
    reason: str | None = None  # why bypassed
    ttl_class: TTLClass = TTLClass.NORMAL


def request_policy(request: ChatRequest, ns: NamespaceConfig) -> RequestPolicy:
    if request.no_cache:
        return RequestPolicy(False, "no_cache_flag")  # P1
    if any(m.role == "tool" for m in request.messages):
        return RequestPolicy(False, "tool_messages")  # P2
    if TIME_SENSITIVE_RE.search(request.question):  # P3
        if ns.allow_volatile:
            return RequestPolicy(True, None, TTLClass.VOLATILE)
        return RequestPolicy(False, "time_sensitive")
    return RequestPolicy(True, None, ns.default_ttl)


def contains_pii(text: str) -> bool:
    return bool(EMAIL_RE.search(text) or PHONE_RE.search(text) or LONG_DIGITS_RE.search(text))


def store_policy(question: str, answer: str, stop_reason: str | None) -> str | None:
    """None if the answer may be stored, else the reason it may not (P4-P6)."""
    if not answer.strip():
        return "empty_answer"  # P4
    if stop_reason not in CLEAN_STOP_REASONS:
        # max_tokens = truncated, tool_use = not a final answer, guardrail/content filter =
        # refusal, None = the stream never finished cleanly (P6).
        return f"stop_reason:{stop_reason}"
    if contains_pii(question) or contains_pii(answer):
        return "pii"  # P5
    return None
