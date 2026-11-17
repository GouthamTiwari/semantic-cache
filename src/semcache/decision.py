"""Accept or reject one semantic candidate (SPEC section 3, steps 5-6).

The engine and the pair evaluation call the *same* function, so the eval measures the code
that actually runs in production, not a re-implementation of it.
"""

from __future__ import annotations

from dataclasses import dataclass

from semcache.config import CacheConfig
from semcache.guards import run_guards
from semcache.types import Decision
from semcache.verify import Verifier


@dataclass
class CandidateVerdict:
    accept: bool
    decision: Decision | None  # SEMANTIC_HIT or VERIFIED_HIT when accepted
    reason: str  # "hit", "verified", "below_t_low", "guard:<name>", "verifier_no", "grey_zone"
    guard: str | None = None
    judged: bool = False
    judge_cost_usd: float = 0.0
    judge_ms: float = 0.0


def evaluate_candidate(
    question: str,
    cached_question: str,
    score: float,
    config: CacheConfig,
    verifier: Verifier | None = None,
) -> CandidateVerdict:
    if score < config.t_low:
        return CandidateVerdict(False, None, "below_t_low")
    # Guards first: they are free, the verifier is not.
    guard = run_guards(question, cached_question, config.guards, config.entity_mode)
    if guard:
        return CandidateVerdict(False, None, f"guard:{guard}", guard=guard)
    if score >= config.t_high:
        return CandidateVerdict(True, Decision.SEMANTIC_HIT, "hit")
    if verifier is None or not config.verifier_enabled:
        return CandidateVerdict(False, None, "grey_zone")
    v = verifier.verify(question, cached_question)
    if v.same:
        return CandidateVerdict(
            True, Decision.VERIFIED_HIT, "verified", None, True, v.cost_usd, v.latency_ms
        )
    return CandidateVerdict(False, None, "verifier_no", None, True, v.cost_usd, v.latency_ms)
