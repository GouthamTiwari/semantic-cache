"""Shared wiring for eval commands: live clients, the budget, and memo-backed embedders."""

from __future__ import annotations

from pathlib import Path

from semcache.budget import LiveBudget
from semcache.config import Settings, get_settings
from semcache.embed import EmbedResult, MemoEmbedder, TitanEmbedder
from semcache.verify import LLMVerifier


class MemoMissError(RuntimeError):
    pass


class _OfflineEmbedder:
    """Inner embedder for offline runs: a memo miss is an error, never a silent AWS call."""

    def __init__(self, model_id: str, dimensions: int):
        self.model_id = model_id
        self.dimensions = dimensions

    def embed(self, text: str) -> EmbedResult:
        raise MemoMissError(
            f"{text!r} is not in the embedding memo. Run `semcache embed-memo --live` first."
        )


def cache_dir(settings: Settings | None = None) -> Path:
    return Path((settings or get_settings()).cache_dir)


def budget(settings: Settings | None = None) -> LiveBudget:
    s = settings or get_settings()
    return LiveBudget(cache_dir(s) / "live_spend.json", s.live_budget_usd)


def live_client(settings: Settings | None = None):
    from semcache.bedrock import get_runtime_client

    return get_runtime_client(settings or get_settings())


def memo_embedder(dim: int, *, live: bool, settings: Settings | None = None) -> MemoEmbedder:
    """Embedder backed by .embed_cache/embeddings.sqlite.

    Offline (default) it only reads the memo. Memo hits report the original call's cost and
    latency, so eval accounting matches production while the run itself is free.
    """
    s = settings or get_settings()
    if live:
        inner = TitanEmbedder(
            live_client(s),
            model_id=s.bedrock_embed_model_id,
            dimensions=dim,
            requests_per_second=s.embed_requests_per_second,
            budget=budget(s),
        )
    else:
        inner = _OfflineEmbedder(s.bedrock_embed_model_id, dim)
    return MemoEmbedder(inner, cache_dir(s) / "embeddings.sqlite", report_cost_on_hit=True)


def memo_verifier(*, live: bool, settings: Settings | None = None) -> LLMVerifier:
    s = settings or get_settings()
    client = live_client(s) if live else _NoClient()
    return LLMVerifier(
        client,
        s.bedrock_judge_model_id,
        memo_path=cache_dir(s) / "verdicts.sqlite",
        budget=budget(s) if live else None,
        report_cost_on_hit=True,
    )


class _NoClient:
    def converse(self, **kw):
        raise MemoMissError("verdict not in the memo. Re-run with --live to judge new pairs.")
