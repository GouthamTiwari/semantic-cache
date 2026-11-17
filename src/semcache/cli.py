"""Command line: `uv run semcache <command>`. Run `semcache -h` for the list."""

from __future__ import annotations

import argparse
import logging
import sys


def _demo_cache():
    """A cache that needs no AWS: hashing embedder + mock upstream. For trying the API only."""
    from semcache.cache import SemanticCache
    from semcache.config import CacheConfig
    from semcache.embed import HashingEmbedder
    from semcache.store import InMemoryStore
    from semcache.upstream import MockUpstream

    return SemanticCache(
        store=InMemoryStore(),
        embedder=HashingEmbedder(256),
        upstream=MockUpstream(latency_ms=800.0),
        # The hashing embedder is not a semantic model, so its scores are lower.
        config=CacheConfig(t_high=0.80, t_low=0.80, verifier_enabled=False),
    )


def cmd_serve(args) -> int:
    import uvicorn

    from semcache.api import create_app
    from semcache.config import get_settings

    settings = get_settings()
    if args.mock:
        cache = _demo_cache()
    else:
        from semcache.cache import SemanticCache

        cache = SemanticCache.from_settings(settings)
    app = create_app(cache, admin_token=settings.admin_token.get_secret_value())
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def cmd_purge(args) -> int:
    from semcache.cache import utcnow
    from semcache.config import get_settings
    from semcache.store.pgvector import PgVectorStore

    settings = get_settings()
    if not settings.database_url:
        print("DATABASE_URL is empty: the in-memory store has nothing to purge.")
        return 1
    store = PgVectorStore(settings.database_url, settings.embed_dimensions)
    if args.expired:
        print(f"deleted {store.purge_expired(utcnow())} expired entries")
    else:
        selector = {"namespace": args.namespace, "scope_key": args.scope_key, "entry_id": args.id}
        print(f"deleted {store.invalidate(**selector)} entries")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="semcache", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("serve", help="run the HTTP service")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--mock", action="store_true", help="no AWS: hashing embedder + mock LLM")
    s.set_defaults(func=cmd_serve)

    s = sub.add_parser("purge", help="delete expired entries or invalidate a selector")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--expired", action="store_true")
    g.add_argument("--namespace")
    g.add_argument("--scope-key")
    g.add_argument("--id")
    s.set_defaults(func=cmd_purge)

    try:
        from semcache.eval.commands import register

        register(sub)
    except ImportError:  # eval tooling not installed yet
        pass
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
