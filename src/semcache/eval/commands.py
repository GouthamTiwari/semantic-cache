"""`semcache` sub-commands for building data and running the evaluation."""

from __future__ import annotations

import json
from pathlib import Path


def cmd_dataset(args) -> int:
    from semcache.eval import dataset as ds

    if args.action == "generate":
        from semcache.config import get_settings
        from semcache.eval.generate import generate_pairs
        from semcache.eval.runtime import budget, live_client

        s = get_settings()
        info = generate_pairs(
            live_client(s),
            s.bedrock_chat_model_id,
            budget(s),
            Path("data/pairs/generated_raw.jsonl"),
        )
        print(json.dumps(info, indent=2))
    pairs = ds.load_pairs()
    if args.action == "review-sample":
        n = ds.write_review_sample(pairs, "data/pairs/review_sample.csv")
        print(f"wrote {n} pairs to data/pairs/review_sample.csv")
        return 0
    if args.action == "apply-review":
        changes = ds.apply_review(pairs, "data/pairs/review_sample.csv")
        ds.save_pairs(pairs)
        print("\n".join(changes) or "no changes")
    problems = ds.validate(pairs)
    print(ds.dump_stats(pairs))
    if problems:
        print("\n".join(problems[:50]))
        print(f"{len(problems)} problems")
        return 1
    print("dataset valid")
    return 0


def cmd_intents(args) -> int:
    from semcache.config import get_settings
    from semcache.eval.generate import generate_intents
    from semcache.eval.runtime import budget, live_client

    s = get_settings()
    info = generate_intents(
        live_client(s), s.bedrock_chat_model_id, budget(s), Path("data/traffic/intents.jsonl")
    )
    print(json.dumps(info, indent=2))
    return 0


def all_eval_texts(include_pairs: bool = True, include_intents: bool = False) -> list[str]:
    from semcache.normalize import normalize

    texts: list[str] = []
    if include_pairs:
        from semcache.eval.dataset import load_pairs

        for p in load_pairs():
            texts += [p.a, p.b]
    if include_intents and Path("data/traffic/intents.jsonl").exists():
        for line in open("data/traffic/intents.jsonl", encoding="utf-8"):
            texts += json.loads(line)["phrasings"]
    # The cache embeds normalised text, so the memo holds exactly what production would embed.
    return sorted({normalize(t) for t in texts})


def cmd_embed_memo(args) -> int:
    from semcache.eval.runtime import MemoMissError, memo_embedder

    texts = all_eval_texts(not args.intents_only, args.intents or args.intents_only)
    for dim in args.dims:
        emb = memo_embedder(dim, live=args.live)
        missing = [t for t in texts if emb.get(t) is None]
        print(f"dim {dim}: {len(texts)} texts, {len(missing)} not yet embedded")
        if missing and not args.live:
            print("run again with --live to embed them (costs a fraction of a cent)")
            return 1
        for i, t in enumerate(missing, 1):
            try:
                emb.embed(t)
            except MemoMissError as exc:
                print(exc)
                return 1
            if i % 100 == 0:
                print(f"  dim {dim}: {i}/{len(missing)}", flush=True)
        emb.close()
    return 0


def register(sub) -> None:
    p = sub.add_parser("dataset", help="build, check and review the labelled pair dataset")
    p.add_argument("action", choices=["generate", "validate", "review-sample", "apply-review"])
    p.set_defaults(func=cmd_dataset)

    p = sub.add_parser("intents", help="generate intent families for the traffic simulation")
    p.add_argument("action", choices=["generate"])
    p.set_defaults(func=cmd_intents)

    p = sub.add_parser("embed-memo", help="embed every eval text once into the on-disk memo")
    p.add_argument("--dims", type=int, nargs="+", default=[256, 1024])
    p.add_argument("--live", action="store_true", help="allowed to call Bedrock")
    p.add_argument("--intents", action="store_true", help="also embed traffic intents")
    p.add_argument("--intents-only", action="store_true")
    p.set_defaults(func=cmd_embed_memo)
