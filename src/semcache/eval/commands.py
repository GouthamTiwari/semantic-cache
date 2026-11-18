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


def cmd_sweep(args) -> int:
    from semcache.eval.pairs_eval import DIMS, run_dev, write_dev_reports
    from semcache.eval.runtime import memo_embedder, memo_verifier

    embedders = {d: memo_embedder(d, live=args.live) for d in DIMS}
    result = run_dev(embedders, memo_verifier(live=args.live))
    write_dev_reports(result)
    print(json.dumps(result["operating_point"], indent=2))
    print("wrote reports/pairs_dev.{json,md}, reports/operating_point.json, reports/sweep_*.svg")
    return 0


def cmd_test(args) -> int:
    from semcache.eval.pairs_eval import op_hash, run_test, write_test_report
    from semcache.eval.runtime import memo_embedder, memo_verifier

    op = json.loads(Path("reports/operating_point.json").read_text(encoding="utf-8"))
    previous = Path("reports/pairs_test.json")
    if previous.exists() and not args.new_operating_point:
        old = json.loads(previous.read_text(encoding="utf-8"))["operating_point_hash"]
        if old != op_hash(op):
            # The test split is reported once. A changed operating point after seeing test
            # results would be tuning on test, so it needs an explicit, visible flag.
            print("operating point changed since the last test run; refusing to re-test.")
            print("If this is a deliberate new experiment, pass --new-operating-point.")
            return 1
    res = run_test(
        memo_embedder(op["embed_dim"], live=args.live), memo_verifier(live=args.live), op
    )
    write_test_report(res)
    print(json.dumps(res["result"] | {"by_category": "see reports/pairs_test.md"}, indent=2))
    return 0


def cmd_calibrate(args) -> int:
    from semcache.config import get_settings
    from semcache.eval.calibrate import calibrate, summarise
    from semcache.eval.report import write_json
    from semcache.eval.runtime import budget, live_client
    from semcache.upstream import BedrockUpstream

    if not args.live:
        print("calibration makes real Bedrock calls: pass --live to confirm")
        return 1
    s = get_settings()
    cal = calibrate(BedrockUpstream(live_client(s), budget=budget(s)), s.bedrock_chat_model_id,
                    n=args.n)  # fmt: skip
    cal["summary"] = summarise(cal, dim=s.embed_dimensions)
    write_json("calibration.json", cal)
    print(json.dumps(cal["summary"], indent=2))
    return 0


def cmd_simulate(args) -> int:
    import logging

    # 5,000 simulated requests would print 15,000 JSON request lines.
    logging.getLogger("semcache.request").setLevel(logging.WARNING)
    from semcache.eval.calibrate import summarise
    from semcache.eval.runtime import memo_embedder, memo_verifier
    from semcache.eval.traffic import run_simulation, write_simulation_report

    op = json.loads(Path("reports/operating_point.json").read_text(encoding="utf-8"))
    cal = json.loads(Path("reports/calibration.json").read_text(encoding="utf-8"))
    res = run_simulation(
        memo_embedder(op["embed_dim"], live=args.live), op, cal["samples"],
        memo_verifier(live=args.live),
    )  # fmt: skip
    summary = summarise(cal, dim=op["embed_dim"])  # fresh: the memo may have grown
    write_simulation_report(res, summary)
    for r in res["results"]:
        print(f"{r['config']:<30} hit {r['hit_rate']:.3f}  false {r['false_hits']:>3}  "
              f"usd {r['usd_total']:.4f}  p50 {r['latency_ms_p50']} ms")  # fmt: skip
    return 0


def cmd_eval(args) -> int:
    """Regenerate every report from the memos (free and repeatable), then sync the README."""
    for step in (cmd_sweep, cmd_test, cmd_simulate):
        code = step(args)
        if code:
            return code
    return cmd_readme_sync(args)


def cmd_readme_sync(args) -> int:
    from semcache.eval.readme_sync import sync_readme

    print("README blocks updated from reports/: " + ", ".join(sync_readme()))
    return 0


def register(sub) -> None:
    p = sub.add_parser(
        "calibrate", help="measure real Nova 2 Lite latency and tokens (<= 50 calls)"
    )
    p.add_argument("--live", action="store_true")
    p.add_argument("-n", type=int, default=40)
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("simulate", help="replay a synthetic day of traffic against 5 configs")
    p.add_argument("--live", action="store_true", help="allowed to call the judge model")
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("readme-sync", help="copy result tables from reports/ into README.md")
    p.set_defaults(func=cmd_readme_sync)

    p = sub.add_parser("eval", help="regenerate all reports from the memos, then the README")
    p.add_argument("--live", action="store_true")
    p.add_argument("--new-operating-point", action="store_true")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("sweep", help="dev split: sweep, dimension, ablation, verifier")
    p.add_argument("--live", action="store_true", help="allowed to call Bedrock")
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("test", help="held-out test split at the frozen operating point")
    p.add_argument("--live", action="store_true", help="allowed to call Bedrock")
    p.add_argument("--new-operating-point", action="store_true")
    p.set_defaults(func=cmd_test)

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
