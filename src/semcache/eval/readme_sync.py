"""Copy result tables from reports/*.json into README.md between marker comments.

Numbers in the README are never typed by hand: `semcache readme-sync` (also run at the end
of `semcache eval`) rebuilds every block between `<!-- results:NAME -->` and
`<!-- /results:NAME -->`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from semcache.eval.report import pct, table, usd

README = Path("README.md")


def _load(name: str) -> dict:
    return json.loads((Path("reports") / name).read_text(encoding="utf-8"))


def _answers(n: int) -> str:
    return f"{n} wrong answer{'' if n == 1 else 's'}"


def _headline(dev: dict, test: dict, sim: dict) -> str:
    op = test["operating_point"]
    m = test["result"]
    recommended = _recommended(sim, op)
    exact = next(r for r in sim["results"] if r["config"] == "exact only")
    guards_only = next(r for r in sim["results"] if r["config"] == "semantic + guards")
    no_guards = test["same_threshold_without_guards"]
    ci = test["false_hit_rate_ci95"]
    lines = [
        f"- **False-hit rate on the held-out test split: {pct(m['false_hit_rate'])}** "
        f"({m['fp']} of {m['fp'] + m['tn']} should-miss pairs; 95 % interval "
        f"{pct(ci[0])}-{pct(ci[1])}), at precision "
        f"{pct(m['precision'])} and recall {pct(m['recall'])}.",
        f"- **Same threshold without guards: {pct(no_guards['false_hit_rate'])} false hits** "
        "on the same test pairs.",
        f"- **Traffic simulation ({sim['stream']['requests']} requests):** exact match alone "
        f"saves {pct(exact['net_saving_pct'])} of LLM spend. With the chosen settings "
        f"({recommended['config']}) the cache saves {pct(recommended['net_saving_pct'])} after "
        f"paying for embeddings and judge calls, with {_answers(recommended['false_hits'])} "
        f"({pct(recommended['false_hits'] / sim['stream']['requests'])} of requests).",
    ]
    if recommended is not guards_only:
        lines.append(
            f"- **Without the verifier** (guards only): {pct(guards_only['net_saving_pct'])} "
            f"saved, {_answers(guards_only['false_hits'])}, hit p95 "
            f"{guards_only['hit_latency_ms_p95']} ms. The verifier buys extra hits but each "
            f"verified hit waits for a judge call (hit p95 {recommended['hit_latency_ms_p95']} ms)."
        )
    lines += [
        f"- **Latency (chosen settings):** hit p50 {recommended['hit_latency_ms_p50']} ms vs miss "
        f"p50 {recommended['miss_latency_ms_p50']} ms.",
        f"- Operating point chosen on the dev split only: Titan v2 at {op['embed_dim']} dims, "
        f"threshold {op['t_high']}, all {len(op['guards'])} guards, verifier "
        f"{'on (grey zone from ' + str(op['t_low']) + ')' if op['verifier_enabled'] else 'off'}.",
    ]
    return "\n".join(lines)


def _recommended(sim: dict, op: dict) -> dict:
    name = "semantic + guards + verifier" if op["verifier_enabled"] else "semantic + guards"
    return next(r for r in sim["results"] if r["config"] == name)


def _test_table(test: dict) -> str:
    m, ng = test["result"], test["same_threshold_without_guards"]
    head = table(
        ["config (test split, threshold " + str(test["operating_point"]["t_high"]) + ")",
         "precision", "recall", "false-hit rate", "false hits"],
        [["with guards (chosen)", pct(m["precision"]), pct(m["recall"]),
          pct(m["false_hit_rate"]), m["fp"]],
         ["without guards", pct(ng["precision"]), pct(ng["recall"]),
          pct(ng["false_hit_rate"]), ng["fp"]]],
    )  # fmt: skip
    cats = table(
        ["category", "should-miss pairs", "false hits", "should-hit pairs", "recall"],
        [[c, v["different"], v["false_hits"], v["same"], pct(v["recall"])]
         for c, v in m["by_category"].items()],
    )  # fmt: skip
    return f"{head}\n\nBy category:\n\n{cats}"


def _dims(dev: dict) -> str:
    rows = []
    for dim, d in dev["dims"].items():
        for name, key in (("threshold only", "operating_point_threshold_only"),
                          ("threshold + guards", "operating_point_with_guards")):  # fmt: skip
            o = d[key]
            rows.append([dim, name, o["threshold"] if o else "none", pct(o["recall"]) if o else "-",
                         pct(o["false_hit_rate"]) if o else "-"])  # fmt: skip
    dc = dev["dimension_choice"]
    return (
        table(
            ["dims", "config", "lowest threshold with dev FHR <= 1%", "dev recall", "dev FHR"], rows
        )  # fmt: skip
        + f"\n\nChosen: **{dc['chosen']} dims**. Rule fixed before the run: {dc['rule']}."
    )


def _ablation(dev: dict) -> str:
    ga = dev["guard_ablation"]
    return (
        f"Dev split, {ga['dim']} dims, threshold {ga['threshold']}:\n\n"
        + table(
            [
                "guards",
                "false hits",
                "removed vs none",
                "false-hit rate",
                "recall",
                "good hits wrongly blocked",
            ],
            [
                [
                    r["guards"],
                    r["false_hits"],
                    r["false_hits_removed"],
                    pct(r["false_hit_rate"]),
                    pct(r["recall"]),
                    r["same_pairs_wrongly_rejected"],
                ]
                for r in ga["rows"]
            ],
        )  # fmt: skip
    )


def _verifier(dev: dict) -> str:
    v = dev["verifier"]
    t = table(
        ["grey zone from", "verifier", "recall", "false-hit rate", "judge calls", "judge cost",
         "judge p50 ms"],
        [[r["t_low"], r["verifier"], pct(r["recall"]), pct(r["false_hit_rate"]),
          r["judge_calls"], usd(r["judge_cost_usd"]), r["judge_ms_p50"]] for r in v["rows"]],
    )  # fmt: skip
    verdict = "worth it" if v["worth_it"] else "**not worth it** on this data"
    return f"{t}\n\nRule fixed before the run: {v['rule']}. Result: {verdict}."


def _simulation(sim: dict) -> str:
    rows = [
        [r["config"], pct(r["hit_rate"]), r["exact_hits"], r["semantic_hits"], r["false_hits"],
         r["upstream_calls"], usd(r["usd_total"], 4), pct(r.get("net_saving_pct", 0.0)),
         r["latency_ms_p50"], r["latency_ms_p95"]]
        for r in sim["results"]
    ]  # fmt: skip
    c = sim["calibration"]
    s = sim["stream"]
    return (
        f"{s['requests']} requests, {s['intents']} intents in near-miss families, Zipf "
        f"popularity, {s['burst_copies']} burst copies. LLM mock calibrated from {c['calls']} "
        f"real Nova 2 Lite calls (p50 {c['upstream_latency_ms_p50']} ms); real Titan vectors.\n\n"
        + table(
            [
                "config",
                "hit rate",
                "exact hits",
                "semantic hits",
                "false hits",
                "LLM calls",
                "total cost",
                "net saving",
                "p50 ms",
                "p95 ms",
            ],
            rows,
        )  # fmt: skip
    )


def _versions(test: dict, sim: dict) -> str:
    """v1 vs v2 side by side, from reports/v1/ (frozen at tag v1.0.0) and reports/."""
    v1_dir = Path("reports") / "v1"
    if not (v1_dir / "pairs_test.json").exists():
        return "No v1 reports found."
    t1 = json.loads((v1_dir / "pairs_test.json").read_text(encoding="utf-8"))
    s1 = json.loads((v1_dir / "simulation.json").read_text(encoding="utf-8"))
    r1, r2 = _recommended(s1, t1["operating_point"]), _recommended(sim, test["operating_point"])

    def row(name, t, r):
        m, op, ci = t["result"], t["operating_point"], t.get("false_hit_rate_ci95")
        ci_text = f" ({pct(ci[0])}-{pct(ci[1])})" if ci else ""
        return [
            name, len(op["guards"]), op["t_high"], op["embed_dim"],
            f"{t['pairs']} pairs, {t['different']} should-miss",
            pct(m["false_hit_rate"]) + ci_text, pct(m["precision"]), pct(m["recall"]),
            r["false_hits"], pct(r["net_saving_pct"]),
        ]  # fmt: skip

    return table(
        ["version", "guards", "threshold", "dims", "test set", "test false-hit rate (95 % CI)",
         "test precision", "test recall", "simulation wrong answers", "simulation saving"],
        [row("v1 (tag v1.0.0)", t1, r1), row("v2", test, r2)],
    )  # fmt: skip


def blocks() -> dict[str, str]:
    dev, test, sim = _load("pairs_dev.json"), _load("pairs_test.json"), _load("simulation.json")
    dim = test["operating_point"]["embed_dim"]
    return {
        "headline": _headline(dev, test, sim),
        "test": _test_table(test),
        "dims": _dims(dev),
        "chart": f"![Threshold sweep](reports/sweep_{dim}.svg)",
        "ablation": _ablation(dev),
        "verifier": _verifier(dev),
        "simulation": _simulation(sim),
        "versions": _versions(test, sim),
    }


def sync_readme(path: Path = README) -> list[str]:
    text = path.read_text(encoding="utf-8")
    done = []
    for name, body in blocks().items():
        pattern = re.compile(
            rf"(<!-- results:{name} -->\n).*?(\n<!-- /results:{name} -->)", re.DOTALL
        )
        text, n = pattern.subn(lambda m, b=body: m.group(1) + b + m.group(2), text)
        if n:
            done.append(name)
    path.write_text(text, encoding="utf-8", newline="\n")
    return done
