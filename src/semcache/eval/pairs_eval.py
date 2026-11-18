"""Pair evaluation: threshold sweep, dimension choice, guard ablation, verifier, test run.

SPEC sections 12.2-12.3. Everything here reads embeddings and verdicts from the on-disk
memos, so re-running it is free and gives the same numbers. Accept/reject decisions go
through `semcache.decision.evaluate_candidate`, the same function the engine uses.

Pair convention: `a` is the cached question, `b` the new question.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from semcache.config import CacheConfig
from semcache.decision import evaluate_candidate
from semcache.eval.dataset import CATEGORIES, Pair, load_pairs
from semcache.normalize import normalize

FHR_TARGET = 0.01  # false-hit rate target on `different` pairs (SPEC section 11)
THRESHOLDS = [round(0.70 + 0.01 * i, 2) for i in range(30)]  # 0.70 .. 0.99
DIMS = (256, 1024)
ALL_GUARDS = ("numbers", "dates", "negation", "entities")
# The cheaper dimension must justify itself: chosen only if its dev recall at the target
# is no more than this many points below the larger one's.
DIM_RECALL_TOLERANCE = 0.02
# The verifier is "worth it" only if it adds at least this much dev recall at the target.
VERIFIER_MIN_RECALL_GAIN = 0.02
GREY_WIDTHS = (0.02, 0.04, 0.06, 0.08, 0.10)
REPORTS = Path("reports")


@dataclass
class Scored:
    pairs: list[Pair]
    scores: np.ndarray  # cosine similarity of normalised a and b
    same: np.ndarray  # bool: label == same
    embed_cost_per_lookup: float  # mean embedding cost of one question
    embed_ms_per_lookup: float


def score_pairs(pairs: list[Pair], embedder) -> Scored:
    scores, costs, lat = [], [], []
    for p in pairs:
        ea, eb = embedder.embed(normalize(p.a)), embedder.embed(normalize(p.b))
        scores.append(float(np.dot(ea.vector, eb.vector)))
        costs.append(eb.cost_usd)
        lat.append(eb.latency_ms)
    return Scored(
        pairs,
        np.asarray(scores),
        np.asarray([p.label == "same" for p in pairs]),
        float(np.mean(costs)),
        float(np.median(lat)),
    )


def metrics(pred: np.ndarray, same: np.ndarray, pairs: list[Pair]) -> dict:
    tp = int(np.sum(pred & same))
    fp = int(np.sum(pred & ~same))
    fn = int(np.sum(~pred & same))
    tn = int(np.sum(~pred & ~same))
    out = {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(tp / (tp + fp), 4) if tp + fp else 1.0,
        "recall": round(tp / (tp + fn), 4) if tp + fn else 0.0,
        "false_hit_rate": round(fp / (fp + tn), 4) if fp + tn else 0.0,
        "by_category": {},
    }  # fmt: skip
    cats = np.asarray([p.category for p in pairs])
    for c in CATEGORIES:
        m = cats == c
        d, s = m & ~same, m & same
        out["by_category"][c] = {
            "different": int(d.sum()),
            "false_hits": int((pred & d).sum()),
            "false_hit_rate": round(float((pred & d).sum() / d.sum()), 4) if d.sum() else None,
            "same": int(s.sum()),
            "recall": round(float((pred & s).sum() / s.sum()), 4) if s.sum() else None,
        }
    return out


def predict(s: Scored, cfg: CacheConfig, verifier=None) -> tuple[np.ndarray, dict]:
    """Run the production decision function on every pair."""
    pred = np.zeros(len(s.pairs), dtype=bool)
    judged = 0
    judge_cost = 0.0
    judge_ms: list[float] = []
    for i, p in enumerate(s.pairs):
        v = evaluate_candidate(p.b, p.a, float(s.scores[i]), cfg, verifier)
        pred[i] = v.accept
        if v.judged:
            judged += 1
            judge_cost += v.judge_cost_usd
            judge_ms.append(v.judge_ms)
    info = {
        "judged": judged,
        "judge_cost_usd": round(judge_cost, 8),
        "judge_ms_p50": round(float(np.median(judge_ms)), 1) if judge_ms else 0.0,
    }
    return pred, info


def cfg_for(t: float, guards: tuple[str, ...], *, t_low: float | None = None,
            verifier: bool = False, entity_mode: str = "swap") -> CacheConfig:  # fmt: skip
    return CacheConfig(
        t_high=t, t_low=t if t_low is None else t_low, guards=guards,
        entity_mode=entity_mode, verifier_enabled=verifier,
    )  # fmt: skip


def sweep(s: Scored, guards: tuple[str, ...]) -> list[dict]:
    rows = []
    for t in THRESHOLDS:
        pred, _ = predict(s, cfg_for(t, guards))
        m = metrics(pred, s.same, s.pairs)
        rows.append({"threshold": t, **{k: v for k, v in m.items() if k != "by_category"}})
    return rows


def pick_operating_point(rows: list[dict]) -> dict | None:
    """Lowest threshold (= highest recall) whose false-hit rate meets the target."""
    ok = [r for r in rows if r["false_hit_rate"] <= FHR_TARGET]
    return min(ok, key=lambda r: r["threshold"]) if ok else None


def _subset(s: Scored, split: str) -> Scored:
    idx = [i for i, p in enumerate(s.pairs) if p.split == split]
    return Scored(
        [s.pairs[i] for i in idx], s.scores[idx], s.same[idx],
        s.embed_cost_per_lookup, s.embed_ms_per_lookup,
    )  # fmt: skip


def run_dev(embedders: dict[int, object], verifier, pairs: list[Pair] | None = None) -> dict:
    """Experiments 1-4 on the dev split. Writes nothing; returns the results."""
    pairs = pairs if pairs is not None else load_pairs()
    result: dict = {
        "fhr_target": FHR_TARGET,
        "embedding_regions": {
            "1024": "ap-south-1",
            "256": "us-east-1 (a second regional quota, to finish in time; a spot check gave "
            "cosine 1.000000 against the same text embedded in ap-south-1)",
        },
        "dims": {},
    }
    scored: dict[int, Scored] = {}
    for dim, emb in embedders.items():
        s = _subset(score_pairs(pairs, emb), "dev")
        scored[dim] = s
        no_guards = sweep(s, ())
        with_guards = sweep(s, ALL_GUARDS)
        result["dims"][dim] = {
            "embed_cost_per_lookup_usd": s.embed_cost_per_lookup,
            "score_quantiles": {
                lab: [round(float(q), 4) for q in np.quantile(s.scores[mask], [0.05, 0.5, 0.95])]
                for lab, mask in (("same", s.same), ("different", ~s.same))
            },
            "sweep_threshold_only": no_guards,
            "sweep_with_guards": with_guards,
            "operating_point_threshold_only": pick_operating_point(no_guards),
            "operating_point_with_guards": pick_operating_point(with_guards),
        }

    # Experiment 2: which dimension?
    def recall_at_target(dim):
        op = result["dims"][dim]["operating_point_with_guards"]
        return op["recall"] if op else 0.0

    small, large = min(embedders), max(embedders)
    chosen_dim = (
        small
        if recall_at_target(small) >= recall_at_target(large) - DIM_RECALL_TOLERANCE
        else large
    )
    result["dimension_choice"] = {
        "rule": f"choose {small} only if its dev recall at FHR <= {FHR_TARGET:.0%} is within "
        f"{DIM_RECALL_TOLERANCE:.0%} of {large}'s",
        "recall_at_target": {d: recall_at_target(d) for d in embedders},
        "chosen": chosen_dim,
    }
    s = scored[chosen_dim]
    op = result["dims"][chosen_dim]["operating_point_with_guards"]
    if op is None:
        raise RuntimeError("no threshold meets the false-hit target on dev")
    t_star = op["threshold"]

    # Experiment 3: guard ablation at the chosen threshold.
    variants: list[tuple[str, tuple[str, ...], str]] = [("none", (), "swap")]
    variants += [(g, (g,), "swap") for g in ALL_GUARDS]
    variants += [("all", ALL_GUARDS, "swap"), ("all (entities strict)", ALL_GUARDS, "strict")]
    base_pred, _ = predict(s, cfg_for(t_star, ()))
    base_fp = int(np.sum(base_pred & ~s.same))
    ablation = []
    for name, guards, mode in variants:
        pred, _ = predict(s, cfg_for(t_star, guards, entity_mode=mode))
        m = metrics(pred, s.same, s.pairs)
        above = s.scores >= t_star
        rejected_same = int(np.sum(above & s.same & ~pred))
        ablation.append({
            "guards": name,
            "false_hits": m["fp"],
            "false_hits_removed": base_fp - m["fp"],
            "false_hit_rate": m["false_hit_rate"],
            "recall": m["recall"],
            "precision": m["precision"],
            "same_pairs_wrongly_rejected": rejected_same,
            "by_category_false_hits": {c: v["false_hits"] for c, v in m["by_category"].items()},
        })  # fmt: skip
    result["guard_ablation"] = {"threshold": t_star, "dim": chosen_dim, "rows": ablation}

    # Experiment 4: grey-zone verifier, off vs on, for several grey-zone widths.
    pred_off, _ = predict(s, cfg_for(t_star, ALL_GUARDS))
    off = metrics(pred_off, s.same, s.pairs)
    rows = [{"t_low": t_star, "verifier": "off", "recall": off["recall"],
             "false_hit_rate": off["false_hit_rate"], "precision": off["precision"],
             "judge_calls": 0, "judge_cost_usd": 0.0, "judge_ms_p50": 0.0}]  # fmt: skip
    for w in GREY_WIDTHS:
        t_low = round(t_star - w, 2)
        pred, info = predict(s, cfg_for(t_star, ALL_GUARDS, t_low=t_low, verifier=True), verifier)
        m = metrics(pred, s.same, s.pairs)
        rows.append({"t_low": t_low, "verifier": "on", "recall": m["recall"],
                     "false_hit_rate": m["false_hit_rate"], "precision": m["precision"],
                     "judge_calls": info["judged"], "judge_cost_usd": info["judge_cost_usd"],
                     "judge_ms_p50": info["judge_ms_p50"]})  # fmt: skip
    eligible = [r for r in rows[1:] if r["false_hit_rate"] <= FHR_TARGET]
    best = max(eligible, key=lambda r: (r["recall"], r["t_low"]), default=None)
    worth = bool(best and best["recall"] - off["recall"] >= VERIFIER_MIN_RECALL_GAIN)
    result["verifier"] = {
        "rule": f"worth it only if it adds >= {VERIFIER_MIN_RECALL_GAIN:.0%} dev recall while "
        f"keeping FHR <= {FHR_TARGET:.0%}",
        "rows": rows,
        "best": best,
        "worth_it": worth,
    }
    result["operating_point"] = {
        "embed_dim": chosen_dim,
        "t_high": t_star,
        "t_low": best["t_low"] if worth else t_star,
        "verifier_enabled": worth,
        "guards": list(ALL_GUARDS),
        "entity_mode": "swap",
        "fhr_target": FHR_TARGET,
        "chosen_on": "dev split only",
    }
    return result


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % Wilson score interval for k successes out of n (sound for small counts)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return (round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4))


def op_hash(op: dict) -> str:
    return hashlib.sha256(json.dumps(op, sort_keys=True).encode()).hexdigest()[:12]


def run_test(embedder, verifier, op: dict, pairs: list[Pair] | None = None) -> dict:
    """Experiment 5: the held-out test split, once, at the frozen operating point."""
    s = _subset(score_pairs(pairs if pairs is not None else load_pairs(), embedder), "test")
    cfg = CacheConfig(
        t_high=op["t_high"], t_low=op["t_low"], guards=tuple(op["guards"]),
        entity_mode=op["entity_mode"], verifier_enabled=op["verifier_enabled"],
    )  # fmt: skip
    pred, info = predict(s, cfg, verifier if op["verifier_enabled"] else None)
    m = metrics(pred, s.same, s.pairs)
    pred_ng, _ = predict(s, cfg_for(op["t_high"], ()))
    m_ng = metrics(pred_ng, s.same, s.pairs)
    return {
        "operating_point_hash": op_hash(op),
        "operating_point": op,
        "pairs": len(s.pairs),
        "same": int(s.same.sum()),
        "different": int((~s.same).sum()),
        "result": m,
        "false_hit_rate_ci95": wilson(m["fp"], m["fp"] + m["tn"]),
        "judge": info,
        "same_threshold_without_guards": {
            k: m_ng[k] for k in ("false_hit_rate", "recall", "precision", "fp")
        },
    }


# --- report rendering -------------------------------------------------------------


def write_dev_reports(result: dict) -> None:
    from semcache.eval.dataset import stats
    from semcache.eval.report import pct, svg_sweep, table, usd, write_json, write_text

    write_json("pairs_dev.json", result)
    write_json("operating_point.json", result["operating_point"])
    op = result["operating_point"]
    lines = ["# Pair evaluation: dev split", ""]
    lines += [
        "Generated by `semcache sweep` from `data/pairs/pairs.jsonl` and the embedding memo. "
        "Only the dev split is used here; the test split is reported once in `pairs_test.md`.",
        "",
        "## Dataset",
        "",
        "```json",
        json.dumps(stats(load_pairs()), indent=2),
        "```",
        "",
        f"## Operating points (lowest threshold with dev false-hit rate <= {pct(FHR_TARGET, 0)})",
        "",
    ]
    rows = []
    for dim, d in result["dims"].items():
        for name, key in (("threshold only", "operating_point_threshold_only"),
                          ("threshold + guards", "operating_point_with_guards")):  # fmt: skip
            o = d[key]
            rows.append([dim, name, o["threshold"] if o else "none", pct(o["precision"]) if o
                         else "-", pct(o["recall"]) if o else "-",
                         pct(o["false_hit_rate"]) if o else "-"])  # fmt: skip
    lines += [table(["dim", "config", "threshold", "precision", "recall", "false-hit rate"],
                    rows), ""]  # fmt: skip
    dc = result["dimension_choice"]
    lines += [
        f"**Dimension:** {dc['chosen']}. Rule: {dc['rule']}. Recall at target: "
        + ", ".join(f"{d} = {pct(r)}" for d, r in dc["recall_at_target"].items()),
        "",
    ]
    lines += ["## Similarity scores (5th / 50th / 95th percentile)", ""]
    lines += [table(["dim", "same pairs", "different pairs"], [
        [dim, " / ".join(f"{q:.3f}" for q in d["score_quantiles"]["same"]),
         " / ".join(f"{q:.3f}" for q in d["score_quantiles"]["different"])]
        for dim, d in result["dims"].items()
    ]), ""]  # fmt: skip
    ga = result["guard_ablation"]
    lines += [f"## Guard ablation (dim {ga['dim']}, threshold {ga['threshold']})", ""]
    lines += [table(
        ["guards", "false hits", "removed vs none", "false-hit rate", "recall",
         "same pairs wrongly rejected"],
        [[r["guards"], r["false_hits"], r["false_hits_removed"], pct(r["false_hit_rate"]),
          pct(r["recall"]), r["same_pairs_wrongly_rejected"]] for r in ga["rows"]],
    ), ""]  # fmt: skip
    cats = list(ga["rows"][0]["by_category_false_hits"])
    lines += ["False hits by category:", ""]
    lines += [table(["guards", *cats], [[r["guards"], *[r["by_category_false_hits"][c]
                    for c in cats]] for r in ga["rows"]]), ""]  # fmt: skip
    v = result["verifier"]
    lines += ["## Grey-zone verifier", "", f"Rule: {v['rule']}.", ""]
    lines += [table(
        ["t_low", "verifier", "recall", "false-hit rate", "precision", "judge calls",
         "judge cost", "judge p50 ms"],
        [[r["t_low"], r["verifier"], pct(r["recall"]), pct(r["false_hit_rate"]),
          pct(r["precision"]), r["judge_calls"], usd(r["judge_cost_usd"]), r["judge_ms_p50"]]
         for r in v["rows"]],
    ), ""]  # fmt: skip
    lines += [f"**Verifier worth it:** {'yes' if v['worth_it'] else 'no'}.", ""]
    lines += ["## Chosen operating point", "", "```json", json.dumps(op, indent=2), "```", ""]
    lines += ["## Sweep (dev, with guards)", ""]
    for dim, d in result["dims"].items():
        lines += [f"### dim {dim}", ""]
        heads = ["threshold", "precision", "recall", "false-hit rate", "FHR without guards"]
        lines += [table(heads,
                        [[r["threshold"], pct(r["precision"]), pct(r["recall"]),
                          pct(r["false_hit_rate"]), pct(n["false_hit_rate"])]
                         for r, n in zip(d["sweep_with_guards"], d["sweep_threshold_only"],
                                         strict=True) if r["threshold"] >= 0.75]), ""]  # fmt: skip
    write_text("pairs_dev.md", "\n".join(lines))

    for dim, d in result["dims"].items():
        g, n = d["sweep_with_guards"], d["sweep_threshold_only"]
        xs = [r["threshold"] for r in g]
        o = d["operating_point_with_guards"]
        svg = svg_sweep(
            f"Titan v2 ({dim} dims), dev split: threshold sweep",
            xs,
            [
                ("precision (guards)", [r["precision"] for r in g], "#2563eb", False),
                ("recall (guards)", [r["recall"] for r in g], "#16a34a", False),
                ("false-hit rate (guards)", [r["false_hit_rate"] for r in g], "#dc2626", False),
                ("false-hit rate (no guards)", [r["false_hit_rate"] for r in n], "#dc2626", True),
                ("recall (no guards)", [r["recall"] for r in n], "#16a34a", True),
            ],
            marker_x=o["threshold"] if o else None,
            marker_label=f"chosen {o['threshold']}" if o else "",
        )
        write_text(f"sweep_{dim}.svg", svg)


def write_test_report(res: dict) -> None:
    from semcache.eval.report import pct, table, write_json, write_text

    write_json("pairs_test.json", res)
    m = res["result"]
    ng = res["same_threshold_without_guards"]
    lines = [
        "# Pair evaluation: held-out test split",
        "",
        f"Run once at the operating point frozen on dev (hash `{res['operating_point_hash']}`). "
        "Nothing was tuned after this run.",
        "",
        f"Test pairs: {res['pairs']} ({res['same']} same, {res['different']} different). "
        f"False-hit rate 95 % interval (Wilson): {pct(res['false_hit_rate_ci95'][0])} to "
        f"{pct(res['false_hit_rate_ci95'][1])}.",
        "",
        table(["config", "precision", "recall", "false-hit rate", "false hits"], [
            ["chosen operating point", pct(m["precision"]), pct(m["recall"]),
             pct(m["false_hit_rate"]), m["fp"]],
            ["same threshold, no guards", pct(ng["precision"]), pct(ng["recall"]),
             pct(ng["false_hit_rate"]), ng["fp"]],
        ]),
        "",
        "By category:",
        "",
        table(["category", "different pairs", "false hits", "false-hit rate", "same pairs",
               "recall"],
              [[c, v["different"], v["false_hits"], pct(v["false_hit_rate"]), v["same"],
                pct(v["recall"])] for c, v in m["by_category"].items()]),
        "",
        "```json",
        json.dumps(res["operating_point"], indent=2),
        "```",
        "",
    ]  # fmt: skip
    write_text("pairs_test.md", "\n".join(lines))
