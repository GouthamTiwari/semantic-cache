import numpy as np
import pytest

from semcache.embed import EmbedResult
from semcache.eval.dataset import Pair
from semcache.eval.pairs_eval import metrics, pick_operating_point, run_dev, run_test
from semcache.eval.report import svg_sweep
from semcache.normalize import normalize
from semcache.verify import StaticVerifier


class AngleEmbedder:
    """Puts b at a chosen cosine similarity from a (via a per-pair angle table)."""

    model_id = "mock-embed"
    dimensions = 2

    def __init__(self, sims: dict[str, float]):
        self.sims = {normalize(k): v for k, v in sims.items()}

    def embed(self, text):
        sim = self.sims.get(text, 1.0)
        theta = np.arccos(np.clip(sim, -1, 1))
        v = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
        return EmbedResult(v, 5, 100.0, cost_usd=1e-7)


def make(i, a, b, label, cat, sim, split="dev"):
    p = Pair(id=f"p{i}", a=a, b=b, label=label, category=cat, split=split, group=f"g{i}",
             source="generated")  # fmt: skip
    return p, sim


def dataset():
    rows = []
    for i in range(40):  # paraphrases, high similarity
        rows.append(make(i, f"how do I reset password {i}", f"reset my password {i}",
                         "same", "paraphrase", 0.90 + (i % 5) * 0.015))  # fmt: skip
    for i in range(40, 60):  # number traps: very similar, guard must catch them
        rows.append(make(i, f"refund in 7 days {i}x", f"refund in 30 days {i}x",
                         "different", "near_miss_number", 0.97))  # fmt: skip
    for i in range(60, 100):  # unrelated-ish, lower similarity
        rows.append(make(i, f"what is kyc {i}", f"how to do kyc online {i}",
                         "different", "same_topic_different_question",
                         0.80 + (i % 5) * 0.01))  # fmt: skip
    for i in range(100, 120):  # test split
        rows.append(make(i, f"open account {i}", f"how to open an account {i}", "same",
                         "paraphrase", 0.93, split="test"))  # fmt: skip
    pairs = [p for p, _ in rows]
    sims = {p.b: s for p, s in rows}
    return pairs, sims


def test_metrics_formulas():
    same = np.array([True, True, False, False, False])
    pred = np.array([True, False, True, False, False])
    pairs = [Pair(id=str(i), a="aaa", b=f"bbb{i}", label="same" if s else "different",
                  category="paraphrase" if s else "negation", split="dev") for i, s in
             enumerate(same)]  # fmt: skip
    m = metrics(pred, same, pairs)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 2)
    assert m["precision"] == 0.5 and m["recall"] == 0.5
    assert m["false_hit_rate"] == pytest.approx(1 / 3, abs=1e-4)


def test_pick_operating_point_is_lowest_threshold_meeting_target():
    rows = [{"threshold": 0.8, "false_hit_rate": 0.05}, {"threshold": 0.85, "false_hit_rate": 0.01},
            {"threshold": 0.9, "false_hit_rate": 0.0}]  # fmt: skip
    assert pick_operating_point(rows)["threshold"] == 0.85
    assert pick_operating_point([{"threshold": 0.9, "false_hit_rate": 0.5}]) is None


def test_run_dev_end_to_end_and_test_split():
    pairs, sims = dataset()
    emb = AngleEmbedder(sims)
    verifier = StaticVerifier(lambda a, b: "reset" in a)  # says yes only to paraphrases
    res = run_dev({2: emb}, verifier, pairs)
    d = res["dims"][2]
    # Without guards, number traps (0.97) force the threshold above 0.97; guards fix that.
    assert d["operating_point_threshold_only"]["threshold"] > 0.97
    assert d["operating_point_with_guards"]["threshold"] <= 0.90
    ab = {r["guards"]: r for r in res["guard_ablation"]["rows"]}
    assert ab["numbers"]["false_hits_removed"] == 20
    assert ab["negation"]["false_hits_removed"] == 0
    op = res["operating_point"]
    assert op["chosen_on"] == "dev split only"
    t = run_test(emb, verifier, op, pairs)
    assert t["pairs"] == 20 and t["result"]["recall"] == 1.0


def test_svg_chart_is_well_formed():
    import xml.dom.minidom

    svg = svg_sweep("t", [0.7, 0.8, 0.9], [("a", [0.1, 0.5, 0.9], "#000", False)], 0.8, "x")
    xml.dom.minidom.parseString(svg)


def test_wilson_interval():
    from semcache.eval.pairs_eval import wilson

    lo, hi = wilson(2, 93)
    assert lo < 2 / 93 < hi
    assert wilson(0, 50)[0] == 0.0 and wilson(0, 50)[1] > 0
