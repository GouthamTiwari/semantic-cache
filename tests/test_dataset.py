import csv

from semcache.eval.dataset import (
    Pair,
    apply_review,
    load_pairs,
    split_for_group,
    validate,
    write_review_sample,
)
from semcache.eval.generate import finalize, parse_pairs


def pair(i, a, b, label="same", cat="paraphrase", split="dev", group="g1"):
    return Pair(id=i, a=a, b=b, label=label, category=cat, split=split, group=group,
                source="generated")  # fmt: skip


def test_committed_dataset_is_valid():
    pairs = load_pairs()
    assert validate(pairs) == []
    assert len(pairs) >= 550
    same = sum(p.label == "same" for p in pairs) / len(pairs)
    assert 0.40 <= same <= 0.60
    assert {p.category for p in pairs} == {
        "paraphrase", "near_miss_number", "negation", "entity_swap",
        "same_topic_different_question", "relation",
    }  # fmt: skip
    # v2: the v1 test split is retired, and the new test split shares no text with it.
    assert {p.split for p in pairs} == {"dev", "test", "retired"}


def test_validate_catches_leakage_and_duplicates():
    pairs = [
        pair("1", "What is KYC?", "KYC meaning?", group="g1", split="dev"),
        pair("2", "kyc meaning", "Define KYC", group="g2", split="test"),  # text in both splits
        pair("3", "KYC meaning?", "What is KYC?", group="g1"),  # duplicate of 1
        pair("4", "Same text", "same text?", group="g1"),  # a == b after normalising
        pair("5", "x is y", "y is x", label="different", group="g1"),  # paraphrase must be same
        pair("6", "Old test question?", "Old test b", group="g3", split="retired"),
        pair("7", "old test question", "New b", group="g4", split="test"),  # reuses retired
    ]
    problems = " | ".join(validate(pairs))
    assert "text in both splits" in problems
    assert "duplicate of 1" in problems
    assert "same question" in problems
    assert "paraphrase pairs must be labelled same" in problems
    assert "'old test question'" in problems


def test_split_is_stable_and_by_group():
    groups = [f"t{i:02d}" for i in range(40)] + ["seed"]
    splits = [split_for_group(g, groups) for g in groups]
    assert splits == [split_for_group(g, list(reversed(groups))) for g in groups]
    assert splits[-1] == "dev"
    assert splits.count("test") == 12


def test_finalize_drops_test_pairs_that_reuse_dev_text():
    groups = [f"t{i:02d}" for i in range(10)]
    test_group = next(g for g in groups if split_for_group(g, groups) == "test")
    dev_group = next(g for g in groups if split_for_group(g, groups) == "dev")
    pairs = [pair("a", "Shared question?", "Other one", group=dev_group)]
    pairs += [pair("b", "shared question", "Third one", group=test_group)]
    pairs += [pair(f"x{g}", f"q {g}", f"r {g}", group=g) for g in groups]
    finalize(pairs)
    assert not any(p.a == "shared question" for p in pairs)


def test_parse_pairs_maps_keys_to_categories():
    raw = {
        "paraphrase_same": [["How do I reset it?", "Reset steps?"]],
        "number_different": [["Return in 7 days?", "Return in 30 days?"], ["bad"]],
        "negation_same": [["Not refundable?", "Can't be refunded?"]],
    }
    out = parse_pairs(raw, "t01", "support")
    assert [(p.category, p.label) for p in out] == [
        ("paraphrase", "same"), ("near_miss_number", "different"), ("negation", "same"),
    ]  # fmt: skip


def test_review_sample_and_apply(tmp_path):
    pairs = [pair(str(i), f"question {i} a", f"question {i} b") for i in range(30)]
    path = tmp_path / "r.csv"
    assert write_review_sample(pairs, path) == 3
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    rows[0]["reviewer_label"] = "different"
    rows[1]["reviewer_label"] = "drop"
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    changes = apply_review(pairs, path)
    assert len(changes) == 2 and len(pairs) == 29
