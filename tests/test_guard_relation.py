import pytest

from semcache.guards import comparisons, negation_guard, relation_guard, run_guards


@pytest.mark.parametrize(
    "a, b",
    [
        # the v1 simulation's false hits
        ("Can I withdraw my EPF funds before turning 55?", "Can I withdraw my EPF funds after turning 55?"),
        ("I want to know if I can redeem my points on June 30.", "I want to know if I can redeem my points following June 30."),
        ("Is it possible to use my points on June 30th?", "Is it possible to use my points before June 30th?"),
        ("Delete my data before next Friday?", "Delete my data before last Friday?"),
        ("Is football older than cricket?", "Is cricket older than football?"),
        # other axes
        ("Fee for orders above Rs 500?", "Fee for orders below Rs 500?"),
        ("Fee for orders over Rs 500?", "Fee for orders of Rs 500?"),
        ("What is the maximum UPI limit?", "What is the minimum UPI limit?"),
        ("Will RBI increase the repo rate?", "Will RBI cut the repo rate?"),
        ("Can I file the return until March?", "Can I file the return since March?"),
    ],
)  # fmt: skip
def test_rejects_opposite_relations(a, b):
    assert not relation_guard(a, b)


@pytest.mark.parametrize(
    "a, b",
    [
        ("Can I withdraw EPF before 55?", "Can I withdraw EPF prior to turning 55?"),
        ("Is football older than cricket?", "Is cricket younger than football?"),
        ("What is the last date to file ITR?", "When is the ITR due date?"),  # "last" one-sided
        ("What is the repo rate?", "Tell me the repo rate"),
        ("Fee for orders above Rs 500?", "Charges for orders exceeding Rs 500?"),
        ("Is this over?", "Has this ended?"),  # "over" without an amount is not a bound
        ("Which of the following documents are needed?", "Which documents are needed?"),
        ("How do I reset my password?", "Steps to reset my password?"),
    ],
)  # fmt: skip
def test_accepts_same_relations(a, b):
    assert relation_guard(a, b)


def test_comparisons_canonicalise_antonyms():
    assert comparisons("Is cricket younger than football?") == comparisons(
        "Is football older than cricket?"
    )


def test_false_that_counts_as_negation():
    a = "Is it false that I have to submit GSTR-1 by the 10th?"
    b = "Do I have to submit GSTR-1 by the 10th?"
    assert not negation_guard(a, b)
    assert negation_guard("It is untrue that X", "Is X not true")  # both negated once


def test_registered_in_run_guards():
    assert run_guards("withdraw before 55", "withdraw after 55", ("relation",)) == "relation"
