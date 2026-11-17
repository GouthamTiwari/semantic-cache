import pytest

from semcache.guards import negation_count, negation_guard


@pytest.mark.parametrize(
    "a, b",
    [
        (
            "Which purchases are eligible for input tax credit?",
            "Which purchases are not eligible for input tax credit?",
        ),
        (
            "When is a bank allowed to freeze an account?",
            "When is a bank not allowed to freeze an account?",
        ),
        ("Can I cancel my order?", "Why can't I cancel my order?"),
        ("Can I withdraw with a penalty?", "Can I withdraw without a penalty?"),
        ("Is GST charged on this?", "Is GST never charged on this?"),
    ],
)
def test_rejects_polarity_mismatch(a, b):
    assert not negation_guard(a, b)


@pytest.mark.parametrize(
    "a, b",
    [
        ("Which items are not refundable?", "Which items can't be refunded?"),
        ("Which payments are not subject to TDS?", "Which payments don't attract TDS?"),
        ("How do I reset my password?", "I forgot my password, how do I change it?"),
        ("Isn't it true that X is not allowed?", "Is X allowed?"),  # double negation = positive
    ],
)
def test_accepts_same_polarity(a, b):
    assert negation_guard(a, b)


def test_curly_apostrophe():
    assert negation_count("I can\u2019t log in") == 1
