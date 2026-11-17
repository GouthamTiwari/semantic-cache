import pytest

from semcache.guards import entities_guard, extract_entities, run_guards


@pytest.mark.parametrize(
    "a, b",
    [
        ("What does RBI say about digital lending?", "What does SEBI say about digital lending?"),
        ("How do I update my address on my Aadhaar?", "How do I update my address on my PAN?"),
        (
            "How do I track my order on the Android app?",
            "How do I track my order on the iPhone app?",
        ),
        ("What is the capital of Australia?", "What is the capital of Austria?"),
        ('What does "net banking" cost?', 'What does "mobile banking" cost?'),
    ],
)
def test_swap_mode_rejects_swaps(a, b):
    assert not entities_guard(a, b, mode="swap")


def test_swap_mode_allows_one_sided_context():
    a, b = "What is the repo rate?", "Current RBI repo rate?"
    assert entities_guard(a, b, mode="swap")
    assert not entities_guard(a, b, mode="strict")


def test_lowercase_swaps_are_not_caught():
    # Known limit, measured in the eval: no capital letters, no key terms.
    assert entities_guard("GST rate on restaurants?", "GST rate on hotels?", mode="strict")


def test_sentence_start_capital_is_not_an_entity():
    assert extract_entities("Which goods are taxed?") == set()
    assert extract_entities("Is it taxed. Which goods?") == set()
    assert extract_entities("What does KYC stand for?") == {"kyc"}


def test_run_guards_returns_first_failure():
    assert run_guards("refund in 7 days", "refund in 30 days", ("negation", "numbers")) == "numbers"
    assert run_guards("a", "a", ("numbers", "dates", "negation", "entities")) is None
    assert run_guards("RBI rules", "repo rules by RBI", ("entities",), entity_mode="strict") is None
