import pytest

from semcache.guards import extract_numbers, numbers_guard


@pytest.mark.parametrize(
    "a, b",
    [
        ("Can I return an item within 7 days?", "Can I return an item within 30 days?"),
        ("Which goods fall under the 5% GST slab?", "Which goods fall under the 18% GST slab?"),
        ("Is a UPI payment of Rs 1 lakh allowed?", "Is a UPI payment of Rs 5 lakh allowed?"),
        ("penalty for filing 10 days late", "penalty for filing 60 days late"),
        ("Is 2.5 kg allowed?", "Is 25 kg allowed?"),
        ("two lakh", "five lakh"),
    ],
)
def test_rejects_different_numbers(a, b):
    assert not numbers_guard(a, b)


@pytest.mark.parametrize(
    "a, b",
    [
        ("Return within 7 days?", "Can I send it back within seven days?"),
        ("Rs 1,00,000 limit", "Rs 100000 limit"),
        ("What is the repo rate?", "Current repo rate?"),  # no numbers on either side
        ("UPI limit of 2 lakh", "UPI limit of two lakh"),
        ("UPI limit of 2 lakh", "UPI limit of 200000"),
        ("What is the due date for GSTR-3B?", "When is GSTR-3B due?"),
    ],
)
def test_accepts_same_numbers(a, b):
    assert numbers_guard(a, b)


def test_years_are_left_to_the_dates_guard():
    assert extract_numbers("TDS rule in 2019") == extract_numbers("TDS rule in 2023")


def test_multiset_not_set():
    # "7 or 7" vs "7": counts matter
    assert not numbers_guard("7 and 7", "7")
