import pytest

from semcache.guards import dates_guard, extract_dates


@pytest.mark.parametrize(
    "a, b",
    [
        (
            "What was the rule for TDS on rent in 2019?",
            "What was the rule for TDS on rent in 2023?",
        ),
        ("GST due on 20/03/2024", "GST due on 20/04/2024"),
        ("ITR deadline for FY 2023-24", "ITR deadline for FY 2024-25"),
        ("Returns filed in March", "Returns filed in April"),
        ("Is the bank open on Saturday?", "Is the bank open on Sunday?"),
        ("Due on 5 May", "Due on 5 June"),
    ],
)
def test_rejects_different_dates(a, b):
    assert not dates_guard(a, b)


@pytest.mark.parametrize(
    "a, b",
    [
        ("What is the repo rate?", "Tell me the repo rate"),
        ("Rules for FY24", "Rules for FY 2024"),
        ("ITR deadline for 2023-24", "ITR deadline for 2023-2024"),
        ("May I return this?", "Can I return this?"),  # "may" as a verb is not a month
        ("TDS rule in 2023", "In 2023 what was the TDS rule"),
    ],
)
def test_accepts_same_dates(a, b):
    assert dates_guard(a, b)


def test_extract_dates_shapes():
    assert extract_dates("on 2024-03-31") == {"date:2024-3-31"}
    assert "year:2019" in extract_dates("in 2019")
