"""The README's result blocks must equal what the committed reports produce."""

import re
from pathlib import Path

import pytest

REPORTS = [Path("reports") / n for n in ("pairs_dev.json", "pairs_test.json", "simulation.json")]


@pytest.mark.skipif(not all(p.exists() for p in REPORTS), reason="reports not generated yet")
def test_readme_numbers_match_reports():
    from semcache.eval.readme_sync import blocks

    readme = Path("README.md").read_text(encoding="utf-8")
    for name, body in blocks().items():
        m = re.search(rf"<!-- results:{name} -->\n(.*?)\n<!-- /results:{name} -->", readme, re.S)
        assert m, f"README has no block {name}"
        assert m.group(1) == body, f"README block {name} is stale: run `semcache readme-sync`"
