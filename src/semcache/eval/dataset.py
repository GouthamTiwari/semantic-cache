"""The labelled pair dataset (SPEC section 12.1): schema, validation, split, review sample."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from semcache.normalize import normalize

CATEGORIES = (
    "paraphrase",
    "near_miss_number",
    "negation",
    "entity_swap",
    "same_topic_different_question",
    "relation",  # v2: before/after, more/less, next/last, flipped comparisons
)
TRAP_CATEGORIES = CATEGORIES[1:]
DEV_SHARE = 0.7

PAIRS_PATH = Path("data/pairs/pairs.jsonl")
SEED_PATH = Path("data/pairs/seed.jsonl")


class Pair(BaseModel):
    id: str
    a: str = Field(min_length=3, max_length=300)
    b: str = Field(min_length=3, max_length=300)
    label: Literal["same", "different"]
    category: Literal[
        "paraphrase",
        "near_miss_number",
        "negation",
        "entity_swap",
        "same_topic_different_question",
        "relation",
    ]
    # "retired": the v1 test split. It was seen once, so v2 never uses it again.
    split: Literal["dev", "test", "retired"]
    group: str = "seed"
    source: str = "seed"
    domain: str = "mixed"


def load_pairs(path: str | Path = PAIRS_PATH) -> list[Pair]:
    with open(path, encoding="utf-8") as f:
        return [Pair.model_validate_json(line) for line in f if line.strip()]


def save_pairs(pairs: Iterable[Pair], path: str | Path = PAIRS_PATH) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for p in pairs:
            f.write(p.model_dump_json() + "\n")


def split_for_group(group: str, groups: list[str]) -> str:
    """Stable split: groups sorted by hash, the first 70 % are dev. Seeds are always dev."""
    if group == "seed":
        return "dev"
    ordered = sorted(set(groups) - {"seed"}, key=lambda g: hashlib.sha256(g.encode()).hexdigest())
    n_dev = round(len(ordered) * DEV_SHARE)
    return "dev" if ordered.index(group) < n_dev else "test"


def pair_signature(a: str, b: str) -> tuple[str, str]:
    x, y = sorted([normalize(a), normalize(b)])
    return x, y


def validate(pairs: list[Pair]) -> list[str]:
    """Problems that make the dataset unusable. Empty list = valid."""
    problems: list[str] = []
    ids = Counter(p.id for p in pairs)
    problems += [f"duplicate id {i}" for i, n in ids.items() if n > 1]
    seen: dict[tuple[str, str], str] = {}
    for p in pairs:
        if normalize(p.a) == normalize(p.b):
            problems.append(f"{p.id}: a and b are the same question")
        sig = pair_signature(p.a, p.b)
        if sig in seen:
            problems.append(f"{p.id}: duplicate of {seen[sig]}")
        seen[sig] = p.id
        if p.category == "paraphrase" and p.label != "same":
            problems.append(f"{p.id}: paraphrase pairs must be labelled same")
    # Leakage: no question may appear in both splits, and a group lives in one split only.
    texts: dict[str, set[str]] = {"dev": set(), "test": set(), "retired": set()}
    group_split: dict[str, set[str]] = {}
    for p in pairs:
        texts[p.split].update({normalize(p.a), normalize(p.b)})
        group_split.setdefault(p.group, set()).add(p.split)
    # The test split must be unseen: nothing from dev or from the retired v1 test.
    shared = (texts["dev"] | texts["retired"]) & texts["test"]
    problems += [f"text in both splits: {t!r}" for t in sorted(shared)]
    problems += [f"group {g} spans both splits" for g, s in group_split.items() if len(s) > 1]
    return problems


def stats(pairs: list[Pair]) -> dict:
    by = Counter((p.split, p.category, p.label) for p in pairs)
    out: dict = {
        "pairs": len(pairs),
        "same": sum(p.label == "same" for p in pairs),
        "different": sum(p.label == "different" for p in pairs),
        "groups": len({p.group for p in pairs}),
        "by_split": dict(Counter(p.split for p in pairs)),
        "by_domain": dict(Counter(p.domain for p in pairs)),
        "by_split_category_label": {f"{s}/{c}/{lab}": n for (s, c, lab), n in sorted(by.items())},
    }
    return out


def write_review_sample(
    pairs: list[Pair], path: str | Path, share: float = 0.10, seed: int = 7
) -> int:
    """A random 10 % sample for the human check, as a CSV that opens in any spreadsheet."""
    rng = random.Random(seed)
    generated = [p for p in pairs if p.source != "seed"]
    k = max(1, round(len(generated) * share))
    sample = sorted(rng.sample(generated, k), key=lambda p: p.id)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "category", "label", "a", "b", "reviewer_label", "reviewer_note"])
        for p in sample:
            w.writerow([p.id, p.category, p.label, p.a, p.b, "", ""])
    return k


def apply_review(pairs: list[Pair], review_csv: str | Path) -> list[str]:
    """Apply reviewer_label corrections from the review CSV; returns a change log."""
    changes: list[str] = []
    by_id = {p.id: p for p in pairs}
    with open(review_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            new = (row.get("reviewer_label") or "").strip().lower()
            if new in ("same", "different") and by_id[row["id"]].label != new:
                p = by_id[row["id"]]
                changes.append(f"{p.id}: {p.label} -> {new} ({row.get('reviewer_note', '')})")
                p.label = new  # type: ignore[assignment]
            elif new == "drop":
                changes.append(f"{row['id']}: dropped ({row.get('reviewer_note', '')})")
                del by_id[row["id"]]
    pairs[:] = list(by_id.values())
    return changes


def dump_stats(pairs: list[Pair]) -> str:
    return json.dumps(stats(pairs), indent=2)
