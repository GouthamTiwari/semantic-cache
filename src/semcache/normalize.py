"""Query normalisation used for matching (SPEC section 5).

Only the *matching* text is normalised; the original question is stored and returned as-is.
Numbers, negations and names are never removed, because they are exactly what the guards
compare.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

# Leading phrases that add politeness, not meaning. Longest first so "can you tell me"
# is removed before "tell me" would be considered.
_FILLERS = sorted(
    [
        "please",
        "hi",
        "hello",
        "hey",
        "can you tell me",
        "could you tell me",
        "can you please tell me",
        "could you please tell me",
        "tell me",
        "i want to know",
        "i would like to know",
        "do you know",
    ],
    key=len,
    reverse=True,
)
_FILLER_RE = re.compile(r"^(?:(?:" + "|".join(map(re.escape, _FILLERS)) + r")\b[\s,!.:]*)+")
_SPACE_RE = re.compile(r"\s+")
_TRAILING_PUNCT_RE = re.compile(r"[\s?.!]+$")


def normalize(text: str) -> str:
    """Lowercase, trim, collapse whitespace, drop leading filler and trailing punctuation."""
    t = unicodedata.normalize("NFKC", text).lower().strip()
    t = _SPACE_RE.sub(" ", t)
    stripped = _FILLER_RE.sub("", t)
    # Never strip a question down to nothing ("hello?" stays "hello").
    if stripped.strip(" ?.!"):
        t = stripped
    t = _TRAILING_PUNCT_RE.sub("", t)
    return t.strip()


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def exact_key(scope_key: str, normalized: str) -> str:
    """The L1 key: identical normalised text inside an identical scope."""
    return sha256(scope_key + "\n" + normalized)
