"""Deterministic guards: cheap checks that catch what embeddings blur (SPEC section 7).

"Refund within 7 days" and "refund within 30 days" embed almost identically, but need
different answers. Each guard compares the new question with the cached one and returns
True when they agree (the hit may proceed) or False (reject the hit).
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

# --- numbers ----------------------------------------------------------------------

_WORD_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1_000,
    "lakh": 100_000, "lakhs": 100_000, "crore": 10_000_000, "crores": 10_000_000,
    "million": 1_000_000, "billion": 1_000_000_000, "dozen": 12, "twice": 2,
}  # fmt: skip
_MULTIPLIERS = {"hundred", "thousand", "lakh", "lakhs", "crore", "crores", "million", "billion"}

# Years and dates belong to the dates guard; strip them before reading numbers.
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_FY_RE = re.compile(r"\bfy\s?'?(\d{2,4})(?:\s?-\s?(\d{2,4}))?\b", re.IGNORECASE)
_YEAR_RANGE_RE = re.compile(r"\b((?:19|20)\d{2})\s?[-/]\s?(\d{2,4})\b")
_DATE_NUMERIC_RE = re.compile(
    r"\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})\b|\b(\d{4})-(\d{2})-(\d{2})\b"
)
_NUMBER_RE = re.compile(r"(?<![\w.])\d+(?:,\d{2,3})*(?:\.\d+)?(?![\w])|(?<![\w.])\d+(?=[a-z%])")
_WORD_RE = re.compile(r"[a-z]+")


def _strip_dates(text: str) -> str:
    t = _DATE_NUMERIC_RE.sub(" ", text)
    t = _FY_RE.sub(" ", t)
    t = _YEAR_RANGE_RE.sub(" ", t)
    return _YEAR_RE.sub(" ", t)


def extract_numbers(text: str) -> Counter:
    """Numeric values in the text, digits and number words alike ("7" == "seven")."""
    t = _strip_dates(text.lower())
    values: list[float] = []
    for m in _NUMBER_RE.finditer(t):
        values.append(float(m.group(0).replace(",", "")))
    # Number words, combining "two lakh" -> 200000 and "five hundred" -> 500.
    words = _WORD_RE.findall(t)
    i = 0
    while i < len(words):
        w = words[i]
        if w in _WORD_NUMBERS and w not in _MULTIPLIERS:
            value = float(_WORD_NUMBERS[w])
            if i + 1 < len(words) and words[i + 1] in _MULTIPLIERS:
                value *= _WORD_NUMBERS[words[i + 1]]
                i += 1
            values.append(value)
        i += 1
    # A digit followed by a multiplier word ("5 lakh") becomes one value.
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(lakhs?|crores?|thousand|million|billion)\b", t):
        base = float(m.group(1))
        if base in values:
            values.remove(base)
        values.append(base * _WORD_NUMBERS[m.group(2)])
    return Counter(round(v, 6) for v in values)


def numbers_guard(a: str, b: str) -> bool:
    return extract_numbers(a) == extract_numbers(b)


# --- dates ------------------------------------------------------------------------

_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, "april": 4,
    "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7, "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9, "october": 10, "oct": 10, "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}  # fmt: skip
_WEEKDAYS = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}


def _full_year(y: str, base: int | None = None) -> int:
    n = int(y)
    if n < 100:
        century = (base // 100) * 100 if base else 2000
        return century + n
    return n


def extract_dates(text: str) -> set[str]:
    t = text.lower()
    out: set[str] = set()
    for m in _DATE_NUMERIC_RE.finditer(t):
        if m.group(4):
            out.add(f"date:{int(m.group(4))}-{int(m.group(5))}-{int(m.group(6))}")
        else:
            out.add(f"date:{_full_year(m.group(3))}-{int(m.group(2))}-{int(m.group(1))}")
    for m in _FY_RE.finditer(t):
        start = _full_year(m.group(1))
        out.add(f"fy:{start}-{_full_year(m.group(2), start)}" if m.group(2) else f"fy:{start}")
    t2 = _FY_RE.sub(" ", _DATE_NUMERIC_RE.sub(" ", t))
    for m in _YEAR_RANGE_RE.finditer(t2):
        start = int(m.group(1))
        out.add(f"range:{start}-{_full_year(m.group(2), start)}")
    t3 = _YEAR_RANGE_RE.sub(" ", t2)
    out.update(f"year:{y}" for y in _YEAR_RE.findall(t3))
    words = set(_WORD_RE.findall(t))
    # "may" is also a verb ("may I return"); count it as a month only next to a number.
    for w in words:
        if w in _MONTHS and w != "may":
            out.add(f"month:{_MONTHS[w]}")
        elif w in _WEEKDAYS:
            out.add(f"weekday:{w}")
    if re.search(r"\b\d{1,2}\s+may\b|\bmay\s+\d{1,4}\b", t):
        out.add("month:5")
    return out


def dates_guard(a: str, b: str) -> bool:
    return extract_dates(a) == extract_dates(b)


# --- negation ---------------------------------------------------------------------

_NEGATION_RE = re.compile(
    r"\b(?:not|no|never|cannot|without|nor|neither|none|nothing|nobody)\b|n't\b|\bcant\b|\bwont\b"
    r"|\bdont\b|\bdoesnt\b|\bisnt\b|\barent\b|\bdidnt\b"
    # v2: "Is it false that ...?" asks the opposite question (found by the v1 simulation).
    r"|\b(?:false|untrue)\s+that\b"
    # v2: opposite-prefix words carry a negation without "not" ("possible" vs "impossible",
    # "allowed" vs "prohibited"); "non-refundable" now agrees with "not refundable".
    r"|\bnon-?[a-z]{3,}\b"
    r"|\b(?:impossible|unsafe|prohibited|forbidden|banned|disallowed|ineligible|invalid"
    r"|illegal|unlawful|unavailable|unable|unauthori[sz]ed)\b",
    re.IGNORECASE,
)


def negation_count(text: str) -> int:
    return len(_NEGATION_RE.findall(text.replace("’", "'")))


def negation_guard(a: str, b: str) -> bool:
    """Polarity must match: an odd number of negations flips the meaning."""
    return negation_count(a) % 2 == negation_count(b) % 2


# --- entities / key terms ---------------------------------------------------------

_QUOTED_RE = re.compile(r"[\"'“‘]([^\"'”’]{2,40})[\"'”’]")
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9&.-]*")
# Capitalised words that are just sentence words, not names.
_COMMON_CAPS = {
    "i", "what", "how", "why", "when", "where", "which", "who", "whom", "whose", "is", "are",
    "can", "could", "do", "does", "did", "should", "would", "will", "may", "my", "the", "a",
    "an", "please", "tell", "explain", "give", "list", "hi", "hello", "hey", "in", "on", "for",
    "if", "and", "or", "to", "of", "it", "am", "was", "were", "has", "have", "any", "rs",
}  # fmt: skip


def extract_entities(text: str) -> set[str]:
    """Key terms: ALL-CAPS tokens (GST), capitalised words not at a sentence start, quoted text."""
    terms = {m.group(1).strip().lower() for m in _QUOTED_RE.finditer(text)}
    sentence_start = True
    for m in _TOKEN_RE.finditer(text):
        tok = m.group(0).rstrip(".-")
        if not tok:
            continue
        low = tok.lower()
        letters = [c for c in tok if c.isalpha()]
        is_acronym = len(letters) >= 2 and all(c.isupper() for c in letters)
        is_capitalised = tok[0].isupper() and not sentence_start
        is_camel = any(c.isupper() for c in tok[1:])  # iPhone, WhatsApp, eBay
        if (is_acronym or is_capitalised or is_camel) and low not in _COMMON_CAPS:
            terms.add(low)
        # A new sentence starts after . ? ! (inside the token, or right after it).
        nxt = text[m.end() : m.end() + 2]
        sentence_start = m.group(0).endswith(".") or bool(re.match(r"\s*[.?!]", nxt))
    return terms


def entities_guard(a: str, b: str, mode: str = "swap") -> bool:
    ea, eb = extract_entities(a), extract_entities(b)
    only_a, only_b = ea - eb, eb - ea
    if mode == "strict":
        return not only_a and not only_b
    # swap: reject only when *each* side has a term the other lacks (RBI vs SEBI),
    # so "repo rate?" vs "RBI repo rate?" (one side adds context) still passes.
    return not (only_a and only_b)


# --- relations (v2) ---------------------------------------------------------------
# Added in v2 after the v1 traffic simulation: almost all remaining false hits were
# relation words ("before 55" vs "after 55", "next" vs "last" Friday) or flipped
# comparisons ("football older than cricket" vs "cricket older than football").

_MONTH_START = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
_DAY_OR_PERIOD = (
    r"(?:week|month|year|quarter|monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
)
# axis -> (patterns meaning +1, patterns meaning -1, strict). Strict axes reject any
# difference, including a word on one side only; the others reject only a direct conflict,
# because words like "last date" or "most" also appear in harmless paraphrases.
_AXES: dict[str, tuple[list[str], list[str], bool]] = {
    "time": (
        [r"after", r"later than", r"since", r"beyond",
         rf"following(?=\s+(?:\d|the\s+\d|{_MONTH_START}))"],
        [r"before", r"prior to", r"earlier than", r"until", r"till", r"ahead of",
         r"in advance of"],
        True,
    ),
    "amount": (
        [r"more than", r"greater than", r"higher than", r"above", r"exceeding", r"exceeds",
         r"at least", r"in excess of", r"over(?=\s*(?:rs\.?|inr|₹|\$|\d))"],
        [r"less than", r"lower than", r"fewer than", r"below", r"at most", r"up to",
         r"under(?=\s*(?:rs\.?|inr|₹|\$|\d))"],
        True,
    ),
    "sequence": (
        [r"next", r"upcoming", rf"coming(?=\s+{_DAY_OR_PERIOD})"],
        [r"last", r"previous", r"preceding", rf"past(?=\s+(?:\d|{_DAY_OR_PERIOD}))"],
        False,
    ),
    "extreme": (
        [r"maximum", r"max", r"highest", r"upper limit"],
        [r"minimum", r"min", r"lowest", r"lower limit"],
        False,
    ),
    "change": (
        [r"increase[sd]?", r"raised?", r"hiked?", r"rise", r"go up"],
        [r"decrease[sd]?", r"cut", r"reduced?", r"drop", r"go down"],
        False,
    ),
}  # fmt: skip
_AXIS_RE = {
    axis: (
        re.compile(r"\b(?:" + "|".join(up) + r")\b", re.IGNORECASE),
        re.compile(r"\b(?:" + "|".join(down) + r")\b", re.IGNORECASE),
        strict,
    )
    for axis, (up, down, strict) in _AXES.items()
}

# Antonym comparatives map to one canonical word: "A younger than B" means the same as
# "B older than A", so an antonym on the other side must come with swapped sides.
_ANTONYM = {
    "younger": "older", "smaller": "bigger", "larger": "bigger", "lower": "higher",
    "slower": "faster", "costlier": "cheaper", "worse": "better", "shorter": "longer",
    "lighter": "heavier", "later": "earlier", "less": "more", "fewer": "more",
    "lesser": "greater",
}  # fmt: skip
_SWAPS_SIDES = {"younger", "smaller", "lower", "slower", "costlier", "worse", "shorter",
                "lighter", "later", "less", "fewer", "lesser"}  # fmt: skip
_WORD_RE_REL = re.compile(r"[a-z0-9]+")
_STOP = {
    "is", "are", "was", "were", "be", "the", "a", "an", "of", "on", "in", "for", "to", "do",
    "does", "did", "have", "has", "take", "takes", "it", "my", "i", "than", "and", "or",
    "what", "which", "how", "can", "will", "would", "should", "there", "at", "by", "with",
}  # fmt: skip
# Bare level words, checked only when neither question has a "than" comparison:
# "is the limit higher for emergencies?" vs "lower for emergencies?".
_LEVEL_UP = {"higher", "greater", "bigger", "larger", "longer", "faster"}
_LEVEL_DOWN = {"lower", "smaller", "shorter", "slower", "fewer", "lesser"}


@dataclass(frozen=True)
class Comparison:
    canon: str  # canonical comparative, e.g. "older", "more expensive", "more moons"
    orient: int  # -1 when the antonym form was used ("younger"), else 1
    left: frozenset[str]  # content words before the comparative
    right: frozenset[str]  # content words after "than"


def comparison(text: str) -> Comparison | None:
    """The first "A <comparative> than B" in the text, split into its two sides."""
    toks = _WORD_RE_REL.findall(text.lower())
    if "than" not in toks:
        return None
    i = toks.index("than")
    if i == 0:
        return None
    word = toks[i - 1]
    comp_start = i - 1
    if (
        word not in ("more", "less", "fewer")
        and i >= 2
        and toks[i - 2] in ("more", "less", "fewer")
    ):
        # "more expensive than", "more moons than": the next word carries the meaning
        canon = f"more {word}"
        orient = -1 if toks[i - 2] in ("less", "fewer") else 1
        comp_start = i - 2
    else:
        canon = _ANTONYM.get(word, word)
        orient = -1 if word in _SWAPS_SIDES else 1
    left = frozenset(t for t in toks[:comp_start] if t not in _STOP)
    right = frozenset(t for t in toks[i + 1 :] if t not in _STOP)
    return Comparison(canon, orient, left, right)


def comparison_conflict(a: str, b: str) -> bool:
    """True when two comparisons say opposite things (sides swapped, or antonym in place)."""
    ca, cb = comparison(a), comparison(b)
    if ca is None or cb is None or ca.canon != cb.canon:
        return False
    only_a, only_b = ca.right - cb.right, cb.right - ca.right
    swapped = bool(only_a and only_b and only_a <= cb.left and only_b <= ca.left)
    antonym = ca.orient != cb.orient
    return swapped != antonym


def relation_signature(text: str) -> dict[str, set[int]]:
    """Per axis, which directions the text uses: {+1}, {-1}, both, or absent."""
    out: dict[str, set[int]] = {}
    for axis, (up, down, _strict) in _AXIS_RE.items():
        signs = ({1} if up.search(text) else set()) | ({-1} if down.search(text) else set())
        if signs:
            out[axis] = signs
    return out


def _level(text: str) -> set[int]:
    toks = set(_WORD_RE_REL.findall(text.lower()))
    return ({1} if toks & _LEVEL_UP else set()) | ({-1} if toks & _LEVEL_DOWN else set())


def relation_guard(a: str, b: str) -> bool:
    sa, sb = relation_signature(a), relation_signature(b)
    for axis, (_up, _down, strict) in _AXIS_RE.items():
        x, y = sa.get(axis, set()), sb.get(axis, set())
        if strict and x != y:
            return False
        if not strict and x and y and x != y:
            return False
    if comparison_conflict(a, b):
        return False
    if comparison(a) is None and comparison(b) is None:
        la, lb = _level(a), _level(b)
        if la and lb and la != lb:
            return False
    return True


# --- registry ---------------------------------------------------------------------

GUARDS: dict[str, Callable[[str, str], bool]] = {
    "numbers": numbers_guard,
    "dates": dates_guard,
    "negation": negation_guard,
    "entities": entities_guard,
    "relation": relation_guard,
}


def run_guards(
    a: str, b: str, enabled: tuple[str, ...] | list[str], entity_mode: str = "swap"
) -> str | None:
    """Run the enabled guards in order; return the name of the first that rejects, or None."""
    for name in enabled:
        if name == "entities":
            ok = entities_guard(a, b, mode=entity_mode)
        else:
            ok = GUARDS[name](a, b)
        if not ok:
            return name
    return None
