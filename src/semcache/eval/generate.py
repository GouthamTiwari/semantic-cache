"""Grow the evaluation data with Nova 2 Lite (SPEC 12.1 and 12.4).

Two generators:
- `generate_pairs`: per topic, 15 labelled pairs across the five categories.
- `generate_intents`: per topic, 2 families of 3 near-miss intents with 5 phrasings each,
  for the traffic simulation.

The raw model output is kept next to the parsed data, so every pair can be traced back.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from semcache.eval.dataset import Pair, load_pairs, pair_signature, save_pairs, split_for_group
from semcache.normalize import normalize

TOPICS: list[tuple[str, str]] = [
    # finance and regulation (India)
    ("finance", "RBI repo rate and monetary policy"),
    ("finance", "filing GST returns (GSTR-1, GSTR-3B)"),
    ("finance", "GST rates on goods and services"),
    ("finance", "GST input tax credit"),
    ("finance", "TDS on salary, rent and interest"),
    ("finance", "income tax return (ITR) filing"),
    ("finance", "PAN and Aadhaar linking"),
    ("finance", "bank account KYC"),
    ("finance", "UPI payments and transaction limits"),
    ("finance", "NEFT, RTGS and IMPS transfers"),
    ("finance", "home loans and EMIs"),
    ("finance", "credit card fees and rewards"),
    ("finance", "fixed deposits and interest rates"),
    ("finance", "mutual fund SIPs"),
    ("finance", "EPF provident fund withdrawals"),
    ("finance", "digital lending apps and RBI rules"),
    # customer support
    ("support", "password reset and account login"),
    ("support", "tracking an online order"),
    ("support", "returns and refunds for online orders"),
    ("support", "shipping times and delivery charges"),
    ("support", "cancelling a subscription"),
    ("support", "deleting an account and personal data"),
    ("support", "downloading invoices and billing history"),
    ("support", "product warranty and repairs"),
    ("support", "failed or double-charged payments"),
    ("support", "changing a delivery address"),
    ("support", "coupon and discount codes"),
    ("support", "mobile app problems on Android and iPhone"),
    ("support", "loyalty points and rewards programme"),
    ("support", "customer support contact options and hours"),
    # general knowledge
    ("general", "countries and their capitals"),
    ("general", "units and conversions"),
    ("general", "basic physics and chemistry facts"),
    ("general", "world history events and dates"),
    ("general", "Python programming basics"),
    ("general", "cooking measurements and recipes"),
    ("general", "passports and travel visas"),
    ("general", "nutrition basics"),
    ("general", "rules of cricket and football"),
    ("general", "planets and space exploration"),
]

PAIR_KEYS = {
    "paraphrase_same": ("paraphrase", "same", 3),
    "number_different": ("near_miss_number", "different", 2),
    "number_same": ("near_miss_number", "same", 1),
    "negation_different": ("negation", "different", 2),
    "negation_same": ("negation", "same", 1),
    "entity_different": ("entity_swap", "different", 2),
    "entity_same": ("entity_swap", "same", 1),
    "topic_different": ("same_topic_different_question", "different", 2),
    "topic_same": ("same_topic_different_question", "same", 1),
}

SYSTEM = "You write evaluation data for a semantic cache. Reply with one JSON object only."

PAIRS_PROMPT = """Topic: {topic} (domain: {domain})

Write short questions (4 to 20 words) about this topic, the way real users type them.
For Indian finance topics use Indian context (Rs, lakh, GST, RBI) where natural.
Vary the wording; do not start every question the same way.

Return a JSON object. Each key holds a list of pairs, each pair a 2-item list [a, b]:
- "paraphrase_same": 3 pairs. b asks exactly the same thing as a in different words. A cached answer to a is fully correct for b.
- "number_different": 2 pairs. Same wording except one number, amount, percentage, duration or year, so the correct answer changes.
- "number_same": 1 pair. The same question with the number written differently (for example "7 days" vs "seven days", "Rs 1 lakh" vs "Rs 100000"), so the answer does not change.
- "negation_different": 2 pairs. b negates a (adds or removes not / cannot / without / never), so the answer changes.
- "negation_same": 1 pair. Both questions carry the same negative meaning in different words (for example "not refundable" vs "can't be refunded").
- "entity_different": 2 pairs. Same wording but one entity, product, place, organisation or item swapped, so the answer changes.
- "entity_same": 1 pair. The same entity named differently (for example full name vs abbreviation), so the answer does not change.
- "topic_different": 2 pairs. Same topic, but a clearly different question that needs a different answer.
- "topic_same": 1 pair. The same question asked with very different words and sentence structure.

Examples of the style:
{examples}"""

INTENTS_PROMPT = """Topic: {topic} (domain: {domain})

Create 2 families of user questions about this topic. Each family has 3 intents that look
very similar but need different answers: they differ by a number, a date, a negation, or a
swapped entity. For each intent give 5 phrasings a real user might type (4 to 20 words):
the first is the plain question, the other 4 are paraphrases that need exactly the same
answer. Paraphrases must keep the detail that separates the intent from its siblings.

Return JSON: {{"families": [{{"intents": [{{"phrasings": ["...", "...", "...", "...", "..."]}}, ...]}}, ...]}}"""


def _extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("no JSON object in model output")
    return json.loads(m.group(0))


def _examples() -> str:
    seeds = load_pairs("data/pairs/seed.jsonl")
    picked = [s for s in seeds if s.id in {"seed001", "seed011", "seed016", "seed021", "seed025"}]
    return "\n".join(f'- {p.category} / {p.label}: ["{p.a}", "{p.b}"]' for p in picked)


def _call(client, model_id: str, prompt: str, budget, max_tokens: int = 2500) -> tuple[str, float]:
    from semcache.bedrock import with_retries
    from semcache.pricing import cost_usd

    if budget:
        budget.check(cost_usd(model_id, len(prompt) // 3, max_tokens))
    resp = with_retries(
        lambda: client.converse(
            modelId=model_id,
            system=[{"text": SYSTEM}],
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": max_tokens, "temperature": 0.7},
        )
    )
    text = "".join(
        b["text"] for b in resp["output"]["message"]["content"] if "text" in b
    )  # fmt: skip
    usage = resp.get("usage", {})
    cost = cost_usd(model_id, usage.get("inputTokens", 0), usage.get("outputTokens", 0))
    if budget:
        budget.add(cost)
    return text, cost


def parse_pairs(raw: dict, group: str, domain: str) -> list[Pair]:
    out: list[Pair] = []
    for key, (category, label, _n) in PAIR_KEYS.items():
        for item in raw.get(key, []) or []:
            if not (isinstance(item, list) and len(item) == 2):
                continue
            a, b = (str(x).strip() for x in item)
            if not (3 <= len(a) <= 300 and 3 <= len(b) <= 300) or normalize(a) == normalize(b):
                continue
            out.append(
                Pair(
                    id="tmp",
                    a=a,
                    b=b,
                    label=label,
                    category=category,
                    split="dev",
                    group=group,
                    source="generated",
                    domain=domain,
                )  # fmt: skip
            )
    return out


def generate_pairs(client, model_id: str, budget, raw_path: Path, workers: int = 3) -> dict:
    """Generate pairs for every topic, merge with the seeds, split by group, save."""
    examples = _examples()
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    def one(i_topic):
        i, (domain, topic) = i_topic
        prompt = PAIRS_PROMPT.format(topic=topic, domain=domain, examples=examples)
        for _attempt in range(2):
            text, cost = _call(client, model_id, prompt, budget)
            try:
                return i, domain, topic, _extract_json(text), text, cost
            except (ValueError, json.JSONDecodeError):
                continue
        return i, domain, topic, {}, text, cost

    with ThreadPoolExecutor(workers) as pool:
        results = sorted(pool.map(one, enumerate(TOPICS)))
    total_cost = sum(r[5] for r in results)
    with open(raw_path, "w", encoding="utf-8", newline="\n") as f:
        for i, _domain, topic, _raw, text, _cost in results:
            f.write(json.dumps({"group": f"t{i:02d}", "topic": topic, "output": text}) + "\n")

    seeds = load_pairs("data/pairs/seed.jsonl")
    for s in seeds:
        s.group, s.source, s.domain, s.split = "seed", "seed", "mixed", "dev"
    pairs: list[Pair] = list(seeds)
    seen = {pair_signature(p.a, p.b) for p in pairs}
    for i, domain, _topic, raw, _text, _cost in results:
        for p in parse_pairs(raw, f"t{i:02d}", domain):
            sig = pair_signature(p.a, p.b)
            if sig not in seen:
                seen.add(sig)
                pairs.append(p)
    finalize(pairs)
    save_pairs(pairs)
    return {"pairs": len(pairs), "cost_usd": round(total_cost, 6), "topics": len(results)}


def finalize(pairs: list[Pair]) -> None:
    """Assign splits by group, drop test pairs whose text also appears in dev, renumber ids."""
    groups = [p.group for p in pairs]
    for p in pairs:
        p.split = split_for_group(p.group, groups)  # type: ignore[assignment]
    dev_texts = {normalize(t) for p in pairs if p.split == "dev" for t in (p.a, p.b)}
    pairs[:] = [
        p for p in pairs if p.split == "dev" or not ({normalize(p.a), normalize(p.b)} & dev_texts)
    ]
    for n, p in enumerate(pairs, 1):
        if p.source != "seed":
            p.id = f"p{n:04d}"


def generate_intents(client, model_id: str, budget, out_path: Path, workers: int = 3) -> dict:
    """Intent families for the traffic simulation (SPEC 12.4)."""

    def one(i_topic):
        i, (domain, topic) = i_topic
        prompt = INTENTS_PROMPT.format(topic=topic, domain=domain)
        for _attempt in range(2):
            text, cost = _call(client, model_id, prompt, budget)
            try:
                return i, domain, topic, _extract_json(text), cost
            except (ValueError, json.JSONDecodeError):
                continue
        return i, domain, topic, {}, cost

    with ThreadPoolExecutor(workers) as pool:
        results = sorted(pool.map(one, enumerate(TOPICS)))
    rows = []
    seen: set[str] = set()
    for i, domain, topic, raw, _cost in results:
        for f_idx, fam in enumerate(raw.get("families", []) or []):
            for k_idx, intent in enumerate(fam.get("intents", []) or []):
                phr = [str(x).strip() for x in intent.get("phrasings", []) if str(x).strip()]
                uniq = []
                for p in phr:
                    if normalize(p) not in seen:
                        seen.add(normalize(p))
                        uniq.append(p)
                if len(uniq) >= 3:
                    rows.append(
                        {
                            "intent_id": f"t{i:02d}f{f_idx}i{k_idx}",
                            "family": f"t{i:02d}f{f_idx}",
                            "domain": domain,
                            "topic": topic,
                            "phrasings": uniq[:5],
                        }
                    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {
        "intents": len(rows),
        "texts": sum(len(r["phrasings"]) for r in rows),
        "cost_usd": round(sum(r[4] for r in results), 6),
    }


# --- v2 ---------------------------------------------------------------------------
# v2 adds a "relation" category and a brand-new test set on topics v1 never used. The v1
# test pairs were seen once, so they are retired rather than reused.

TOPICS_V2_TEST: list[tuple[str, str]] = [
    ("finance", "credit scores and CIBIL reports"),
    ("finance", "life and health insurance claims"),
    ("finance", "gold loans"),
    ("finance", "senior citizen savings schemes"),
    ("finance", "NPS pension accounts"),
    ("finance", "sending money abroad under LRS"),
    ("support", "food delivery orders"),
    ("support", "train and flight ticket bookings and cancellations"),
    ("support", "mobile recharges and data plans"),
    ("support", "online course enrolment and certificates"),
    ("support", "hotel bookings and check-in"),
    ("support", "ride-hailing trips and fares"),
    ("general", "rivers, mountains and geography"),
    ("general", "the human body"),
    ("general", "weather and climate"),
    ("general", "famous books and authors"),
]

RELATION_KEYS = {
    "relation_different": ("relation", "different", 2),
    "relation_same": ("relation", "same", 1),
}

RELATION_RULES = """- "relation_different": 2 pairs. Same wording except one relation flips the meaning: before vs after, more than vs less than, above vs below, next vs last, maximum vs minimum, increase vs decrease, or the two sides of a comparison swapped ("Is A older than B?" vs "Is B older than A?"). The correct answer changes.
- "relation_same": 1 pair. The same relation said in different words (for example "before" vs "prior to"), so the answer does not change."""

RELATION_PROMPT = (
    """Topic: {topic} (domain: {domain})

Write short questions (4 to 20 words) about this topic, the way real users type them.
Return a JSON object. Each key holds a list of pairs, each pair a 2-item list [a, b]:
"""
    + RELATION_RULES
)

PAIRS_PROMPT_V2 = PAIRS_PROMPT.replace(
    "\n\nExamples of the style:", "\n" + RELATION_RULES + "\n\nExamples of the style:"
)


def _parse_v2(raw: dict, group: str, domain: str, split: str, keys: dict) -> list[Pair]:
    out: list[Pair] = []
    for key, (category, label, _n) in keys.items():
        for item in raw.get(key, []) or []:
            if not (isinstance(item, list) and len(item) == 2):
                continue
            a, b = (str(x).strip() for x in item)
            if not (3 <= len(a) <= 300 and 3 <= len(b) <= 300) or normalize(a) == normalize(b):
                continue
            out.append(
                Pair(
                    id="tmp",
                    a=a,
                    b=b,
                    label=label,
                    category=category,
                    split=split,
                    group=group,
                    source="generated-v2",
                    domain=domain,
                )  # fmt: skip
            )
    return out


def generate_v2(client, model_id: str, budget, raw_path: Path, workers: int = 3) -> dict:
    """Retire the v1 test split, add relation pairs to dev, build a fresh test split."""
    pairs = load_pairs()
    for p in pairs:
        if p.split == "test":
            p.split = "retired"  # type: ignore[assignment]
    dev_groups = sorted({p.group for p in pairs if p.split == "dev" and p.group != "seed"})
    examples = _examples()
    jobs = []
    for g in dev_groups:
        domain, topic = TOPICS[int(g[1:])]
        jobs.append((g, domain, "dev", RELATION_PROMPT.format(topic=topic, domain=domain),
                     RELATION_KEYS, topic))  # fmt: skip
    for i, (domain, topic) in enumerate(TOPICS_V2_TEST):
        prompt = PAIRS_PROMPT_V2.format(topic=topic, domain=domain, examples=examples)
        jobs.append((f"n{i:02d}", domain, "test", prompt, {**PAIR_KEYS, **RELATION_KEYS}, topic))

    def one(job):
        group, domain, split, prompt, keys, topic = job
        text, cost = "", 0.0
        for _attempt in range(2):
            text, c = _call(client, model_id, prompt, budget)
            cost += c
            try:
                return job, _extract_json(text), text, cost
            except (ValueError, json.JSONDecodeError):
                continue
        return job, {}, text, cost

    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(one, jobs))
    with open(raw_path, "w", encoding="utf-8", newline="\n") as f:
        for (group, _d, split, _p, _k, topic), _raw, text, _c in results:
            f.write(json.dumps({"group": group, "split": split, "topic": topic,
                                "output": text}) + "\n")  # fmt: skip

    seen = {pair_signature(p.a, p.b) for p in pairs}
    used = {normalize(t) for p in pairs for t in (p.a, p.b)}  # dev + retired texts
    next_id = max(int(p.id[1:]) for p in pairs if p.id.startswith("p")) + 1
    added = {"dev": 0, "test": 0}
    dropped_overlap = 0
    for (group, domain, split, _p, keys, _t), raw, _text, _c in results:
        for p in _parse_v2(raw, group, domain, split, keys):
            sig = pair_signature(p.a, p.b)
            if sig in seen:
                continue
            if split == "test" and ({normalize(p.a), normalize(p.b)} & used):
                dropped_overlap += 1  # a test question must be unseen
                continue
            seen.add(sig)
            p.id = f"p{next_id:04d}"
            next_id += 1
            pairs.append(p)
            added[split] += 1
            if split == "dev":
                used.update({normalize(p.a), normalize(p.b)})
    save_pairs(pairs)
    return {
        "added": added,
        "dropped_test_overlap": dropped_overlap,
        "calls": len(results),
        "cost_usd": round(sum(r[3] for r in results), 6),
    }
