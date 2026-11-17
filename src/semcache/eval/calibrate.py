"""Measure the real upstream once, so the simulation's mock behaves like it (SPEC 12.4).

At most 50 real Nova 2 Lite calls on questions sampled from the traffic intents. The
samples (latency, input and output tokens) are saved; the mock upstream then draws from
them. Embedding latency comes from the embedding memo, which recorded every real call.
"""

from __future__ import annotations

import json
import random
import sqlite3
from pathlib import Path

import numpy as np

SIM_SYSTEM = (
    "You are the help assistant of an Indian fintech and e-commerce app. Answer the user's "
    "question clearly in under 120 words."
)
MAX_CALLS = 50


def load_intents(path: str | Path = "data/traffic/intents.jsonl") -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def calibrate(upstream, model_id: str, n: int = 40, seed: int = 11) -> dict:
    from semcache.types import ChatRequest

    if n > MAX_CALLS:
        raise ValueError(f"at most {MAX_CALLS} live calibration calls")
    rng = random.Random(seed)
    intents = load_intents()
    questions = [rng.choice(i["phrasings"]) for i in rng.sample(intents, n)]
    samples = []
    for q in questions:
        req = ChatRequest(
            messages=[{"role": "user", "content": q}], system=SIM_SYSTEM, max_tokens=400
        )
        r = upstream.complete(req, model_id)
        samples.append(
            {
                "latency_ms": round(r.latency_ms, 1),
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "stop_reason": r.stop_reason,
            }
        )
    return {"model_id": model_id, "system_prompt": SIM_SYSTEM, "samples": samples}


def summarise(cal: dict, memo_path: str | Path = ".embed_cache/embeddings.sqlite") -> dict:
    s = cal["samples"]
    lat = np.array([x["latency_ms"] for x in s])
    out = {
        "calls": len(s),
        "upstream_latency_ms_p50": round(float(np.percentile(lat, 50)), 1),
        "upstream_latency_ms_p95": round(float(np.percentile(lat, 95)), 1),
        "input_tokens_mean": round(float(np.mean([x["input_tokens"] for x in s])), 1),
        "output_tokens_mean": round(float(np.mean([x["output_tokens"] for x in s])), 1),
    }
    if Path(memo_path).exists():
        db = sqlite3.connect(str(memo_path))
        emb = np.array([r[0] for r in db.execute("SELECT latency_ms FROM memo").fetchall()])
        db.close()
        out["embed_calls_in_memo"] = int(len(emb))
        out["embed_latency_ms_p50"] = round(float(np.percentile(emb, 50)), 1)
        out["embed_latency_ms_p95"] = round(float(np.percentile(emb, 95)), 1)
    return out
