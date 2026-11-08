#!/usr/bin/env python3
"""Retrieval-quality gate: python scripts/eval.py [dataset] [k]

Needs models + seeded corpus (scripts/seed_eval.sh). Scores the eval set,
prints per-case diagnostics, and exits non-zero when any threshold fails —
the same triple CI would gate on (recall@10 ≥ 0.85, mrr ≥ 0.65,
ndcg@10 ≥ 0.70).
"""

import asyncio
import json
import sys

from rag_engine.db import SessionFactory
from rag_engine.evaluation.runner import run_eval

DATASET = sys.argv[1] if len(sys.argv) > 1 else "tests/eval_set.json"
K = int(sys.argv[2]) if len(sys.argv) > 2 else 10
THRESHOLDS = {"mrr": 0.65, f"recall@{K}": 0.85, f"ndcg@{K}": 0.70}


async def main() -> int:
    async with SessionFactory() as session:
        results = await run_eval(session, DATASET, k=K)

    summary = {key: round(value, 4) for key, value in results.items() if key != "per_case"}
    print(json.dumps(summary, indent=2))
    print("\nper case:")
    for row in results["per_case"]:
        flag = "OK " if row["mrr"] > 0 else "MISS"
        print(f'  {flag} recall={row["recall"]:.2f} mrr={row["mrr"]:.2f} '
              f'ndcg={row["ndcg"]:.2f}  {row["question"][:60]}')

    failures = [
        f"{name}: {results[name]:.4f} < {low}"
        for name, low in THRESHOLDS.items()
        if results.get(name, 0.0) < low
    ]
    if failures:
        print("\nGATE FAILED — below threshold:")
        print("\n".join(f"  {f}" for f in failures))
        return 1
    print("\nGATE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
