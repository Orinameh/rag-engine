"""Retrieval-quality metrics, one function per question a RAG system must answer.

All three take the same three arguments:

- ``retrieved``  — ranked list of chunk/doc identifiers, best first, exactly as
  the search returned them (order matters; these are list metrics).
- ``relevant``   — the ground truth: identifiers a grader says SHOULD have been
  found for this question.
- ``k`` (where present) — how deep of the ranked list to trust.

Returns are floats in ``[0.0, 1.0]``; higher is better. ``1.0`` does NOT mean
"perfect system", it means "from this metric's point of view, nothing was
missed". The three metrics disagree on purpose — see the docstrings.

Tests in ``tests/test_metrics.py`` document the edge cases; this module has no
I/O and is safe to run in CI.
"""

import math


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Of the relevant items, how many did we get back within the first k?

    recall@k = |retrieved ∩ relevant (top k)| / |relevant|

    Plain-English: "the answer WAS in my library — did I pull it into the
    context window?" For a RAG answer to cite a chunk, that chunk must reach
    the LLM, so recall@k is the metric that says whether retrieval ever gave
    the model the source it needed. It ignores position entirely: a hit at
    rank 1 counts the same as one at rank 40.

    Interpretation:
    - 1.0  — every relevant item reached the reader (position may be bad, but
      nothing was missed).
    - 0.5  — half of what the grader expects was found.
    - 0.0  — the window missed everything relevant; the LLM cannot possibly
      have answered from ground truth (it either refuses or hallucinates).

    Edge cases: an empty ``relevant`` set is unscoreable → 0.0; ``k`` beyond
    the end of ``retrieved`` just caps the intersection (no error).

    Example: relevant = {a, b, c}, retrieved[:5] = [b, x, a] → 2 / 3 ≈ 0.667.
    """
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    """How high up does the FIRST relevant hit land? (1/rank of the first hit)

    MRR = 1 / rank of the first retrieved item that is in ``relevant``.

    Plain-English: "when I found something true, how quickly did the user see
    it?" Users skim the top of result lists, so quality concentrates on early
    ranks regardless of ``k`` — this metric is position-sensitive where
    recall@k is not.

    Interpretation (the reciprocal of "first useful result's rank"):
    - 1.0  — the very first result is relevant (rank 1).
    - 0.5  — it took two results to say something true (rank 2).
    - 0.33 — the user waded through two wrong hits first (rank 3).
    - 0.0  — never hit anything relevant, no matter how deep it lurked.

    Note it is deliberately blind to *everything after* the first hit: a
    perfect rank-1 hit followed by garbage scores the same as a perfect
    rank-1 hit followed by more truth. Recall covers "staying true"; MRR
    covers "leading with truth".

    Edge cases: an empty ``relevant`` set is unscoreable → 0.0; ranks are
    1-based (rank 1 = first element), so the returned value is never smaller
    than 1/len(retrieved) when there is any hit.

    Example: relevant = {b}, retrieved = [x, b] → 1 / 2 = 0.5.
    """
    for rank, doc_id in enumerate(retrieved, 1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Quality of the WHOLE ranked ordering, discounted by position. [0, 1]

    NDCG@k = DCG@k / IDCG@k, with  DCG@k = Σ over hits of 1 / log2(rank+1).

    Plain-English: "how closely does my ordering match the ideal ordering?"
    Every hit contributes value, but each deeper hit de-values itself by a
    log discount: useful at rank 1 is worth ~1.0, at rank 2 ~0.63, at rank 10
    ~0.29, at rank 40 ~0.14. That logarithm is the whole point — it encodes
    "truth at the top is worth much more than truth buried below", but
    unlike MRR it keeps crediting *all* the hits after the first.

    Why the denominator (IDCG): DCG's raw scale grows with the number of hits,
    so a list containing every relevant item twice over isn't "better". IDCG
    is the discount-weighted score of the *ideal* list — all ``min(len(
    relevant), k)`` relevant items packed at the top. The ratio therefore
    normalizes to [0, 1]:
    - 1.0  — the list equals the ideal ordering (all relevant items, ranked
      highest-first).
    - 0.66 — the useful material exists but is diluted/pushed down: truth
      spread lower in the list, or fewer relevant items within k than exist.

    Interpretation with the others: recall is "did we return truth at all",
    MRR is "did we lead with it", NDCG is "is everything arranged well".
    For RAG NDCG@k is the ranking-quality gate (see README's CI section).

    Edge cases: an empty ``relevant`` set → 0.0 (IDCG would be 0; div-zero
    guard); zero or negative ``k`` → 0.0; binary relevance only — items in
    ``relevant`` count as gain 1, everything else 0 (graded relevance is a
    future extension; the shape of the code doesn't change, only the gain
    function).

    Example: relevant = {b}, retrieved[:2] = [b, x] → DCG = 1/log2(3) ≈ 0.63,
    IDCG = 1/log2(2) = 1.0 → NDCG ≈ 0.63. Same doc at rank 1 → 1.0.
    """
    dcg = sum(
        1.0 / math.log2(i + 2) for i, d in enumerate(retrieved[:k]) if d in relevant
    )
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg else 0.0
