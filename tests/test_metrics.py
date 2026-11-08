from rag_engine.evaluation.metrics import mrr, ndcg_at_k, recall_at_k


def test_recall_perfect():
    assert recall_at_k(["a", "b", "c"], {"a", "b"}, 3) == 1.0


def test_recall_partial():
    assert recall_at_k(["a", "x", "y"], {"a", "b"}, 3) == 0.5


def test_recall_empty_relevant():
    assert recall_at_k(["a"], set(), 3) == 0.0


def test_mrr_first():
    assert mrr(["a", "b"], {"a"}) == 1.0


def test_mrr_second():
    assert mrr(["x", "a"], {"a"}) == 0.5


def test_mrr_miss():
    assert mrr(["x", "y"], {"a"}) == 0.0


def test_ndcg_bounds():
    assert 0.0 <= ndcg_at_k(["a", "b", "x"], {"a", "b"}, 3) <= 1.0


def test_ndcg_perfect():
    assert ndcg_at_k(["a", "b"], {"a", "b"}, 2) == 1.0
