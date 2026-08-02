"""Tests for cost governance, semantic cache, and parent-child expansion."""


# --------------------------------------------------------------------------- #
# Cost tracker
# --------------------------------------------------------------------------- #
class _CostOn:
    class cost:
        enabled = True
        daily_token_budget = 100
        redis_url = ""
    class api:
        redis_url = ""


class _CostOff:
    class cost:
        enabled = False


def test_count_tokens_nonzero():
    from services.cost_tracker import count_tokens
    assert count_tokens("hello world this is a test") > 0
    assert count_tokens("") == 0


def test_budget_enforced(monkeypatch):
    import services.cost_tracker as ct
    ct._mem_usage.clear()
    cfg = _CostOn()
    # Under budget initially.
    allowed, info = ct.check_budget(cfg, user_id="u1")
    assert allowed is True
    # Consume over the budget.
    ct.add_usage(cfg, 150, user_id="u1")
    allowed, info = ct.check_budget(cfg, user_id="u1")
    assert allowed is False
    assert info["used"] >= 100


def test_cost_disabled_is_noop():
    from services.cost_tracker import check_budget, record_interaction
    allowed, info = check_budget(_CostOff(), user_id="x")
    assert allowed is True and info == {}
    assert record_interaction(_CostOff(), "a", "b", user_id="x") == 0


# --------------------------------------------------------------------------- #
# Semantic cache
# --------------------------------------------------------------------------- #
class _FakeEmb:
    """Deterministic tiny embedding: bag-of-first-letters vector."""
    def embed_query(self, text):
        v = [0.0] * 26
        for ch in text.lower():
            if "a" <= ch <= "z":
                v[ord(ch) - 97] += 1.0
        return v


class _SemOn:
    class semantic_cache:
        enabled = True
        threshold = 0.98
        max_entries = 50
    class api:
        redis_url = ""
    class rag:
        embedding_model = _FakeEmb()


class _SemOff:
    class semantic_cache:
        enabled = False


def test_semantic_cache_hit_and_miss(monkeypatch):
    import services.semantic_cache as sc
    sc.reset_memory()
    sc._embedder = None
    sc._embedder_tried = False
    cfg = _SemOn()

    assert sc.semantic_get(cfg, "headache treatment") is None  # empty -> miss
    sc.semantic_set(cfg, "headache treatment", {"response": "rest and fluids"})
    # Same text -> identical vector -> similarity 1.0 -> hit.
    hit = sc.semantic_get(cfg, "headache treatment")
    assert hit and hit["response"] == "rest and fluids"
    # Very different text -> miss.
    assert sc.semantic_get(cfg, "xyzzy quux") is None


def test_semantic_cache_disabled_is_noop():
    import services.semantic_cache as sc
    assert sc.semantic_get(_SemOff(), "anything") is None


# --------------------------------------------------------------------------- #
# Parent-child expansion
# --------------------------------------------------------------------------- #
class _PCOn:
    class rag:
        parent_child_enabled = True
        parent_chunk_size = 4096


class _PCOff:
    class rag:
        parent_child_enabled = False


def test_parent_child_merges_same_source():
    from agents.rag_agent.parent_child import expand_to_parents
    docs = [
        {"content": "chunk A1", "source": "doc1.pdf", "combined_score": 0.9},
        {"content": "chunk A2", "source": "doc1.pdf", "combined_score": 0.7},
        {"content": "chunk B1", "source": "doc2.pdf", "combined_score": 0.6},
    ]
    out = expand_to_parents(_PCOn(), docs)
    assert len(out) == 2  # doc1 merged into one, doc2 separate
    doc1 = next(d for d in out if d["source"] == "doc1.pdf")
    assert "chunk A1" in doc1["content"] and "chunk A2" in doc1["content"]


def test_parent_child_disabled_is_noop():
    from agents.rag_agent.parent_child import expand_to_parents
    docs = [{"content": "x", "source": "d"}]
    assert expand_to_parents(_PCOff(), docs) == docs
