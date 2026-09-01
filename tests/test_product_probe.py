"""
Tests for detect_product_dynamic — the function behind every false-product
bug found in production ("other", "pricing", "delayed", "farmers", and from
the app's own suggested prompts, "provided" and "past").

It previously confirmed a candidate as a product if the string appeared
ANYWHERE inside the concatenated free text of 50 retrieved records. In a
corpus made entirely of grower feedback that is true of almost any common
word. It now corroborates against the curated `products` metadata tag with
a word-boundary match and a minimum hit count.
"""
import vog_core as vc


class _ProbeIndex:
    """Returns the same match set for any probe vector. `products` holds the
    comma-joined tag written at ingestion; `value` is the raw feedback text
    (which the new implementation must ignore)."""

    def __init__(self, tagged_products=(), free_text=""):
        self._matches = [
            {"metadata": {"products": p, "value": free_text}} for p in tagged_products
        ]

    def query(self, vector, top_k=50, include_metadata=True, filter=None):
        return {"matches": self._matches}


class _CountingPinecone:
    """Counts embed calls so the batching guarantee can be asserted."""

    class _Inf:
        def __init__(self, outer):
            self.outer = outer

        def embed(self, model, inputs, parameters):
            self.outer.embed_calls += 1
            self.outer.embedded_inputs.extend(inputs)
            dim = parameters.get("dimension", vc.EMBEDDING_DIMENSION)
            return [type("V", (), {"values": [0.0] * dim})() for _ in inputs]

    def __init__(self):
        self.embed_calls = 0
        self.embedded_inputs = []
        self.inference = self._Inf(self)


# ── The real-world false positives ──

def test_common_word_in_free_text_is_not_confirmed_as_a_product():
    # "provided" appears in the feedback text but is NOT a tagged product.
    index = _ProbeIndex(
        tagged_products=["Isabion", "Axial"],
        free_text="growers provided detailed notes about pricing and delayed delivery",
    )
    assert vc.detect_product_dynamic("provided rice cultivation", index, _CountingPinecone()) is None


def test_shipped_suggested_prompt_no_longer_yields_a_junk_product():
    index = _ProbeIndex(
        tagged_products=["Isabion"],
        free_text="what growers provided for rice cultivation in the past three years",
    )
    for q in [
        "what suggestions have growers provided for rice cultivation?",
        "show the monthly complaint trend for the past three years.",
        "what misconceptions do growers commonly have about our products?",
    ]:
        assert vc.detect_product_dynamic(q, index, _CountingPinecone()) is None, q


def test_genuine_tagged_product_is_still_confirmed():
    index = _ProbeIndex(tagged_products=["Biologicals", "Biologicals", "Isabion"])
    assert vc.detect_product_dynamic("tell me about biologicals", index, _CountingPinecone()) == "biologicals"


def test_single_tag_hit_is_not_enough():
    # One incidental hit shouldn't promote a word to a product.
    index = _ProbeIndex(tagged_products=["Biologicals"])
    assert vc.detect_product_dynamic("tell me about biologicals", index, _CountingPinecone()) is None


def test_tag_match_is_word_bounded_not_substring():
    # "gram" must not match the tag "Programme".
    index = _ProbeIndex(tagged_products=["Programme", "Programme", "Programme"])
    assert vc.detect_product_dynamic("what about gram", index, _CountingPinecone()) is None


# ── Cost / latency bounds ──

def test_embeddings_are_batched_into_a_single_call():
    pc = _CountingPinecone()
    index = _ProbeIndex(tagged_products=["Isabion"])
    vc.detect_product_dynamic("alpha bravo charlie delta echo", index, pc)
    assert pc.embed_calls == 1, "candidate embeddings must be batched, not one call per word"


def test_candidate_count_is_capped():
    pc = _CountingPinecone()
    index = _ProbeIndex(tagged_products=["Isabion"])
    vc.detect_product_dynamic("alpha bravo charlie delta echo foxtrot golf", index, pc)
    assert len(pc.embedded_inputs) <= vc.MAX_DYNAMIC_CANDIDATES


def test_long_query_skips_the_probe_entirely():
    pc = _CountingPinecone()
    index = _ProbeIndex(tagged_products=["Isabion"])
    long_q = " ".join(f"word{i}" for i in range(60))
    assert vc.detect_product_dynamic(long_q, index, pc) is None
    assert pc.embed_calls == 0, "a 60-word query must not trigger any embedding calls"


# ── Candidate ordering ──

def test_capitalized_original_token_is_probed_first():
    # "cultivation" is longer than "isabion", so length alone would rank it
    # first; being capitalized in the original query must outrank length.
    pc = _CountingPinecone()
    index = _ProbeIndex(tagged_products=["Isabion"])
    vc.detect_product_dynamic(
        "isabion cultivation notes",
        index, pc, original_query="Isabion cultivation notes",
    )
    joined = " ".join(pc.embedded_inputs)
    assert "isabion" in joined and "cultivation" in joined, "both should be candidates"
    assert joined.index("isabion") < joined.index("cultivation"), \
        "a capitalized proper noun must be probed before a longer lowercase word"


def test_empty_and_stopword_only_queries_are_safe():
    pc = _CountingPinecone()
    index = _ProbeIndex(tagged_products=["Isabion"])
    assert vc.detect_product_dynamic("", index, pc) is None
    assert vc.detect_product_dynamic("what are the products", index, pc) is None
    assert pc.embed_calls == 0
