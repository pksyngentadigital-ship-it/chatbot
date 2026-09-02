"""Shared fixtures: a fake Pinecone and a fake Groq that behave enough like
the real things (metadata filtering, zero-vector aggregation queries) to
exercise the whole pipeline with no network and no API keys.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vog import compose, llm, retrieval  # noqa: E402
from vog.catalog import EMBEDDING_DIMENSION  # noqa: E402
from vog.parsing import CAPABILITY_REPLY, CORRECTION_ACK_REPLY  # noqa: E402
from vog.plan import MODE_REPLY, build_plan  # noqa: E402


def _matches_filter(metadata: dict, conditions: dict | None) -> bool:
    if not conditions:
        return True
    for field, cond in conditions.items():
        if "$eq" in cond and str(metadata.get(field, "")) != str(cond["$eq"]):
            return False
        if "$in" in cond and str(metadata.get(field, "")) not in [str(v) for v in cond["$in"]]:
            return False
    return True


class FakeEmbedItem:
    def __init__(self, values):
        self.values = values


class FakeInference:
    """Returns a zero vector of the right dimension. Fine, because FakeIndex
    answers purely from the metadata filter — which is how the real code
    resolves everything these tests care about getting exactly right."""

    def embed(self, model, inputs, parameters):
        dim = parameters.get("dimension", EMBEDDING_DIMENSION)
        return [FakeEmbedItem([0.0] * dim) for _ in inputs]


class FakeIndex:
    def __init__(self, records):
        self.records = records          # [{"metadata": {...}}, ...]
        self.deleted_all = False

    def query(self, vector, top_k=10, include_metadata=True, filter=None):
        matches = [r for r in self.records if _matches_filter(r["metadata"], filter)]
        return {"matches": matches[:top_k]}

    def fetch(self, ids):
        # No stats record by default, so dataset_extent exercises its
        # sampling fallback — the path a not-yet-re-ingested index takes.
        return {"vectors": {}}

    def upsert(self, vectors):
        for v in vectors:
            self.records.append({"metadata": v["metadata"]})

    def delete(self, delete_all=False):
        self.deleted_all = bool(delete_all)
        if delete_all:
            self.records.clear()


class FakePineconeClient:
    def __init__(self, api_key=None, records=None):
        self.inference = FakeInference()
        self._index = FakeIndex(records if records is not None else [])

    def Index(self, name):
        return self._index


def make_record(month, year, sentiment, category, value, week="1st Week", crop="", products=""):
    return {
        "metadata": {
            "month": month,
            "year": year,
            "week": week,
            "week_num": 1,
            "category": category,
            "sentiment": sentiment,
            "value": value,
            "crop": crop,
            "products": products,
        }
    }


@pytest.fixture
def fake_pinecone_factory(monkeypatch):
    """factory(records) -> a FakePineconeClient that retrieval.connect returns."""

    def _factory(records):
        client = FakePineconeClient(records=records)
        monkeypatch.setattr(retrieval, "connect",
                            lambda api_key: (client, client.Index(None)))
        # Ingestion constructs Pinecone itself, so give it the same client.
        monkeypatch.setitem(sys.modules, "pinecone",
                            type(sys)("pinecone"))
        sys.modules["pinecone"].Pinecone = lambda api_key=None: client
        return client

    return _factory


# ─────────────────────────────── fake Groq ───────────────────────────

class FakeGroqResponse:
    def __init__(self, content):
        self.choices = [type("C", (), {
            "message": type("M", (), {"content": content})(),
            "delta": type("D", (), {"content": content})(),
        })()]


class FakeGroqCompletions:
    def __init__(self, content_or_fn):
        self._content_or_fn = content_or_fn

    def create(self, **kwargs):
        content = (self._content_or_fn(**kwargs) if callable(self._content_or_fn)
                   else self._content_or_fn)
        if kwargs.get("stream"):
            return iter([FakeGroqResponse(content)])
        return FakeGroqResponse(content)


class FakeGroqClient:
    """content_or_fn is a fixed response string, or a callable (**kwargs) -> str
    for tests that need to inspect the prompt that was sent."""

    def __init__(self, content_or_fn="[]", api_key=None):
        self.chat = type("Chat", (), {"completions": FakeGroqCompletions(content_or_fn)})()


@pytest.fixture
def fake_groq_factory(monkeypatch):
    def _factory(content_or_fn="[]"):
        monkeypatch.setattr(llm, "_client", lambda api_key: FakeGroqClient(content_or_fn))
        return _factory

    return _factory


def raising_groq_factory(monkeypatch):
    """Every Groq call raises — used to verify each LLM-assisted feature
    degrades quietly instead of breaking the answer."""
    def _raise(api_key):
        raise RuntimeError("simulated Groq outage")
    monkeypatch.setattr(llm, "_client", _raise)


# ─────────────────────────── pipeline under test ─────────────────────

def run_query(user_query, pinecone_api_key, groq_api_key=None, prior_context=None):
    """The three real calls a caller makes — build_plan, gather, compose —
    flattened into the one dict these tests assert against.

    Deliberately thin: it renames fields and does no logic of its own, so a
    behaviour change shows up as a failing assertion rather than being
    absorbed here.
    """
    # Reply-only questions are answered before anything needs an index.
    prior = prior_context
    plan = build_plan(user_query, prior_context=prior)
    if plan.mode == MODE_REPLY:
        kind = ("blocked" if plan.blocked
                else "capability" if plan.canned_reply == CAPABILITY_REPLY
                else "meta_feedback" if plan.canned_reply == CORRECTION_ACK_REPLY
                else "reply")
        return {"kind": kind, "reply": plan.canned_reply}

    if not pinecone_api_key:
        return {"kind": "no_key",
                "reply": "🤖 Execution Halted: Pinecone API key is not configured."}

    pc, index = retrieval.connect(pinecone_api_key)
    latest = retrieval.dataset_extent(index)
    plan = build_plan(user_query, latest_month_year=latest, prior_context=prior)

    if plan.needs_assist and groq_api_key:
        assist = llm.classify_query(user_query, groq_api_key)
        if assist:
            plan = build_plan(user_query, latest_month_year=latest,
                              prior_context=prior, assist=assist)

    evidence = retrieval.gather(plan, index, pc)
    answer = compose.compose(plan, evidence)

    reply = answer.text
    if answer.top and groq_api_key:
        extra = compose.narrate_result(
            plan.rank_dimension or "month", answer.top[0], answer.top[1],
            [str((m.get("metadata") or {}).get("value", "")) for m in evidence.matches[:200]],
            groq_api_key)
        if extra:
            reply += '\n\n' + extra
    segments = evidence.segments
    first = segments[0] if segments else None

    return {
        "kind": {"prompt": "normal"}.get(answer.kind, answer.kind),
        "reply": reply,
        "badge": answer.badge,
        "header": answer.header,
        "system_prompt": answer.system_prompt,
        "user_prompt": answer.user_prompt,
        "chart": answer.chart,
        "context": answer.context,
        "query_intent": plan.intent,
        "output_format": plan.output_format,
        "timeframe_label": plan.segments[0].timeframe.label if plan.segments else None,
        "response_token_budget": answer.token_budget,
        "positive_bullets": first.positive if first else [],
        "negative_bullets": first.negative if first else [],
        "neutral_bullets": first.neutral if first else [],
        "actual_point_count": evidence.shown,
        "period_results": [(s.label, s.positive, s.negative, s.neutral) for s in segments]
                          if len(segments) > 1 else None,
        "export_rows": answer.export_rows,
        "downloads": compose.build_exports(answer, answer.text),
        "_plan": plan,
        "_evidence": evidence,
        "_answer": answer,
    }
