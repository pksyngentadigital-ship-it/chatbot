"""
Shared test fixtures: a fake Pinecone client/index that behaves enough like
the real thing (metadata filtering, dummy-vector "aggregation" queries) for
vog_core's pure logic to be exercised without any network calls or API keys.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import vog_core  # noqa: E402


def _matches_filter(metadata: dict, filter_conditions: dict | None) -> bool:
    if not filter_conditions:
        return True
    for field, cond in filter_conditions.items():
        wanted = cond.get("$eq")
        if str(metadata.get(field, "")) != str(wanted):
            return False
    return True


class FakeEmbedItem:
    def __init__(self, values):
        self.values = values


class FakeInference:
    """Deterministic fake embedder — returns a zero vector of the right
    dimension. Fine because FakeIndex.query ignores the vector's actual
    content and answers purely from the metadata filter, matching how the
    real code always filters at the metadata level for anything the tests
    care about getting right."""

    def embed(self, model, inputs, parameters):
        dim = parameters.get("dimension", vog_core.EMBEDDING_DIMENSION)
        return [FakeEmbedItem([0.0] * dim) for _ in inputs]


class FakeIndex:
    def __init__(self, records):
        # records: list of {"metadata": {...}}
        self.records = records

    def query(self, vector, top_k=10, include_metadata=True, filter=None):
        matches = [r for r in self.records if _matches_filter(r["metadata"], filter)]
        return {"matches": matches[:top_k]}

    def upsert(self, vectors):
        for v in vectors:
            self.records.append({"metadata": v["metadata"]})


class FakePineconeClient:
    def __init__(self, api_key=None, records=None):
        self.inference = FakeInference()
        self._index = FakeIndex(records or [])

    def Index(self, name):
        return self._index


def make_record(month, year, sentiment, category, value, week="1st Week", crop="", products=""):
    return {
        "metadata": {
            "text": f"{category}: {value}",
            "month": month,
            "year": year,
            "week": week,
            "category": category,
            "sentiment": sentiment,
            "value": value,
            "crop": crop,
            "products": products,
        }
    }


@pytest.fixture
def fake_pinecone_factory(monkeypatch):
    """Returns a function(records) -> patches vog_core.Pinecone to serve those
    records, and gives back the FakePineconeClient instance for direct use."""
    created = {}

    def _factory(records):
        client = FakePineconeClient(records=records)

        def _ctor(api_key=None):
            return client

        monkeypatch.setattr(vog_core, "Pinecone", _ctor)
        created["client"] = client
        return client

    return _factory
