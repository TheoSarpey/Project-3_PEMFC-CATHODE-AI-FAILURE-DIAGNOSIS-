"""Pinecone adapter tested with a fake client (the live API was not reachable when this was written)."""
from conftest import ROOT  # noqa: F401


class FakeIndex:
    def __init__(self): self.records, self.batches, self.deleted = {}, [], []
    def upsert_records(self, namespace, records):
        self.batches.append((namespace, len(records)))
        self.records.setdefault(namespace, {}).update({r["_id"]: r for r in records})
    def search(self, namespace, query):
        recs = list(self.records.get(namespace, {}).values())
        flt = query.get("filter", {}).get("topic", {}).get("$eq")
        recs = [r for r in recs if not flt or r.get("topic") == flt]
        words = set(query["inputs"]["text"].lower().split())
        recs.sort(key=lambda r: -len(words & set(r["chunk_text"].lower().split())))
        return {"result": {"hits": [{"_id": r["_id"], "_score": 1.0, "fields": {k: v for k, v in r.items() if k != "_id"}}
                                    for r in recs[: query["top_k"]]]}}
    def delete(self, delete_all, namespace): self.deleted.append(namespace); self.records.pop(namespace, None)


class FakePC:
    def __init__(self): self.idx, self.created = FakeIndex(), []
    def has_index(self, name): return bool(self.created)
    def create_index_for_model(self, **kw): self.created.append(kw)
    def Index(self, name): return self.idx


def test_notes_are_indexed_with_metadata_and_searched_with_a_topic_filter():
    from src.vectorstore import PineconeStore, get_pinecone_settings
    pc = FakePC(); st = PineconeStore(get_pinecone_settings("x"), client=pc)
    n = st.index_notes()
    assert n >= 40 and pc.created[0]["embed"]["field_map"] == {"text": "chunk_text"}
    assert all(b[1] <= 96 for b in pc.idx.batches)                       # batch size limit respected
    hits = st.search_notes("flooding dehydration water management", k=3)
    assert hits and hits[0].chunk.doc_id and hits[0].chunk.evidence
    only = st.search_notes("adsorption energy", k=5, topic="dft-method")
    assert only and all(h.chunk.topic == "dft-method" for h in only)


def test_session_memory_is_isolated_and_deletable():
    from src.vectorstore import PineconeStore, get_pinecone_settings
    pc = FakePC(); st = PineconeStore(get_pinecone_settings("x"), client=pc); st.ensure_index()
    a, b = st.new_session_id(), st.new_session_id()
    st.save_comments(a, ["membrane dried out at 30 percent RH"])
    assert st.recall_comments(a, "membrane dried") and st.recall_comments(b, "membrane dried") == []   # no leakage across sessions
    st.delete_session(a)
    assert st.recall_comments(a, "membrane dried") == []


def test_retriever_falls_back_to_local_when_pinecone_fails_and_key_is_never_shown():
    from src.rag import build_retriever
    from src.vectorstore import PineconeRetriever, PineconeStore, get_pinecone_settings

    class Broken(FakePC):
        def has_index(self, name): raise RuntimeError("network down")
    local = build_retriever("bm25")
    r = PineconeRetriever(PineconeStore(get_pinecone_settings("pk-secret-value"), client=Broken()), local)
    hits = r.search("flooding at high current", k=3)
    assert hits and r.last_backend == "local fallback"
    assert "pk-secret" not in PineconeStore(get_pinecone_settings("pk-secret-value"), client=FakePC()).status()


def test_missing_key_means_unavailable_not_a_crash():
    import os
    from src.vectorstore import PineconeStore, get_pinecone_settings
    old = os.environ.pop("PINECONE_API_KEY", None)
    try:
        st = PineconeStore(get_pinecone_settings())
        assert not st.available and "no PINECONE_API_KEY" in st.status()
    finally:
        if old: os.environ["PINECONE_API_KEY"] = old


def test_in_memory_review_store_never_touches_disk_and_can_be_cleared():
    import tempfile
    from pathlib import Path
    from src.reviews import ReviewStore
    with tempfile.TemporaryDirectory() as d:
        s = ReviewStore(Path(d) / "x.jsonl", in_memory=True)
        s.add("hello", 3, "fam"); assert len(s.load()) == 1 and not (Path(d) / "x.jsonl").exists()
        s.clear(); assert len(s.load()) == 0
