"""Optional Pinecone vector store: the knowledge notes as a retriever, and per-session comment memory.

STATUS: written against Pinecone's documented *integrated-embedding* Python API (SDK v5-v7):
    pc.has_index / pc.create_index_for_model / pc.Index / index.upsert_records / index.search / index.delete
It could NOT be run against Pinecone while this was written (no network, SDK not installed); the logic is tested
with a fake client (tests/test_vectorstore.py).  If your SDK version names these calls differently, only this file
needs a change.  Check `pip show pinecone` and the current Pinecone docs first.

Design rules
* Local retrieval (BM25/dense/hybrid) stays the default; Pinecone is opt-in and falls back to local if it fails.
* Keys come from the environment / st.secrets / the sidebar (session only); they are never logged or stored.
* The index (and the notes in it) are shared and public to anyone with your key; comment memory is isolated by
  **one namespace per browser session**, can be deleted with one call, and is OFF unless the user turns it on.
* Pinecone indexing is eventually consistent: records may take a few seconds to become searchable.
"""
from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from .rag import Chunk, Hit, chunk_documents, load_documents

BATCH = 96                      # documented upper bound for integrated-embedding upserts per request
DEFAULT_INDEX = "pemfc-cathode-ai"
DEFAULT_EMBED_MODEL = "llama-text-embed-v2"       # a Pinecone-hosted embedding model
NOTES_NS = "knowledge-notes"


@dataclass(frozen=True)
class PineconeSettings:
    api_key: str | None
    index_name: str
    cloud: str
    region: str
    embed_model: str

    @property
    def has_key(self) -> bool:
        return bool(self.api_key)


def get_pinecone_settings(api_key: str | None = None, index_name: str | None = None) -> PineconeSettings:
    return PineconeSettings(
        api_key=(api_key or os.getenv("PINECONE_API_KEY") or "").strip() or None,
        index_name=(index_name or os.getenv("PINECONE_INDEX_NAME") or DEFAULT_INDEX).strip(),
        cloud=os.getenv("PINECONE_CLOUD", "aws").strip(),
        region=os.getenv("PINECONE_REGION", "us-east-1").strip(),
        embed_model=os.getenv("PINECONE_EMBED_MODEL", DEFAULT_EMBED_MODEL).strip())


class VectorStoreUnavailable(RuntimeError):
    pass


def _get(obj, key, default=None):
    """Read a field from a dict-like or attribute-style SDK object."""
    try:
        return obj[key]
    except Exception:
        return getattr(obj, key, default)


class PineconeStore:
    """Thin adapter. Pass `client` (an object with .has_index/.create_index_for_model/.Index) to test offline."""

    def __init__(self, settings: PineconeSettings | None = None, client=None):
        self.settings = settings or get_pinecone_settings()
        self.last_error: str | None = None
        self._pc = client
        self._index = None
        if client is None:
            if not self.settings.has_key:
                self.last_error = "no PINECONE_API_KEY"
                return
            try:
                from pinecone import Pinecone
                self._pc = Pinecone(api_key=self.settings.api_key)
            except Exception as exc:
                self.last_error = f"pinecone SDK unavailable: {type(exc).__name__}"

    @property
    def available(self) -> bool:
        return self._pc is not None

    def status(self) -> str:
        if self.available:
            return f"Pinecone ready · index {self.settings.index_name}"
        return f"Pinecone off ({self.last_error})"

    # -- index ------------------------------------------------------------------------------------
    def ensure_index(self):
        if not self.available:
            raise VectorStoreUnavailable(self.status())
        if self._index is not None:
            return self._index
        try:
            name = self.settings.index_name
            if not self._pc.has_index(name):
                self._pc.create_index_for_model(
                    name=name, cloud=self.settings.cloud, region=self.settings.region,
                    embed={"model": self.settings.embed_model, "field_map": {"text": "chunk_text"}})
            self._index = self._pc.Index(name)
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {str(exc)[:160]}"
            raise VectorStoreUnavailable(self.last_error) from exc
        return self._index

    def _upsert(self, namespace: str, records: list[dict]) -> int:
        index = self.ensure_index()
        try:
            for i in range(0, len(records), BATCH):
                index.upsert_records(namespace, records[i:i + BATCH])
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {str(exc)[:160]}"
            raise VectorStoreUnavailable(self.last_error) from exc
        return len(records)

    def _search(self, namespace: str, text: str, k: int, flt: dict | None) -> list:
        index = self.ensure_index()
        query = {"top_k": int(k), "inputs": {"text": text}}
        if flt:
            query["filter"] = flt
        try:
            res = index.search(namespace=namespace, query=query)
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {str(exc)[:160]}"
            raise VectorStoreUnavailable(self.last_error) from exc
        return list(_get(_get(res, "result"), "hits", []) or [])

    # -- knowledge notes -----------------------------------------------------------------------------
    def index_notes(self, chunk_words: int | None = None, wait_s: float = 0.0) -> int:
        """Upsert every chunk of data/knowledge/*.md (idempotent: same ids overwrite)."""
        chunks = chunk_documents(load_documents(), chunk_words)
        records = [{"_id": c.chunk_id, "chunk_text": c.title + ". " + c.text, "doc_id": c.doc_id, "title": c.title,
                    "topic": c.topic, "evidence": c.evidence, "body": c.text} for c in chunks]
        n = self._upsert(NOTES_NS, records)
        if wait_s:
            time.sleep(wait_s)
        return n

    def search_notes(self, query: str, k: int = 4, topic: str | None = None) -> list[Hit]:
        flt = {"topic": {"$eq": topic}} if topic else None
        hits = []
        for rank, h in enumerate(self._search(NOTES_NS, query, k, flt), 1):
            f = _get(h, "fields", {}) or {}
            chunk = Chunk(str(_get(h, "_id")), str(f.get("doc_id", "")), str(f.get("title", "")), str(f.get("topic", "")),
                          str(f.get("evidence", "")), str(f.get("body", f.get("chunk_text", ""))))
            hits.append(Hit(chunk, float(_get(h, "_score", 0.0) or 0.0), rank))
        return hits

    # -- per-session comment memory ----------------------------------------------------------------------
    @staticmethod
    def new_session_id() -> str:
        return uuid.uuid4().hex

    @staticmethod
    def _ns(session_id: str) -> str:
        return f"session-{session_id}"

    def save_comments(self, session_id: str, comments: list[str]) -> int:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        recs = [{"_id": uuid.uuid4().hex, "chunk_text": c[:2000], "timestamp": now} for c in comments if c.strip()]
        return self._upsert(self._ns(session_id), recs) if recs else 0

    def recall_comments(self, session_id: str, query: str, k: int = 5) -> list[str]:
        return [str((_get(h, "fields", {}) or {}).get("chunk_text", "")) for h in self._search(self._ns(session_id), query, k, None)]

    def delete_session(self, session_id: str) -> None:
        index = self.ensure_index()
        try:
            index.delete(delete_all=True, namespace=self._ns(session_id))
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {str(exc)[:160]}"
            raise VectorStoreUnavailable(self.last_error) from exc


class PineconeRetriever:
    """Same interface as the local retrievers (`name`, `chunks`, `search`); falls back to a local one on error."""

    def __init__(self, store: PineconeStore, fallback):
        self.store, self.fallback = store, fallback
        self.chunks = fallback.chunks
        self.name = "pinecone (fallback: " + fallback.name + ")"
        self.last_backend = "pinecone"

    def search(self, query: str, k: int = 4, topic: str | None = None) -> list[Hit]:
        try:
            hits = self.store.search_notes(query, k, topic)
            if hits:
                self.last_backend = "pinecone"
                return hits
        except VectorStoreUnavailable:
            pass
        self.last_backend = "local fallback"
        return self.fallback.search(query, k, topic)
