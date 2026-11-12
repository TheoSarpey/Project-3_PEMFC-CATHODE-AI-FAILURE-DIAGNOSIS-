"""Retrieval-augmented generation over the curated PEMFC knowledge base.

Pipeline
--------
documents (data/knowledge/*.md)  ->  chunks  ->  retriever  ->  grounded prompt
        ->  LLM (or offline extractive fallback)  ->  citation check

Three retrievers are provided so the evaluation can compare them honestly:

* ``BM25Retriever``      keyword baseline (no learning, no embeddings)
* ``DenseRetriever``     embeddings: sentence-transformers if installed, OpenAI
                         embeddings if a key is set, otherwise TF-IDF + LSA
* ``HybridRetriever``    reciprocal-rank fusion of the two above

Every answer carries its sources, and the answer is checked: citations that
point to a source that was not retrieved are flagged, and any DOI or URL that
is not present in the retrieved text is reported as unsupported.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import KNOWLEDGE_DIR
from .llm import LLMUnavailable, Usage

STOP = set("""a an the and or of to in on for with by is are was were be been it its this that these those as at from
what which who how why when where do does did can could should would may might will not no than then so if into
about over under between within across per vs versus i we you your our their there here also very more most less
much many any some such use used using cell cells pemfc pem fuel""".split())


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens with a deliberately light suffix stripper."""
    out = []
    for w in re.findall(r"[a-z0-9][a-z0-9\-\.]*[a-z0-9]|[a-z0-9]", text.lower()):
        w = w.strip(".-")
        if w in STOP or not w:
            continue
        for suf in ("ings", "ing", "ed", "es", "s"):
            if len(w) > 4 and w.endswith(suf):
                w = w[: -len(suf)]
                break
        out.append(w)
    return out


# --------------------------------------------------------------------------- documents
@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    topic: str
    evidence: str
    text: str


def load_documents(folder: Path = KNOWLEDGE_DIR) -> list[dict]:
    docs = []
    for p in sorted(Path(folder).glob("*.md")):
        raw = p.read_text(encoding="utf-8")
        m = re.match(r"---\n(.*?)\n---\n(.*)", raw, re.S)
        if not m:
            continue
        meta = dict(line.split(": ", 1) for line in m.group(1).splitlines() if ": " in line)
        meta["text"] = m.group(2).strip()
        docs.append(meta)
    return docs


def chunk_documents(docs: list[dict], chunk_words: int | None = None, overlap: int = 15) -> list[Chunk]:
    """chunk_words=None keeps each document whole (they are 70-110 words long)."""
    chunks = []
    for d in docs:
        words = d["text"].split()
        if chunk_words is None or len(words) <= chunk_words:
            pieces = [" ".join(words)]
        else:
            step = max(1, chunk_words - overlap)
            pieces = [" ".join(words[i:i + chunk_words]) for i in range(0, len(words), step)
                      if len(words[i:i + chunk_words]) > overlap]
        for i, piece in enumerate(pieces):
            chunks.append(Chunk(f"{d['id']}-{i}", d["id"], d["title"], d.get("topic", ""),
                                d.get("evidence", ""), piece))
    return chunks


@dataclass
class Hit:
    chunk: Chunk
    score: float
    rank: int

    @property
    def doc_id(self) -> str:
        return self.chunk.doc_id


# --------------------------------------------------------------------------- retrievers
class BM25Retriever:
    """Okapi BM25, written out so there is nothing hidden (k1=1.5, b=0.75)."""
    name = "bm25"

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        # titles are included in the searchable text
        self.tokens = [tokenize(c.title + " " + c.text) for c in chunks]
        self.df = Counter(t for toks in self.tokens for t in set(toks))
        self.avgdl = np.mean([len(t) for t in self.tokens])
        self.N = len(chunks)

    def scores(self, query: str) -> np.ndarray:
        q = tokenize(query)
        s = np.zeros(self.N)
        for i, toks in enumerate(self.tokens):
            tf = Counter(toks)
            dl = len(toks)
            for term in q:
                if term not in tf:
                    continue
                idf = math.log(1 + (self.N - self.df[term] + 0.5) / (self.df[term] + 0.5))
                f = tf[term]
                s[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s

    def search(self, query: str, k: int = 4, topic: str | None = None) -> list[Hit]:
        return _top_k(self.chunks, self.scores(query), k, topic)


class DenseRetriever:
    """Semantic retrieval.  Backend is chosen automatically unless forced.

    backend = "sbert"   sentence-transformers (all-MiniLM-L6-v2), local, free
              "openai"  OpenAI embeddings through LLMClient (costs tokens)
              "lsa"     TF-IDF + truncated SVD (always available, offline)
    """
    def __init__(self, chunks: list[Chunk], backend: str = "auto", llm=None, lsa_dim: int = 48):
        self.chunks = chunks
        self.llm = llm
        self.backend = self._pick(backend)
        texts = [c.title + ". " + c.text for c in chunks]
        if self.backend == "sbert":
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            self.matrix = self._model.encode(texts, normalize_embeddings=True)
        elif self.backend == "openai":
            self.matrix = _normalise(np.array(llm.embed(texts)))
        else:
            from sklearn.decomposition import TruncatedSVD
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._tfidf = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None,
                                          ngram_range=(1, 2), sublinear_tf=True)
            x = self._tfidf.fit_transform(texts)
            dim = max(2, min(lsa_dim, x.shape[0] - 1, x.shape[1] - 1))
            self._svd = TruncatedSVD(dim, random_state=42)
            self.matrix = _normalise(self._svd.fit_transform(x))
        self.name = f"dense-{self.backend}"

    def _pick(self, backend: str) -> str:
        if backend != "auto":
            return backend
        try:
            import sentence_transformers  # noqa: F401
            return "sbert"
        except Exception:
            pass
        if self.llm is not None and getattr(self.llm, "available", False):
            return "openai"
        return "lsa"

    def _embed_query(self, q: str) -> np.ndarray:
        if self.backend == "sbert":
            return self._model.encode([q], normalize_embeddings=True)[0]
        if self.backend == "openai":
            return _normalise(np.array(self.llm.embed([q])))[0]
        return _normalise(self._svd.transform(self._tfidf.transform([q])))[0]

    def search(self, query: str, k: int = 4, topic: str | None = None) -> list[Hit]:
        return _top_k(self.chunks, self.matrix @ self._embed_query(query), k, topic)


class HybridRetriever:
    """Reciprocal-rank fusion (RRF, k=60) of BM25 and dense rankings."""
    def __init__(self, chunks: list[Chunk], dense_backend: str = "auto", llm=None):
        self.chunks = chunks
        self.bm25 = BM25Retriever(chunks)
        self.dense = DenseRetriever(chunks, dense_backend, llm)
        self.name = f"hybrid(bm25+{self.dense.name})"

    def search(self, query: str, k: int = 4, topic: str | None = None) -> list[Hit]:
        fused: dict[str, float] = {}
        for r in (self.bm25, self.dense):
            for h in r.search(query, k=len(self.chunks), topic=topic):
                fused[h.chunk.chunk_id] = fused.get(h.chunk.chunk_id, 0.0) + 1.0 / (60 + h.rank)
        by_id = {c.chunk_id: c for c in self.chunks}
        ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        return [Hit(by_id[cid], sc, i + 1) for i, (cid, sc) in enumerate(ranked)]


def _normalise(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.where(n == 0, 1, n)


def _top_k(chunks, scores, k, topic) -> list[Hit]:
    idx = np.argsort(-scores)
    hits = []
    for i in idx:
        if topic and chunks[i].topic != topic:
            continue
        hits.append(Hit(chunks[i], float(scores[i]), len(hits) + 1))
        if len(hits) == k:
            break
    return hits


def build_retriever(kind: str = "hybrid", chunk_words: int | None = None, llm=None,
                    dense_backend: str = "auto"):
    chunks = chunk_documents(load_documents(), chunk_words)
    if kind == "bm25":
        return BM25Retriever(chunks)
    if kind == "dense":
        return DenseRetriever(chunks, dense_backend, llm)
    return HybridRetriever(chunks, dense_backend, llm)


# --------------------------------------------------------------------------- answering
SYSTEM_PROMPT = """You are a PEM fuel-cell electrochemistry assistant.
Answer ONLY from the numbered SOURCES. Rules:
1. Cite every factual sentence with its source number like [S1] or [S2][S3].
2. If the sources do not contain the answer, reply exactly: NOT_IN_KB: <one sentence on what is missing>.
3. Never invent numbers, DOIs, URLs, paper titles or catalyst names that are not in the sources.
4. Separate what the ECSIM data shows from what the literature notes say.
5. Be concise: at most 6 sentences."""

DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\]\)\"'<>,;]+", re.I)
URL_RE = re.compile(r"https?://[^\s\]\)\"'<>]+", re.I)
CITE_RE = re.compile(r"\[S(\d+)\]")


@dataclass
class RagAnswer:
    question: str
    answer: str
    hits: list[Hit]
    mode: str                       # "llm" | "offline-extractive" | "abstained"
    abstained: bool = False
    cited: list[int] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)
    unsupported_links: list[str] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)

    @property
    def cited_doc_ids(self) -> list[str]:
        return [self.hits[i - 1].doc_id for i in self.cited if 1 <= i <= len(self.hits)]


def query_coverage(query: str, hits: list[Hit]) -> float:
    """Share of the query's content words that occur in the retrieved text.

    A cheap, explainable relevance gate: a question about something the
    knowledge base never mentions has coverage near 0 and is not sent to the LLM.
    """
    q = set(tokenize(query))
    if not q:
        return 0.0
    seen = set()
    for h in hits:
        seen |= set(tokenize(h.chunk.title + " " + h.chunk.text))
    return len(q & seen) / len(q)


def verify_answer(answer: str, hits: list[Hit]) -> tuple[list[int], list[int], list[str]]:
    """Return (cited, invalid_citations, unsupported_links)."""
    cited = sorted({int(n) for n in CITE_RE.findall(answer)})
    invalid = [n for n in cited if not 1 <= n <= len(hits)]
    context = " ".join(h.chunk.text for h in hits).lower()
    links = DOI_RE.findall(answer) + URL_RE.findall(answer)
    unsupported = [l for l in links if l.lower().rstrip(".") not in context]
    return cited, invalid, unsupported


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.split()) >= 5]


def extractive_answer(question: str, hits: list[Hit], n_sentences: int = 3) -> str:
    """Offline fallback: copy the best-matching sentences from the sources, with citations."""
    q = set(tokenize(question))
    scored = []
    for si, h in enumerate(hits, 1):
        for s in _sentences(h.chunk.text):
            overlap = len(q & set(tokenize(s)))
            scored.append((overlap, -si, s, si))
    scored.sort(reverse=True)
    picked = [(s, si) for ov, _, s, si in scored[:n_sentences] if ov > 0]
    if not picked:
        return "NOT_IN_KB: no sentence in the retrieved sources matches the question."
    return " ".join(f"{s} [S{si}]" for s, si in picked)


def answer_question(question: str, retriever, llm=None, k: int = 4, topic: str | None = None,
                    min_coverage: float = 0.34, history: list[dict] | None = None) -> RagAnswer:
    hits = retriever.search(question, k=k, topic=topic)
    cov = query_coverage(question, hits)
    if not hits or cov < min_coverage:
        return RagAnswer(question, f"NOT_IN_KB: the knowledge base does not cover this question "
                                   f"(query coverage {cov:.2f} < {min_coverage}).",
                         hits, "abstained", abstained=True)
    if llm is not None and getattr(llm, "available", False):
        ctx = "\n\n".join(f"[S{i}] ({h.doc_id}, {h.chunk.topic}, evidence={h.chunk.evidence}) "
                          f"{h.chunk.title}: {h.chunk.text}" for i, h in enumerate(hits, 1))
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + (history or []) + [
            {"role": "user", "content": f"SOURCES:\n{ctx}\n\nQUESTION: {question}"}]
        try:
            r = llm.chat(messages)
            text, usage, mode = r.text.strip(), r.usage, "llm"
        except LLMUnavailable:
            text, usage, mode = extractive_answer(question, hits), Usage(), "offline-extractive"
    else:
        text, usage, mode = extractive_answer(question, hits), Usage(), "offline-extractive"
    cited, invalid, unsupported = verify_answer(text, hits)
    return RagAnswer(question, text, hits, mode, abstained=text.startswith("NOT_IN_KB"),
                     cited=cited, invalid_citations=invalid, unsupported_links=unsupported, usage=usage)
