"""Honest evaluation helpers (used by notebook 07 and the app's evaluation tab).

Three questions, three evaluations:

1. Does retrieval find the right note?            ``retrieval_eval`` / ``retrieval_experiment``
2. Does the whole answer behave (cite, abstain)?  ``qa_eval``
3. Does the failure screening find known faults?  ``fault_injection_eval``

Limitation to keep in view: the knowledge base and the question list were written by the
same author in the same session, so retrieval scores are optimistic.  Add questions written
by someone who has not seen the notes before trusting the numbers.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from . import physics
from .config import EVAL_DIR
from .rag import answer_question, build_retriever


def load_questions(path=EVAL_DIR / "eval_questions.json") -> list[dict]:
    return json.loads(open(path, encoding="utf-8").read())


def retrieval_eval(retriever, questions: list[dict], k: int = 4) -> dict:
    """hit@k and MRR over answerable questions (gold = acceptable source notes)."""
    hits, rr = [], []
    for q in questions:
        if not q["answerable"]:
            continue
        ids = [h.doc_id for h in retriever.search(q["question"], k=10)]
        gold = set(q["gold_docs"])
        hits.append(any(i in gold for i in ids[:k]))
        first = next((r for r, i in enumerate(ids, 1) if i in gold), None)
        rr.append(1 / first if first else 0.0)
    return {"hit@k": float(np.mean(hits)), "MRR": float(np.mean(rr)), "n": len(hits)}


def retrieval_experiment(questions: list[dict], llm=None) -> pd.DataFrame:
    """Change one design choice at a time and measure it (the LLM-track requirement)."""
    rows = []
    for kind in ["bm25", "dense", "hybrid"]:
        for chunk_words in [None, 40]:
            r = build_retriever(kind, chunk_words=chunk_words, llm=llm, dense_backend="lsa" if llm is None else "auto")
            for k in [2, 4]:
                m = retrieval_eval(r, questions, k)
                rows.append({"retriever": r.name, "chunking": "whole note" if chunk_words is None else f"{chunk_words} words",
                             "k": k, "hit@k": m["hit@k"], "MRR": m["MRR"]})
    return pd.DataFrame(rows)


def qa_eval(ctx, questions: list[dict], k: int = 4, min_coverage: float = 0.34) -> pd.DataFrame:
    """Run the full question-answering path on every question and record behaviour and cost."""
    rows = []
    for q in questions:
        t0 = time.perf_counter()
        a = answer_question(q["question"], ctx.retriever, ctx.llm, k=k, min_coverage=min_coverage)
        dt = time.perf_counter() - t0
        gold = set(q["gold_docs"])
        rows.append({
            "id": q["id"], "answerable": q["answerable"], "category": q["category"], "mode": a.mode,
            "abstained": a.abstained,
            "behaved": (not a.abstained) if q["answerable"] else a.abstained,       # right call on answer vs refuse
            "retrieved_gold": bool(gold & {h.doc_id for h in a.hits}) if gold else None,
            "cited_gold": bool(gold & set(a.cited_doc_ids)) if gold and not a.abstained else None,
            "has_citation": bool(a.cited) if not a.abstained else None,
            "invalid_citations": len(a.invalid_citations), "unsupported_links": len(a.unsupported_links),
            "prompt_tokens": a.usage.prompt_tokens, "completion_tokens": a.usage.completion_tokens,
            "cost_usd": a.usage.cost_usd, "seconds": dt, "answer": a.answer})
    return pd.DataFrame(rows)


def qa_summary(df: pd.DataFrame) -> dict:
    ans, oos = df[df.answerable], df[~df.answerable]
    return {
        "answerable_answered": float((~ans.abstained).mean()),
        "out_of_scope_refused": float(oos.abstained.mean()),
        "cited_gold_when_answered": float(ans["cited_gold"].dropna().astype(bool).mean()) if ans["cited_gold"].notna().any() else None,
        "invalid_citations_total": int(df.invalid_citations.sum()),
        "unsupported_links_total": int(df.unsupported_links.sum()),
        "mean_seconds": float(df.seconds.mean()),
        "mean_prompt_tokens": float(df.prompt_tokens.mean()),
        "mean_completion_tokens": float(df.completion_tokens.mean()),
        "mean_cost_usd": None if df.cost_usd.isna().all() else float(df.cost_usd.mean()),
    }


# ----------------------------------------------------------------------------- failure screening
def fault_injection_eval(ctx, n_per_cell: int = 25, seed: int = 42,
                         severities: dict | None = None) -> pd.DataFrame:
    """Inject known faults into healthy curves and see whether the screening names the right zone.

    Healthy = curves that neither physics nor ML flagged.  A fault of each kind is injected at
    three severities; the curve's descriptors are recomputed and the *whole corpus is re-screened*
    with that curve replaced, exactly as a new measurement would be.
    """
    severities = severities or {"kinetic": [0.04, 0.08, 0.15], "ohmic": [0.10, 0.20, 0.40],
                                "mass_transport": [0.05, 0.12, 0.25]}
    rng = np.random.default_rng(seed)
    base = ctx.curves[["curve_id"] + [c for c in physics.META_COLS if c != "curve_id"]
                      + [f.name for f in physics.CurveDescriptors.__dataclass_fields__.values()]].copy()
    healthy = ctx.curves[(ctx.curves.dominant_zone == "none") & (~ctx.curves.ml_underperforms.fillna(False))].curve_id.to_numpy()
    rows = []
    for kind, sevs in severities.items():
        for sev in sevs:
            for cid in rng.choice(healthy, min(n_per_cell, len(healthy)), replace=False):
                g = ctx.pol[ctx.pol.curve_id == cid]
                d = physics.curve_descriptors(physics.inject_fault(g, kind, sev))
                if d is None:
                    continue
                t = base.copy()
                idx = t.index[t.curve_id == cid][0]
                for k, v in d.as_dict().items():
                    t.loc[idx, k] = v
                predicted = physics.screen_curves(t).loc[idx, "dominant_zone"]
                rows.append({"fault": kind, "severity": sev, "curve_id": int(cid), "predicted": predicted,
                             "correct": predicted == kind, "detected": predicted not in ("none", "unreliable_fit")})
    return pd.DataFrame(rows)
