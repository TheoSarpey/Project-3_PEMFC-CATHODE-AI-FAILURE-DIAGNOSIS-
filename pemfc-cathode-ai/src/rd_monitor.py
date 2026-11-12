"""Scientific R&D Intelligence Agent (bonus): watch the electrocatalysis literature.

Hard rule: **bibliographic facts never come from the LLM.**  DOIs, titles, authors and
years are copied from a source record (the public Crossref API or a CSV the user
provides).  The LLM only reads the title/abstract of a record that is already in hand and
returns a relevance score and a short summary.  The final digest is checked: any DOI in it
that is not in the fetched records is reported, and records without an abstract are never
"summarised" (the digest says that no abstract was available).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from .llm import LLMUnavailable, Usage

CROSSREF = "https://api.crossref.org/works"
DEFAULT_QUERIES = ["oxygen reduction catalyst PEM fuel cell cathode",
                   "PEM fuel cell cathode catalyst durability",
                   "platinum alloy ORR membrane electrode assembly"]

RELEVANCE_TERMS = {          # weight per matched term (title + abstract, lower case)
    "oxygen reduction": 3, "orr": 2, "cathode": 2, "pemfc": 2, "pem fuel cell": 2, "proton exchange membrane": 2,
    "membrane electrode assembly": 2, "catalyst": 1, "platinum": 1, "pt/c": 2, "ptco": 2, "ptni": 2, "fe-n-c": 2,
    "ionomer": 1, "durability": 1, "degradation": 1, "mass transport": 1, "flooding": 1, "electrochemical impedance": 1,
}


@dataclass
class Paper:
    doi: str
    title: str
    authors: str = ""
    year: int | None = None
    journal: str = ""
    abstract: str = ""
    source: str = "user-csv"

    @property
    def url(self) -> str:
        return f"https://doi.org/{self.doi}" if self.doi else ""


# ----------------------------------------------------------------------------- sources
def fetch_crossref(query: str, days: int = 60, rows: int = 15, timeout: int = 20) -> list[Paper]:
    """Recent works matching `query` from the public Crossref REST API (no key needed)."""
    import requests
    since = (date.today() - timedelta(days=days)).isoformat()
    params = {"query.bibliographic": query, "filter": f"from-pub-date:{since},type:journal-article",
              "rows": rows, "sort": "published", "order": "desc",
              "select": "DOI,title,author,issued,container-title,abstract"}
    mailto = os.getenv("CROSSREF_MAILTO")          # optional: puts you in Crossref's "polite pool"
    headers = {"User-Agent": f"pemfc-cathode-ai/1.0 (mailto:{mailto})" if mailto else "pemfc-cathode-ai/1.0"}
    r = requests.get(CROSSREF, params=params, headers=headers, timeout=timeout)
    r.raise_for_status()
    out = []
    for it in r.json().get("message", {}).get("items", []):
        title = (it.get("title") or [""])[0]
        authors = ", ".join(f"{a.get('given', '')} {a.get('family', '')}".strip() for a in it.get("author", [])[:4])
        year = (it.get("issued", {}).get("date-parts") or [[None]])[0][0]
        abstract = re.sub(r"<[^>]+>", " ", it.get("abstract", "") or "").strip()
        if it.get("DOI") and title:
            out.append(Paper(it["DOI"], title, authors, year, (it.get("container-title") or [""])[0], abstract, "crossref"))
    return out


def load_watchlist(path: str | Path) -> list[Paper]:
    """Records you exported yourself (CSV columns: doi,title,authors,year,journal,abstract)."""
    df = pd.read_csv(path).fillna("")
    return [Paper(str(r.doi), str(r.title), str(r.authors), int(r.year) if str(r.year).isdigit() else None,
                  str(r.journal), str(r.abstract), "user-csv") for r in df.itertuples()]


def dedupe(papers: list[Paper]) -> list[Paper]:
    seen, out = set(), []
    for p in papers:
        key = p.doi.lower()
        if key and key not in seen:
            seen.add(key)
            out.append(p)
    return out


# ----------------------------------------------------------------------------- triage
def keyword_relevance(p: Paper) -> int:
    text = (p.title + " " + p.abstract).lower()
    return sum(w for term, w in RELEVANCE_TERMS.items() if term in text)


TRIAGE_SYSTEM = """You triage one scientific record for a PEM fuel-cell cathode catalyst team.
Use ONLY the title and abstract given. Reply with JSON:
{"relevance": 0-3, "topic": "catalyst|durability|diagnostics|membrane/ionomer|other",
 "summary": "<= 2 sentences restating what the abstract says, or 'No abstract available.'",
 "claims_need_verification": true|false}
Do not add facts, numbers, DOIs or citations that are not in the text. If there is no abstract, relevance is
based on the title only and the summary must be 'No abstract available.'"""


def triage(papers: list[Paper], llm=None) -> tuple[pd.DataFrame, Usage]:
    rows, usage = [], Usage()
    for p in papers:
        kw = keyword_relevance(p)
        row = {"doi": p.doi, "title": p.title, "year": p.year, "journal": p.journal, "source": p.source,
               "keyword_score": kw, "relevance": min(3, kw // 3), "topic": "unclassified",
               "summary": p.abstract[:240] + ("..." if len(p.abstract) > 240 else "") if p.abstract else "No abstract available.",
               "method": "keyword"}
        if llm is not None and getattr(llm, "available", False):
            try:
                js, u = llm.complete_json(TRIAGE_SYSTEM, f"TITLE: {p.title}\nABSTRACT: {p.abstract or '(none)'}")
                usage.add(u)
                if js:
                    row.update({"relevance": int(js.get("relevance", row["relevance"])),
                                "topic": js.get("topic", "other"),
                                "summary": js.get("summary", row["summary"]) if p.abstract else "No abstract available.",
                                "method": "llm"})
            except LLMUnavailable:
                pass
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["relevance", "keyword_score"], ascending=False).reset_index(drop=True), usage


# ----------------------------------------------------------------------------- digest
def build_digest(triaged: pd.DataFrame, min_relevance: int = 1, top: int = 10) -> str:
    """Markdown digest assembled by code from source records (not written by the LLM)."""
    sel = triaged[triaged["relevance"] >= min_relevance].head(top)
    if sel.empty:
        return "No records at or above the relevance threshold."
    lines = ["# R&D intelligence digest", ""]
    for r in sel.itertuples():
        lines += [f"**{r.title}** ({r.year or 'n.d.'}, {r.journal or 'journal n/a'})  ",
                  f"DOI: {r.doi} · relevance {r.relevance}/3 · topic: {r.topic} · scored by {r.method}  ",
                  f"{r.summary}", ""]
    lines.append("_Records come from the listed source; summaries restate abstracts only and must be checked against the paper._")
    return "\n".join(lines)


def verify_digest(text: str, papers: list[Paper]) -> list[str]:
    """DOIs in the digest that are not in the source records (should always be empty)."""
    known = {p.doi.lower() for p in papers}
    found = re.findall(r"10\.\d{4,9}/[^\s\]\)\"'<>,;]+", text)
    return [d for d in found if d.rstrip(".").lower() not in known]
