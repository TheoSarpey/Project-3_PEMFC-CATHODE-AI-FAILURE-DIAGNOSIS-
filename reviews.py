"""Lab comments and reviews: NLP for free-text feedback on test runs.

What is real and what is not
----------------------------
No public corpus of PEMFC lab-notebook comments exists that we may redistribute,
so the *training* corpus here is **synthetic**: it is assembled from hand-written
observation phrases (see ``OBSERVATIONS``).  It exists to demonstrate and test the
pipeline.  Because the phrases are limited, accuracy on it is optimistic; the
notebook therefore splits **by phrase** so that no observation wording appears in
both train and test.  Comments typed into the app by real users are stored
locally (git-ignored) and are never used for training unless you do it yourself.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import REVIEWS_DIR, SEED
from .llm import LLMUnavailable, Usage

ASPECTS = ["kinetics", "ohmic", "mass_transport", "durability", "reproducibility", "cost"]
SENTIMENTS = ["negative", "neutral", "positive"]
ASPECT_TO_ZONE = {"kinetics": "kinetic", "ohmic": "ohmic", "mass_transport": "mass_transport"}

# (aspect, sentiment) -> observation phrases.  The index of a phrase is its group id.
OBSERVATIONS: dict[tuple[str, str], list[str]] = {
    ("kinetics", "negative"): [
        "low-current voltage sagged and the open-circuit value looked poor",
        "activation loss at the cathode seemed larger than in the last assembly",
        "the early part of the curve fell steeply, suggesting sluggish oxygen reduction",
        "the kinetic region was weak and the catalyst looked under-utilised"],
    ("kinetics", "neutral"): [
        "the kinetic region looked similar to the reference assembly",
        "open-circuit voltage was in the expected range",
        "the low-current part of the curve was unremarkable",
        "activation losses were comparable to the previous run"],
    ("kinetics", "positive"): [
        "open-circuit voltage was high and the kinetic region was sharp",
        "the catalyst showed strong activity at low current",
        "activation loss was clearly smaller than before",
        "oxygen reduction kinetics looked excellent after conditioning"],
    ("ohmic", "negative"): [
        "resistance climbed during the run and the membrane seemed dry",
        "the linear region was steep, pointing to high ohmic loss",
        "high-frequency resistance was much higher than expected",
        "contact resistance looked poor after reassembly"],
    ("ohmic", "neutral"): [
        "resistance stayed near the usual value",
        "the linear slope matched the reference curve",
        "the high-frequency intercept was unchanged",
        "ohmic behaviour was typical for this humidity"],
    ("ohmic", "positive"): [
        "resistance dropped nicely once humidity was raised",
        "the membrane stayed well hydrated and the slope was shallow",
        "high-frequency resistance was low and stable",
        "contact resistance improved after adjusting compression"],
    ("mass_transport", "negative"): [
        "voltage collapsed at high current, consistent with flooding",
        "an early knee appeared and the signal became noisy",
        "oxygen transport looked limited at the top of the curve",
        "water seemed to block the cathode pores at high load"],
    ("mass_transport", "neutral"): [
        "the knee appeared at the expected current density",
        "high-current behaviour was similar to earlier tests",
        "no unusual transport limitation was visible",
        "the curve bent slightly at high current as usual"],
    ("mass_transport", "positive"): [
        "no knee was visible even at the highest current",
        "high-current performance was excellent with stable voltage",
        "water removal was effective and transport losses were small",
        "peak power increased thanks to better oxygen supply"],
    ("durability", "negative"): [
        "voltage decayed steadily over the hold and did not recover",
        "performance fell after cycling, probably from surface area loss",
        "power declined across the activation sets",
        "there are signs of catalyst degradation after the long hold"],
    ("durability", "neutral"): [
        "voltage over the hold was roughly flat",
        "performance after cycling was close to the starting value",
        "there was no clear trend across activation sets",
        "the decay rate was within the usual scatter"],
    ("durability", "positive"): [
        "performance improved over the activation sets and then levelled off",
        "no measurable decay after the cycling block",
        "voltage held steady over the long constant-current step",
        "the cell recovered fully after the purge"],
    ("reproducibility", "negative"): [
        "the repeat run disagreed strongly with the first measurement",
        "scatter between assemblies was larger than the effect we wanted to see",
        "the curve was not reproducible on a second day",
        "results changed after the same protocol was repeated"],
    ("reproducibility", "neutral"): [
        "the repeat measurement was within normal scatter",
        "two runs agreed reasonably well",
        "day-to-day variation was moderate",
        "the second assembly behaved similarly"],
    ("reproducibility", "positive"): [
        "repeat measurements overlapped almost perfectly",
        "three assemblies gave consistent curves",
        "results were highly reproducible across days",
        "the repeat run confirmed the earlier curve"],
    ("cost", "negative"): [
        "the platinum loading needed for this result seems too expensive",
        "a high loading gives little gain for the extra metal",
        "the improvement does not justify the added catalyst cost",
        "this approach would be costly to scale"],
    ("cost", "neutral"): [
        "catalyst loading was the same as in the reference",
        "cost was not affected by this change",
        "there was no change in material use",
        "loading stayed at the standard value"],
    ("cost", "positive"): [
        "similar power with less platinum is promising",
        "good performance at a lower loading",
        "lower catalyst use without losing power",
        "the cheaper variant performed nearly as well"],
}
FAMILIES = ["Standard Test of Nafion Membrane 112", "Activation Test MEA Standard Protocol",
            "Activation Test MEA Constant Current 0.25A", "Activation Test MEA Cycling Potential",
            "Activation Test MEA Constant Voltage 0.6V"]
CLOSERS = {"negative": ["Needs follow-up.", "I would not trust this assembly yet.", "", "", ""],
           "neutral": ["Logging for the record.", "No action needed.", "", "", ""],
           "positive": ["Worth repeating.", "Encouraging.", "", "", ""]}


def generate_synthetic_reviews(per_phrase: int = 3, seed: int = SEED) -> pd.DataFrame:
    """Synthetic comments.  `phrase_id` identifies the observation wording (group for splitting)."""
    rng = random.Random(seed)
    rows, pid = [], 0
    for (aspect, sentiment), phrases in OBSERVATIONS.items():
        for phrase in phrases:
            pid += 1
            for _ in range(per_phrase):
                fam = rng.choice(FAMILIES)
                p, rh = rng.choice([5, 15, 25]), rng.choice([30, 50, 100])
                ctx = rng.choice([f"{fam}, {p} psig, {rh}% RH:", f"Run at {p} psig and {rh}% RH -", f"{fam}:", ""])
                text = " ".join(x for x in [ctx, phrase[0].upper() + phrase[1:] + ".", rng.choice(CLOSERS[sentiment])] if x)
                rating = {"negative": rng.choice([1, 2]), "neutral": 3, "positive": rng.choice([4, 5])}[sentiment]
                rows.append({"text": text, "sentiment": sentiment, "aspect": aspect, "rating": rating,
                             "test_family": fam, "phrase_id": pid, "synthetic": True})
    return pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)


# --------------------------------------------------------------------------- lexicon baseline
POS_WORDS = ("excellent high stable improved improve better strong sharp promising good encouraging consistent "
             "reproducible lower shallow hydrated recovered effective confirmed overlapped steady nicely cheaper "
             "clearly").split()
NEG_WORDS = ("poor sag sagged steep weak sluggish collapsed noisy limited block blocked flood flooding dry climbed "
             "higher decayed fell declined degradation disagreed scatter expensive costly strongly larger "
             "unreproducible changed").split()
NEUTRAL_CUES = ("similar same expected usual typical unchanged comparable normal reasonably moderate unremarkable "
                "standard stayed roughly close matched").split()


def lexicon_sentiment(text: str) -> str:
    """Transparent rule baseline: count domain-specific positive/negative/neutral cue words."""
    toks = set(re.findall(r"[a-z]+", text.lower()))
    pos, neg, neu = (len(toks & set(w)) for w in (POS_WORDS, NEG_WORDS, NEUTRAL_CUES))
    if neg > pos and neg >= neu:
        return "negative"
    if pos > neg and pos >= neu:
        return "positive"
    return "neutral"


def lexicon_features(texts):
    """[pos, neg, neutral] cue counts as a dense array (an extra feature block for the classifier)."""
    out = []
    for t in texts:
        toks = set(re.findall(r"[a-z]+", t.lower()))
        out.append([len(toks & set(w)) for w in (POS_WORDS, NEG_WORDS, NEUTRAL_CUES)])
    return np.array(out, dtype=float)


# --------------------------------------------------------------------------- classifiers
@dataclass
class ReviewModels:
    sentiment: object
    aspect: object


def train_review_models(df: pd.DataFrame | None = None) -> ReviewModels:
    """TF-IDF (1-2 grams) + logistic regression, one model per label."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    df = generate_synthetic_reviews() if df is None else df

    from sklearn.pipeline import FeatureUnion
    from sklearn.preprocessing import FunctionTransformer

    def make(with_lexicon: bool):
        tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
        feats = (FeatureUnion([("tfidf", tfidf), ("lex", FunctionTransformer(lexicon_features))])
                 if with_lexicon else tfidf)
        return make_pipeline(feats, LogisticRegression(max_iter=2000, C=5.0))
    return ReviewModels(make(True).fit(df["text"], df["sentiment"]), make(False).fit(df["text"], df["aspect"]))


# --------------------------------------------------------------------------- extraction
NUM_PATTERNS = {
    "pressure_psig": r"(\d+(?:\.\d+)?)\s*psig",
    "relative_humidity_pct": r"(\d+(?:\.\d+)?)\s*%\s*(?:rh|relative humidity)",
    "voltage_V": r"(\d+(?:\.\d+)?)\s*v\b",
    "current_density_mA_cm2": r"(\d+(?:\.\d+)?)\s*ma\s*/?\s*cm",
    "temperature_C": r"(\d+(?:\.\d+)?)\s*(?:°\s*c|deg\s*c|celsius)",
}
SYMPTOMS = {
    "flooding": r"flood|blocked|water block|waterlog",
    "dehydration": r"\bdry\b|dried|dehydrat|resistance (?:climb|rose|increas)",
    "low_ocv": r"open[- ]circuit.*(?:low|poor)|low ocv|poor open",
    "voltage_decay": r"decay|declin|degrad|fell after|drifted",
    "noisy_signal": r"noisy|unstable|oscillat",
    "poor_reproducibility": r"not reproducible|disagree|scatter",
}


def extract_rules(text: str) -> dict:
    """Offline structured extraction: numbers with units and symptom keywords."""
    t = text.lower()
    nums = {k: [float(x) for x in re.findall(p, t)] for k, p in NUM_PATTERNS.items()}
    return {"numbers": {k: v for k, v in nums.items() if v},
            "symptoms": [s for s, p in SYMPTOMS.items() if re.search(p, t)]}


EXTRACT_SYSTEM = """Extract facts from ONE lab comment about a PEM fuel-cell test. Reply with a JSON object:
{"sentiment": "negative|neutral|positive", "aspects": [subset of kinetics, ohmic, mass_transport, durability,
reproducibility, cost], "symptoms": [short phrases copied from the comment], "conditions": {"pressure_psig": number|null,
"relative_humidity_pct": number|null}, "suspected_zone": "kinetic|ohmic|mass_transport|unknown"}.
Use only what the comment says; use null or "unknown" when it does not say."""


def analyze_comment(text: str, models: ReviewModels, llm=None) -> dict:
    """Full NLP pass on one comment: classifiers + rule extraction (+ LLM extraction if online)."""
    s_prob = models.sentiment.predict_proba([text])[0]
    a_prob = models.aspect.predict_proba([text])[0]
    s_cls, a_cls = models.sentiment.classes_, models.aspect.classes_
    order = np.argsort(-a_prob)
    aspects = [(a_cls[i], float(a_prob[i])) for i in order[:2]]
    out = {"sentiment": str(s_cls[int(np.argmax(s_prob))]),
           "sentiment_confidence": float(np.max(s_prob)),
           "aspects": aspects, "top_aspect": aspects[0][0],
           "suspected_zone": ASPECT_TO_ZONE.get(aspects[0][0], "unknown"),
           **extract_rules(text), "llm_extraction": None, "usage": Usage()}
    if llm is not None and getattr(llm, "available", False):
        try:
            js, usage = llm.complete_json(EXTRACT_SYSTEM, text)
            out["llm_extraction"], out["usage"] = js, usage
        except LLMUnavailable:
            pass
    return out


# --------------------------------------------------------------------------- store
class ReviewStore:
    """Comments typed into the app.

    in_memory=True (the public-app default): rows live only in this object, i.e. in one browser session, and
    vanish with it. Otherwise they are appended to a git-ignored JSON-lines file shared by everyone who uses
    that machine: fine for a single user, wrong for a public deployment.
    """

    COLS = ["ts", "author", "text", "rating", "test_family", "sentiment", "top_aspect", "suspected_zone", "symptoms"]

    def __init__(self, path: Path | None = None, in_memory: bool = False):
        self.in_memory = in_memory
        self._rows: list[dict] = []
        self.path = Path(path) if path else REVIEWS_DIR / "user_reviews.jsonl"
        if not in_memory:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, text: str, rating: int | None, test_family: str | None, analysis: dict | None = None,
            author: str = "anonymous") -> dict:
        rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "author": author[:40],
               "text": text.strip()[:2000], "rating": rating, "test_family": test_family}
        if analysis:
            rec.update({"sentiment": analysis["sentiment"], "top_aspect": analysis["top_aspect"],
                        "suspected_zone": analysis["suspected_zone"], "symptoms": analysis.get("symptoms", [])})
        if self.in_memory:
            self._rows.append(rec)
        else:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    def load(self) -> pd.DataFrame:
        if self.in_memory:
            return pd.DataFrame(self._rows) if self._rows else pd.DataFrame(columns=self.COLS)
        if not self.path.exists():
            return pd.DataFrame(columns=self.COLS)
        rows = [json.loads(l) for l in self.path.read_text(encoding="utf-8").splitlines() if l.strip()]
        return pd.DataFrame(rows)

    def clear(self) -> None:
        """Delete everything this store holds (the 'delete my comments' button)."""
        self._rows = []
        if not self.in_memory and self.path.exists():
            self.path.unlink()


SUMMARY_SYSTEM = """You summarise lab comments about PEM fuel-cell tests for a project lead.
Use ONLY the numbered comments provided. Reference comments as [C1], [C2]... after each claim.
Report: (1) recurring problems, (2) what went well, (3) disagreements between comments.
Never add causes or numbers that are not in the comments. Maximum 8 sentences."""


def summarise_comments(df: pd.DataFrame, llm=None, max_comments: int = 40) -> tuple[str, Usage]:
    """LLM summary with citations to comment numbers; offline fallback counts aspects and quotes examples."""
    if df.empty:
        return "No comments yet.", Usage()
    df = df.tail(max_comments).reset_index(drop=True)
    if llm is not None and getattr(llm, "available", False):
        body = "\n".join(f"[C{i + 1}] ({r.get('test_family') or 'n/a'}; rating {r.get('rating')}) {r['text']}"
                         for i, r in df.iterrows())
        try:
            r = llm.chat([{"role": "system", "content": SUMMARY_SYSTEM}, {"role": "user", "content": body}])
            return r.text.strip(), r.usage
        except LLMUnavailable:
            pass
    neg = df[df.get("sentiment", pd.Series(dtype=str)) == "negative"] if "sentiment" in df else df.iloc[0:0]
    counts = df["top_aspect"].value_counts().to_dict() if "top_aspect" in df else {}
    lines = [f"Offline summary of {len(df)} comments. Aspects mentioned: {counts or 'n/a'}."]
    if len(neg):
        lines.append("Examples of negative comments: " + " | ".join(neg["text"].head(3)))
    return " ".join(lines), Usage()
