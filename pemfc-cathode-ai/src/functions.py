"""Product-level functions shared by the notebooks, the agent and the Streamlit app.

Everything the app shows comes from a function in this file, so a result seen in the
app can always be reproduced in a notebook.  Heavy artefacts are built once and cached
in ``models/`` (git-ignored); ``get_context()`` rebuilds them from the raw data if they
are missing, which is what makes the app run from a fresh clone.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd

from . import physics
from .config import (DEFAULT_EMBED_MODEL, DEFAULT_MODEL, MODELS_DIR, PROJECT_ROOT, RAW_DIR, SEED, ensure_dirs)
from .data import eis_summary, load_polarization
from .models import DESIGN_VARS, DesignSurrogate, oof_curve_residuals
from .rag import RagAnswer, answer_question, build_retriever
from .reviews import ReviewModels, ReviewStore, train_review_models

DATASET_DOI = "10.26434/chemrxiv.11902023"

# --- names used by the course template (notebooks do `from src.functions import *`) -------------
ROOT = PROJECT_ROOT
RAW = RAW_DIR
MODELS = MODELS_DIR
RANDOM_STATE = SEED                 # one seed, defined in src/config.py
LLM_MODEL = DEFAULT_MODEL           # "GPT-6-LUNA"; overridden by OPEN_AI_MODEL in .env
EMBEDDING_MODEL = DEFAULT_EMBED_MODEL


@dataclass
class Context:
    """Everything the product needs, loaded once."""
    pol: pd.DataFrame
    curves: pd.DataFrame            # one row per curve: metadata, descriptors, screening, OOF residual
    eis: pd.DataFrame
    surrogate: DesignSurrogate
    retriever: object
    review_models: ReviewModels
    store: ReviewStore
    llm: object = None
    extras: dict = field(default_factory=dict)


def build_curve_corpus(pol: pd.DataFrame) -> pd.DataFrame:
    """Descriptors + peer screening + ML out-of-fold residual (second opinion)."""
    table = physics.screen_curves(physics.build_curve_table(pol))
    resid = oof_curve_residuals(pol)[["curve_id", "mean_resid", "resid_z"]]
    t = table.merge(resid, on="curve_id", how="left")
    t["ml_underperforms"] = t["resid_z"] <= -1.5
    return t


def get_context(llm=None, rebuild: bool = False, retriever_kind: str = "hybrid") -> Context:
    ensure_dirs()
    pol = load_polarization()
    cache = MODELS_DIR / "curve_corpus.csv"
    if cache.exists() and not rebuild:
        curves = pd.read_csv(cache)
    else:
        curves = build_curve_corpus(pol)
        curves.to_csv(cache, index=False)
    sur_path = MODELS_DIR / "design_surrogate.joblib"
    if sur_path.exists() and not rebuild:
        surrogate = DesignSurrogate.load(sur_path)
    else:
        surrogate = DesignSurrogate().fit(pol)
        surrogate.save(sur_path)
    rm_path = MODELS_DIR / "review_models.joblib"
    if rm_path.exists() and not rebuild:
        review_models = joblib.load(rm_path)
    else:
        review_models = train_review_models()
        joblib.dump(review_models, rm_path)
    return Context(pol, curves, eis_summary(), surrogate,
                   build_retriever(retriever_kind, llm=llm), review_models, ReviewStore(), llm)


# ----------------------------------------------------------------------------- curves
def list_curves(ctx: Context, family: str | None = None, flagged_only: bool = False,
                limit: int = 20) -> pd.DataFrame:
    t = ctx.curves
    if family:
        t = t[t["test_family"].str.contains(family, case=False, regex=False)]
    if flagged_only:
        t = t[t["dominant_zone"].isin(["kinetic", "ohmic", "mass_transport"])]
    cols = ["curve_id", "test_family", "pressure", "relative_humidity", "operating_set",
            "p_max", "dominant_zone", "resid_z"]
    return t.sort_values("p_max")[cols].head(limit)


def curve_points(ctx: Context, curve_id: int) -> pd.DataFrame:
    return ctx.pol[ctx.pol["curve_id"] == curve_id].sort_values("current_density")


def eis_context(ctx: Context, curve_id: int) -> dict | None:
    """HFR / arc-width proxies from impedance files of the same test family and conditions."""
    row = ctx.curves[ctx.curves.curve_id == curve_id].iloc[0]
    e = ctx.eis[ctx.eis["test_family"] == row["test_family"]]
    if pd.notna(row["pressure"]):
        e = e[(e["pressure"] == row["pressure"]) & (e["relative_humidity"] == row["relative_humidity"])]
    elif pd.notna(row["operating_set"]):
        e = e[e["operating_set"] == row["operating_set"]]
    if e.empty:
        return None
    return {"n_spectra": int(len(e)), "hfr_proxy_median_ohm": float(e["hfr_proxy"].median()),
            "arc_width_median_ohm": float(e["arc_width"].median()),
            "note": "proxies: the impedance files have no frequency column"}


def diagnose_curve(ctx: Context, curve_id: int, with_guidance: bool = True, k: int = 3) -> dict:
    """Physics screening + ML second opinion + EIS context + retrieved guidance."""
    rows = ctx.curves[ctx.curves.curve_id == curve_id]
    if rows.empty:
        raise KeyError(f"unknown curve_id {curve_id}")
    row = rows.iloc[0]
    diag = physics.diagnose_row(row)
    g = curve_points(ctx, curve_id)
    diag["loss_breakdown_V"] = {k_: round(float(v), 4) for k_, v in physics.breakdown_from_curve(g).items()}
    diag["ml_second_opinion"] = {
        "resid_z": None if pd.isna(row.get("resid_z")) else round(float(row["resid_z"]), 2),
        "underperforms_prediction": bool(row.get("ml_underperforms", False)),
        "meaning": "measured voltage is lower than a model that never saw this file predicts from conditions alone"}
    diag["eis_context"] = eis_context(ctx, curve_id)
    diag["agreement"] = ("physics and ML agree this curve is weak" if diag["flags"] and diag["ml_second_opinion"]["underperforms_prediction"]
                         else "physics flag only" if diag["flags"]
                         else "ML flag only" if diag["ml_second_opinion"]["underperforms_prediction"]
                         else "neither method flags this curve")
    if with_guidance and diag["kb_query"]:
        hits = ctx.retriever.search(diag["kb_query"], k=k)
        diag["guidance"] = [{"doc_id": h.doc_id, "title": h.chunk.title, "text": h.chunk.text,
                             "evidence": h.chunk.evidence} for h in hits]
    return diag


# ----------------------------------------------------------------------------- uploaded curves
def diagnose_uploaded_curve(ctx: Context, df: pd.DataFrame) -> dict:
    """Screen a user-supplied curve (columns current_density [mA/cm2], cell_voltage [V]).

    No peer conditions are known, so the curve is compared with *all* curves in the corpus
    using robust z-scores of its descriptors.  This is weaker than the peer comparison.
    """
    need = {"current_density", "cell_voltage"}
    if not need.issubset(df.columns):
        raise ValueError(f"CSV needs columns {sorted(need)}")
    d = df[list(need)].apply(pd.to_numeric, errors="coerce").dropna()
    d = d[d.current_density > 0]
    desc = physics.curve_descriptors(d)
    if desc is None:
        raise ValueError("need at least 6 valid points")
    ref = ctx.curves[ctx.curves.fit_ok]
    z = {}
    for zone, (col, direction) in physics.SCREEN_RULES.items():
        x = getattr(desc, col)
        med = ref[col].median()
        mad = (ref[col] - med).abs().median() * 1.4826 or 1.0
        z[zone] = float(direction * (x - med) / mad)
    flags = [zn for zn in physics.ZONE_INFO if z[zn] >= 1.5]
    zone = max(physics.ZONE_INFO, key=lambda zn: z[zn]) if flags else "none"
    info = physics.ZONE_INFO.get(zone)
    return {"descriptors": {k: (None if pd.isna(v) else round(float(v), 4)) for k, v in desc.as_dict().items()},
            "badness_z_vs_corpus": {k: round(v, 2) for k, v in z.items()}, "flags": flags,
            "dominant_zone": zone, "zone_label": info["label"] if info else "No zone stands out",
            "likely_component": info["component"] if info else "-",
            "parameters_to_check": info["parameters"] if info else [],
            "kb_query": info["kb_query"] if info else "",
            "loss_breakdown_V": {k: round(float(v), 4) for k, v in physics.breakdown_from_curve(d).items()},
            "caveat": "compared with the whole corpus (conditions unknown); a curve at low pressure or humidity "
                      "will look weak for that reason alone"}


# ----------------------------------------------------------------------------- what-if
def suggest_changes(ctx: Context, pressure: float, relative_humidity: float,
                    membrane_compression: float, nafion_percent: float) -> pd.DataFrame:
    """Single-variable moves (inside the measured design space) ranked by predicted gain in peak power."""
    s = ctx.surrogate
    base = {"pressure": pressure, "relative_humidity": relative_humidity,
            "membrane_compression": membrane_compression, "nafion_percent": nafion_percent}

    def pmax(cond):
        return float(s.predict_curve(**cond)["power_density"].max())
    p0 = pmax(base)
    rows = []
    for var in DESIGN_VARS:
        for level in s.levels[var]:
            if level == base[var]:
                continue
            c = dict(base, **{var: level})
            rows.append({"change": f"{var}: {base[var]:g} -> {level:g}", "pred_p_max": pmax(c),
                         "gain_mW_cm2": pmax(c) - p0})
    out = pd.DataFrame(rows).sort_values("gain_mW_cm2", ascending=False).reset_index(drop=True)
    out.attrs["baseline_p_max"] = p0
    out.attrs["surrogate_cv_mae_V"] = s.cv_mae_
    return out


def empirical_best(ctx: Context, top: int = 5) -> pd.DataFrame:
    """Best *measured* conditions in the design data, without any model (sanity check for the surrogate)."""
    t = ctx.curves.dropna(subset=DESIGN_VARS)
    return t.sort_values("p_max", ascending=False)[DESIGN_VARS + ["p_max", "j_at_0p6"]].head(top).reset_index(drop=True)


# ----------------------------------------------------------------------------- QA
def ask_knowledge(ctx: Context, question: str, k: int = 4, topic: str | None = None,
                  history: list[dict] | None = None) -> RagAnswer:
    return answer_question(question, ctx.retriever, ctx.llm, k=k, topic=topic, history=history)


# ----------------------------------------------------------------------------- generative report
REPORT_SYSTEM = """You write the short 'Summary' paragraph of a PEM fuel-cell troubleshooting report.
Use ONLY the JSON facts provided. Max 5 sentences. Say what zone is weak, the evidence, what second measurement
would confirm it, and one cautious next step. Do not add numbers, causes, DOIs or citations that are not in the JSON."""


def troubleshooting_report(ctx: Context, curve_id: int) -> tuple[str, "Usage"]:
    """Markdown report for one curve: facts assembled by code, an optional LLM summary on top."""
    from .llm import LLMUnavailable, Usage
    d = diagnose_curve(ctx, curve_id, k=3)
    c = d["conditions"]
    lines = [f"# Troubleshooting report: curve {curve_id}", "",
             f"**Test family:** {c['test_family']}  ", f"**Conditions:** pressure {c['pressure']}, RH {c['relative_humidity']}, "
             f"compression {c['membrane_compression']}, Nafion {c['nafion_percent']}, activation set {c['operating_set']}", ""]
    summary, usage = None, Usage()
    if ctx.llm is not None and getattr(ctx.llm, "available", False):
        facts = {k: d[k] for k in ("zone_label", "likely_component", "flags", "severity_z", "descriptors",
                                   "loss_breakdown_V", "ml_second_opinion", "agreement", "eis_context")}
        try:
            r = ctx.llm.chat([{"role": "system", "content": REPORT_SYSTEM},
                              {"role": "user", "content": __import__("json").dumps(facts, default=str)}])
            summary, usage = r.text.strip(), r.usage
        except LLMUnavailable:
            pass
    lines += ["## Summary", summary or f"{d['zone_label']}. Physics and ML: {d['agreement']}.", "",
              "## Evidence", f"* Severity (z against peers): {d['severity_z']}",
              f"* Loss budget at j = {d['loss_breakdown_V'].get('j_ref', float('nan')):.0f} mA/cm2 (V): kinetic "
              f"{d['loss_breakdown_V'].get('kinetic', float('nan')):.3f}, ohmic {d['loss_breakdown_V'].get('ohmic', float('nan')):.3f}, "
              f"mass transport {d['loss_breakdown_V'].get('mass_transport', float('nan')):.3f}",
              f"* Peak power {d['descriptors']['p_max']} mW/cm2; apparent resistance {d['descriptors']['asr_ohm_cm2']} ohm cm2"]
    if d.get("eis_context"):
        lines.append(f"* Impedance proxies: HFR {d['eis_context']['hfr_proxy_median_ohm']:.3f} ohm, arc width "
                     f"{d['eis_context']['arc_width_median_ohm']:.3f} ohm (no frequency axis)")
    lines += ["", "## Parameters to check", ", ".join(d["parameters_to_check"]) or "none identified", "",
              "## Troubleshooting notes (sources)"]
    for g in d.get("guidance", []):
        lines.append(f"* **[{g['doc_id']}] {g['title']}** ({g['evidence']}): {g['text'].split('. ')[0]}.")
    lines += ["", "## Possible improvements"]
    if all(c[k] is not None for k in DESIGN_VARS):
        try:
            sug = suggest_changes(ctx, *(float(c[k]) for k in DESIGN_VARS)).head(3)
            lines += [f"* {r.change}: predicted {r.gain_mW_cm2:+.0f} mW/cm2 (surrogate, interpolation within measured design)"
                      for r in sug.itertuples()]
        except ValueError as exc:
            lines.append(f"* surrogate not applicable: {exc}")
    else:
        lines.append("* Conditions of this file are not in the surrogate's design; use the troubleshooting notes above.")
    lines += ["", "## Confirm before acting",
              "A polarisation curve narrows the cause but does not prove it. Confirm with EIS (HFR and arc), a crossover "
              "measurement, CV/ECSA, or a repeat run. Data: Pt/C cathodes in oxygen only (see note K26).", ""]
    return "\n".join(lines), usage


def ask_both(ctx: Context, question: str, k: int = 4) -> dict:
    """Same question answered offline (extractive) and live (LLM), for side-by-side comparison."""
    import dataclasses
    offline = answer_question(question, ctx.retriever, None, k=k)
    live = None
    if ctx.llm is not None and getattr(ctx.llm, "available", False):
        live = answer_question(question, ctx.retriever, ctx.llm, k=k)
    return {"offline": offline, "live": live}


# ----------------------------------------------------------------------------- template API
def load_data() -> pd.DataFrame:
    """Template name for `load_reviews`: all polarisation points with a `curve_id` (see src/data.py)."""
    return load_polarization()


def compare_models(train: pd.DataFrame, n_splits: int = 5, models: dict | None = None) -> pd.DataFrame:
    """Grouped CV of every approach on the same folds; returns a tidy table with training time."""
    from .models import compare_models as _compare
    return _compare(train, n_splits, models)


def build_index(chunk_words: int | None = None, kind: str = "hybrid", llm=None):
    """Chunk the knowledge notes (with metadata) and build the retriever. Returns the retriever."""
    return build_retriever(kind, chunk_words=chunk_words, llm=llm)


def evaluate(ctx: Context, questions: list[dict] | None = None, k: int = 4) -> pd.DataFrame:
    """Run the whole evaluation set: one row per question (hit, cites, refused, tokens, cost, seconds)."""
    from .evaluation import load_questions, qa_eval
    return qa_eval(ctx, questions or load_questions(), k=k)


_REFUSAL_HINTS = ("don't know", "do not know", "cannot", "can't", "no information", "not able", "unable", "not sure")


def llm_alone_eval(questions: list[dict], llm) -> pd.DataFrame | None:
    """LLM-track baseline: the model answers with NO retrieval and NO tools.

    Returns None when no live model is connected. `refused` is a keyword check on the reply, and
    `cites_note` is always False because the model has no sources to cite.
    """
    if llm is None or not getattr(llm, "available", False):
        return None
    rows = []
    for q in questions:
        r = llm.chat([{"role": "system", "content": "You are a PEM fuel-cell expert. Answer in at most 4 sentences. "
                       "If you do not know, say so."}, {"role": "user", "content": q["question"]}])
        rows.append({"id": q["id"], "answerable": q["answerable"], "answer": r.text.strip(),
                     "refused": any(h in r.text.lower() for h in _REFUSAL_HINTS), "cites_note": False,
                     "prompt_tokens": r.usage.prompt_tokens, "completion_tokens": r.usage.completion_tokens,
                     "cost_usd": r.usage.cost_usd, "seconds": r.usage.seconds})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- DFT layer
DFT_TABLE_PATH = PROJECT_ROOT / "data" / "dft" / "dft_table.csv"


def get_dft(ctx: Context, rebuild: bool = False) -> dict:
    """The DFT table and the screener, cached in ctx.extras.

    Source: data/dft/dft_table.csv if you supplied one (validated, provenance kept); otherwise the SYNTHETIC
    demo table, which switches every recommendation off.
    """
    if "dft" in ctx.extras and not rebuild:
        return ctx.extras["dft"]
    from . import dft as D
    from .dft_ml import Screener
    if DFT_TABLE_PATH.exists():
        table, source = D.load_dft_table(DFT_TABLE_PATH), "user table: data/dft/dft_table.csv"
    else:
        table = D.add_orr_metrics(D.make_demo_table())
        table["chemical_system"] = table["composition"].map(D.chemical_system)
        source = "SYNTHETIC DEMO TABLE (not DFT)"
    table = table.join(D.build_feature_table(table)[["pt_fraction"]])
    ctx.extras["dft"] = {"table": table, "screener": Screener(table), "source": source,
                         "trusted": bool(table["provenance"].isin(D.TRUSTED).mean() >= 0.5)}
    return ctx.extras["dft"]


def dft_rank(ctx: Context, compositions: list[str], facet: str = "111", structure: str = "skin") -> pd.DataFrame:
    """Score candidates (no DFT values needed) with the screener; recommendations are off for demo data."""
    d = get_dft(ctx)
    cand = pd.DataFrame({"composition": compositions, "facet": facet, "structure": structure})
    out = d["screener"].score(cand)
    from . import dft as D
    flags = []
    for r in out.itertuples():
        feat = D.composition_features(r.composition)
        flags.append(D.stability_flags(pd.Series({"composition": r.composition, "structure": structure, **feat})))
    out["stability_flags"] = flags
    return out
