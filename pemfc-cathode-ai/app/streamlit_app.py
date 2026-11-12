"""PEMFC Cathode AI: diagnosis, what-if, RAG chatbot, review analytics and R&D watch.

Run from the project root:  streamlit run app/streamlit_app.py
Keys: OPEN_AI_KEY / OPEN_AI_MODEL from .env, st.secrets, or typed in the sidebar (session only).
Without a key every tab still works in offline mode.
"""
import dataclasses
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src import functions as F
from deck import DEMO_BUDGET_S, SLIDE_BUDGET_S, SLIDES
from src.agent import run_agent
from src.config import EVAL_DIR, get_llm_settings
from src.evaluation import load_questions
from src.llm import LLMClient
from src.vectorstore import PineconeRetriever, PineconeStore, VectorStoreUnavailable, get_pinecone_settings
from src.bootstrap import copy_ecsim, download_ecsim, git_hygiene, raw_data_ok
from src.config import MODELS_DIR
from src.rd_monitor import (DEFAULT_QUERIES, build_digest, dedupe, fetch_crossref, load_watchlist, triage,
                            verify_digest)
from src import dft as D
from src.dft_ml import suggest_next_dft
from src.reviews import ReviewStore, analyze_comment, summarise_comments

st.set_page_config(page_title="PEMFC Cathode AI", page_icon="⚡", layout="wide")


# ----------------------------------------------------------------------------- setup
@st.cache_resource(show_spinner="Loading data, models and knowledge base (first run builds them)...")
def load_ctx():
    return F.get_context()


def secret(name: str) -> str | None:
    try:
        return st.secrets.get(name)
    except Exception:
        return None



# ----------------------------------------------------------------------------- first run (fresh clone)
if not raw_data_ok():
    st.title("⚡ PEMFC Cathode AI: first run")
    st.info("The ECSIM dataset (CC BY 4.0, Hamidi, Haghighi & Askari 2020, doi:10.26434/chemrxiv.11902023) is not in git, "
            "so a fresh clone has to fetch it once. Nothing here needs an API key.")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Download it")
        if st.button("Download ECSIM dataset (about 4 MB)"):
            with st.spinner("Downloading from GitHub..."):
                try:
                    n = download_ecsim()
                    st.success(f"{n} CSV files ready. Reloading...")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Download failed ({type(exc).__name__}). Offline? Use the other option, or run `python download_data.py`.")
    with c2:
        st.subheader("Or copy an existing copy")
        folder = st.text_input("Folder containing the raw CSV folders (e.g. Project 2's data/raw)")
        if st.button("Copy from this folder") and folder:
            try:
                st.success(f"{copy_ecsim(folder)} CSV files copied. Reloading...")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
    st.stop()

base_ctx = load_ctx()
with st.sidebar:
    st.header("Live LLM  ⇄  Offline")
    cfg = get_llm_settings()
    key = st.text_input("OPEN_AI_KEY", type="password", value="",
                        help="Kept in this browser session only. Empty = use .env / secrets, else offline.")
    model = st.text_input("OPEN_AI_MODEL", value=secret("OPEN_AI_MODEL") or cfg.model)
    mode = st.radio("Mode", ["Auto (live if the key works)", "Offline only", "Side by side (offline + live)"], index=0)
    llm = LLMClient(get_llm_settings(api_key=key or secret("OPEN_AI_KEY"), model=model))
    if st.button("Test live connection"):
        ok, msg = llm.ping()
        (st.success if ok else st.error)(msg)
    live_ok = llm.available and not mode.startswith("Offline")
    st.markdown(f"**Now:** {'🟢 LIVE · ' + llm.model if live_ok else '⚪ OFFLINE (extractive answers, keyword-routed agent)'}")
    if mode.startswith("Side") and not llm.available:
        st.warning("Side-by-side needs a working key; showing offline only.")
    st.divider()
    st.header("Vector store")
    vs_mode = st.radio("Retrieval backend", ["Local (default)", "Pinecone"], index=0,
                       help="Local needs nothing. Pinecone stores the knowledge notes in your own index; falls back to local if it fails.")
    pc_store = None
    if vs_mode == "Pinecone":
        pkey = st.text_input("PINECONE_API_KEY", type="password", value="", help="Session only. Empty = environment / secrets.")
        pidx = st.text_input("PINECONE_INDEX_NAME", value=secret("PINECONE_INDEX_NAME") or get_pinecone_settings().index_name)
        cache_key = (bool(pkey), pidx)
        if st.session_state.get("pc_key") != cache_key:
            st.session_state["pc_store"] = PineconeStore(get_pinecone_settings(pkey or secret("PINECONE_API_KEY"), pidx))
            st.session_state["pc_key"] = cache_key
        pc_store = st.session_state["pc_store"]
        st.caption(pc_store.status())
        if pc_store.available and st.button("Index the notes in Pinecone"):
            with st.spinner("Creating the index if needed and uploading the notes..."):
                try:
                    st.success(f"{pc_store.index_notes(wait_s=5)} chunks indexed (searchable after a few seconds)")
                except VectorStoreUnavailable as exc:
                    st.error(f"Pinecone failed: {exc}. The app keeps using the local retriever.")
    remember = st.toggle("Remember my comments in Pinecone (this session only)", value=False,
                         disabled=not (pc_store and pc_store.available),
                         help="Stored in a namespace only this browser session knows; delete button in the Comments tab.")
    st.session_state.setdefault("session_id", PineconeStore.new_session_id())
    with st.expander("Keys and Git check"):
        h = git_hygiene()
        st.write("Key source: " + ("typed here" if key else "environment / .env / secrets" if cfg.has_key or secret("OPEN_AI_KEY") else "none (offline)"))
        for name, ok_ in h["ignored"].items():
            st.write(("✅" if ok_ else "❌") + f" `{name}` is git-ignored")
        st.caption("`.env` present: " + str(h[".env_file_present"]) + ". Copy `.env.example` to `.env` for a persistent key; "
                   "the key typed above lives only in this browser session.")
    st.divider()
    st.caption("Data: ECSIM PEM dataset, Hamidi et al. 2020, doi:10.26434/chemrxiv.11902023, CC BY 4.0. "
               "Pt/C cathodes, H2/O2. A screening aid, not a safety or design authority.")
# comments live in this browser session only (public-safe); set PERSIST_COMMENTS=1 for a single-user machine
import os as _os
if "review_store" not in st.session_state:
    st.session_state["review_store"] = ReviewStore(in_memory=_os.getenv("PERSIST_COMMENTS") != "1")
retr = PineconeRetriever(pc_store, base_ctx.retriever) if (pc_store is not None and pc_store.available) else base_ctx.retriever
base_ctx = dataclasses.replace(base_ctx, retriever=retr, store=st.session_state["review_store"])
ctx = dataclasses.replace(base_ctx, llm=llm if live_ok else None)
side_by_side = mode.startswith("Side") and llm.available
ctx_off = dataclasses.replace(base_ctx, llm=None)

st.title("⚡ PEMFC Cathode Catalyst Intelligence")
st.caption("DFT-guided catalyst screening with uncertainty, sourced explanations, and device-level failure diagnosis.")

tabs = st.tabs(["Presentation", "Live demo", "DFT discovery", "Overview", "Diagnose a curve", "What-if", "Chatbot agent", "Knowledge Q&A (RAG)",
                "Comments & reviews", "R&D watch", "Evaluation"])


def show_usage(u):
    if u is not None and getattr(u, "calls", 0):
        c = "price unknown" if u.cost_usd is None else f"${u.cost_usd:.5f}"
        st.caption(f"LLM usage: {u.calls} call(s), {u.prompt_tokens}+{u.completion_tokens} tokens, {u.seconds:.1f}s, {c}")


def plot_curve(points: pd.DataFrame, title: str = "", extra: pd.DataFrame | None = None):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(points.current_density, points.cell_voltage, "o-", label="measured")
    if extra is not None:
        ax.plot(extra.current_density, extra.cell_voltage, "--", label="model")
    ax.set_xlabel("current density (mA/cm²)")
    ax.set_ylabel("cell voltage (V)")
    ax2 = ax.twinx()
    ax2.plot(points.current_density, points.current_density * points.cell_voltage, ":", color="tab:red")
    ax2.set_ylabel("power density (mW/cm²)", color="tab:red")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    st.pyplot(fig)
    plt.close(fig)


# ----------------------------------------------------------------------------- presentation
def fmt(sec):
    return f"{int(sec // 60)}:{int(sec % 60):02d}"


def _go(delta):
    st.session_state.slide = max(0, min(len(SLIDES) - 1, st.session_state.slide + delta))


def _start():
    st.session_state.t0 = time.time()


def _reset():
    st.session_state.t0 = None
    st.session_state.slide = 0


with tabs[0]:
    st.session_state.setdefault("slide", 0)
    st.session_state.setdefault("t0", None)
    sl = SLIDES[st.session_state.slide]
    planned_end = sum(x["secs"] for x in SLIDES[: st.session_state.slide + 1])
    b1, b2, b3, b4, b5 = st.columns([1, 1, 1, 1, 4])
    b1.button("◀ Prev", on_click=_go, args=(-1,), disabled=st.session_state.slide == 0, use_container_width=True)
    b2.button("Next ▶", on_click=_go, args=(1,), disabled=st.session_state.slide == len(SLIDES) - 1, use_container_width=True)
    b3.button("Start timer", on_click=_start, use_container_width=True)
    b4.button("Reset", on_click=_reset, use_container_width=True)
    elapsed = time.time() - st.session_state.t0 if st.session_state.t0 else None
    b5.caption(f"Slide {sl['n']} of {len(SLIDES)} · plan: finish by {fmt(planned_end)} of {fmt(SLIDE_BUDGET_S)} · "
               + (f"elapsed {fmt(elapsed)} (updates when you click)" if elapsed is not None else "timer not started"))
    st.progress((st.session_state.slide + 1) / len(SLIDES))
    if sl["kicker"]:
        st.markdown(f"<p style='color:#00776F;font-weight:700;letter-spacing:2px;margin:0'>{sl['kicker']}</p>", unsafe_allow_html=True)
    size = "3.2rem" if sl["kind"] == "title" else "1.9rem"
    st.markdown(f"<h1 style='font-size:{size};line-height:1.15;margin:0 0 .6rem 0'>{sl['title']}</h1>", unsafe_allow_html=True)
    if sl["kind"] == "title":
        st.markdown(f"<p style='font-size:1.4rem;color:#51606F'>{sl['sub']}</p><p><b>Theophilus Kobina Sarpey</b></p>", unsafe_allow_html=True)
    elif sl["kind"] == "flow":
        cols = st.columns(len(sl["flow"]) * 2 - 1)
        for i, label in enumerate(sl["flow"]):
            cols[i * 2].markdown(f"<div style='background:#fff;border:1px solid #DCE4E8;border-radius:12px;padding:14px;text-align:center;"
                                 f"font-weight:700'>{label}</div>", unsafe_allow_html=True)
            if i < len(sl["flow"]) - 1:
                cols[i * 2 + 1].markdown("<div style='font-size:2rem;color:#00776F;text-align:center'>→</div>", unsafe_allow_html=True)
    else:
        cols = st.columns(len(sl["cards"]))
        for col, (hd, tx, big) in zip(cols, sl["cards"]):
            body = (f"<div style='font-size:2.6rem;font-weight:700;color:#00776F;line-height:1.1'>{big}</div>" if big else "")
            col.markdown(f"<div style='background:#fff;border:1px solid #DCE4E8;border-radius:14px;padding:18px;height:100%'>{body}"
                         f"<div style='font-size:1.2rem;font-weight:700;margin:.3rem 0'>{hd}</div>"
                         f"<div style='font-size:1.05rem;color:#51606F'>{tx}</div></div>", unsafe_allow_html=True)
    if sl.get("foot"):
        st.markdown(f"<p style='font-size:1.05rem;color:#51606F;margin-top:1rem'>{sl['foot']}</p>", unsafe_allow_html=True)
    with st.expander(f"Speaker notes · {sl['secs']} s"):
        st.write(sl["notes"])
    if st.session_state.slide == len(SLIDES) - 1:
        st.success("Next: open the **Live demo** tab (3 minutes).")

# ----------------------------------------------------------------------------- live demo
def dft_banner(info):
    if not info["trusted"]:
        st.error("SYNTHETIC DEMO DATA: " + info["source"] + ". These numbers are not DFT results. Rankings are shown to demonstrate the "
                 "method; the product gives **no recommendation** until a real DFT table (data/dft/dft_table.csv or the upload below) is loaded.")
    else:
        st.success("Data source: " + info["source"])


def free_energy_plot(row):
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for U, st_ in ((0.0, "-"), (float(row["U_L"]), "--")):
        g = D.free_energy_diagram(row, U)
        ax.step(range(5), g.G_eV, where="mid", ls=st_, label=f"U = {U:.2f} V")
    ax.set_xticks(range(5)); ax.set_xticklabels(["O2", "*OOH", "*O", "*OH", "H2O"]); ax.set_ylabel("free energy (eV)")
    ax.set_title(f"{row['composition']}({row['facet']}): U_L = {row['U_L']:.2f} V, step {int(row['limiting_step'])} limits")
    ax.legend(); ax.grid(alpha=.3)
    return fig


DEMO_COMMENT_A = "Resistance climbed during the run and the membrane seemed dry."
DEMO_COMMENT_B = "Resistance climbed during the run and the membrane seemed wet."
DEMO_Q = {"Answered with sources": "How can I tell flooding from membrane dehydration?",
          "Must refuse (out of scope)": "Who won the football world cup in 2018?",
          "Look-alike, offline answers it (live should refuse)": "What activation energy for ORR did the ECSIM authors report for their Pt/C electrodes?"}


def _rag_pair(question, k=4):
    off = F.ask_knowledge(ctx_off, question, k=k)
    live = F.ask_knowledge(ctx, question, k=k) if llm.available and live_ok else None
    return off, live


def _show_rag(question):
    off, live = _rag_pair(question)
    c1, c2 = st.columns(2)
    c1.markdown("**⚪ Offline (extractive)**")
    c1.write(off.answer)
    c2.markdown(f"**🟢 Live · {llm.model}**" if live else "**🟢 Live**")
    c2.write(live.answer if live else "Not connected: paste a key in the sidebar and click *Test live connection*. "
                                      "The demo still works offline.")
    if live:
        show_usage(live.usage)
    for i, h in enumerate((live or off).hits, 1):
        with st.expander(f"[S{i}] {h.doc_id} · {h.chunk.title} · evidence: {h.chunk.evidence}"):
            st.write(h.chunk.text)


with tabs[1]:
    st.caption("Examples are preloaded: nothing to type. Budget 3:00. If something breaks, say what it should have shown and move on.")
    with st.expander("Preflight (run before your slot)", expanded=False):
        have_cache = all((MODELS_DIR / f).exists() for f in ("curve_corpus.csv", "design_surrogate.joblib", "review_models.joblib"))
        st.write(("✅" if raw_data_ok() else "❌") + " dataset present")
        st.write(("✅" if have_cache else "⚠️") + " models cached in models/ (built on first run, about a minute)")
        st.write(("🟢 live" if live_ok else "⚪ offline") + f" · model {llm.model}")
        if st.button("Warm up: run every demo step once"):
            with st.spinner("Warming up..."):
                fl = F.list_curves(ctx, family="Nafion", flagged_only=True, limit=1)
                F.diagnose_curve(ctx, int(fl.curve_id.iloc[0]))
                F.dft_rank(base_ctx, ["Pt3Ni", "Pt3Co", "Pt3Ru"])
                for q_ in DEMO_Q.values():
                    _rag_pair(q_)
                analyze_comment(DEMO_COMMENT_A, ctx.review_models, None)
            st.success("Warm. First clicks will now be fast.")
    step = st.radio("Step", ["1 DFT screen", "2 Explain and refuse", "3 Offline vs live", "4 Device failure (ECSIM)", "5 Change one word"], horizontal=True)
    if step.startswith("1"):
        info_d = F.get_dft(base_ctx)
        dft_banner(info_d)
        res = F.dft_rank(base_ctx, ["Pt3Ni", "Pt3Co", "Pt3Ru"], "111", "skin")
        st.dataframe(res[["rank", "composition", "pred_U_L", "pred_sd", "confidence", "outside_training_distribution", "recommendation"]].round(3),
                     hide_index=True, use_container_width=True)
        tbl_d = info_d["table"]; row_d = tbl_d.sort_values("U_L", ascending=False).iloc[0]
        l, r = st.columns(2)
        l.pyplot(free_energy_plot(row_d)); plt.close("all")
        r.markdown("**Say:** the model predicts the limiting potential from cheap features, with an uncertainty. Pt3Ru is a new chemistry for this table: "
                   "flagged *outside the training distribution*, so the next step is DFT, not synthesis.")
    elif step.startswith("2"):
        qd = st.radio("Question", ["Explain a DFT concept (sourced)", "Next calculation", "Must refuse (invented value)"], horizontal=True)
        qtext = {"Explain a DFT concept (sourced)": "Why do the OOH and OH adsorption energies limit how good a catalyst can be?",
                 "Next calculation": "Which DFT calculation should I run next for a promising Pt alloy?",
                 "Must refuse (invented value)": "What is the DFT-calculated OH adsorption energy of Pt3Ni(111) in database entry 12345?"}[qd]
        _show_rag(qtext)
        st.caption("Notes are labelled textbook / literature; no value is invented. The last question must be refused (look-alike: offline mode may answer it, live should not).")
    elif step.startswith("3"):
        _show_rag(DEMO_Q["Answered with sources"])
        st.caption("Same sources, same citation check. Open a source to show where each claim comes from.")
    elif step.startswith("4"):
        fl = F.list_curves(ctx, family="Nafion", flagged_only=True, limit=1)
        cid = int(fl.curve_id.iloc[0])
        d = F.diagnose_curve(ctx, cid)
        l, r = st.columns(2)
        with l:
            plot_curve(F.curve_points(ctx, cid), f"Curve {cid}: flagged by the physics screening")
        with r:
            st.subheader(d["zone_label"])
            st.write(f"**Likely component:** {d['likely_component']}")
            lb = d["loss_breakdown_V"]
            st.bar_chart(pd.DataFrame({"loss (V)": [lb["kinetic"], lb["ohmic"], lb["mass_transport"]]}, index=["kinetic", "ohmic/linear", "mass transport"]))
            st.info("A better catalyst (lower overpotential) only helps the kinetic zone; ohmic and mass-transport losses need membrane, ionomer and water-management fixes.")
    else:
        ca, cb = st.columns(2)
        ta = ca.text_area("Original comment", DEMO_COMMENT_A)
        tb = cb.text_area("Change one word", DEMO_COMMENT_B)
        for col, t in ((ca, ta), (cb, tb)):
            an = analyze_comment(t, ctx.review_models, None)
            col.metric("Sentiment", an["sentiment"], f"confidence {an['sentiment_confidence']:.2f}")
            col.write(f"Top aspect: **{an['top_aspect']}** · suspected zone: **{an['suspected_zone']}** · symptoms: {an['symptoms'] or 'none'}")
        st.caption("Trained on a synthetic corpus: tags are suggestions.")

# ----------------------------------------------------------------------------- DFT discovery
with tabs[2]:
    info = F.get_dft(base_ctx)
    up = st.file_uploader("Load your own DFT table (CSV, see data/dft/README.md)", type="csv", key="dft_up")
    if up is not None:
        try:
            tmp = ROOT / "models" / "_dft_upload.csv"
            tmp.write_bytes(up.getvalue())
            tbl = D.load_dft_table(tmp)
            tbl = tbl.join(D.build_feature_table(tbl)[["pt_fraction"]])
            from src.dft_ml import Screener
            info = {"table": tbl, "screener": Screener(tbl), "source": f"uploaded table ({len(tbl)} rows)",
                    "trusted": bool(tbl["provenance"].isin(D.TRUSTED).mean() >= 0.5)}
            base_ctx.extras["dft"] = info
        except Exception as exc:
            st.error(f"Table rejected: {exc}")
    dft_banner(info)
    tbl = info["table"]
    page = st.radio("DFT page", ["Catalyst explorer", "Descriptor explorer", "AI catalyst predictor", "Failure intelligence", "Next DFT calculation"],
                    horizontal=True)
    if page == "Catalyst explorer":
        comp = st.selectbox("Composition", sorted(tbl.composition.unique()))
        sub = tbl[tbl.composition == comp]
        st.dataframe(sub[["system_id", "facet", "structure", "E_OH", "E_O", "E_OOH", "dG_OH", "U_L", "eta", "provenance", "reference"]].round(3),
                     hide_index=True, use_container_width=True)
        row = sub.sort_values("U_L", ascending=False).iloc[0]
        l, r = st.columns(2)
        l.pyplot(free_energy_plot(row)); plt.close("all")
        r.write(f"**Value types:** `provenance` says where each number comes from: " + ", ".join(sorted(sub.provenance.unique())) + ".")
        r.write("U_L is derived by formula from the three adsorption energies (computational hydrogen electrode); it is not a measurement.")
    elif page == "Descriptor explorer":
        l, r = st.columns(2)
        fig, ax = plt.subplots(figsize=(5.2, 3.8))
        x = np.linspace(tbl.dG_OH.min() - .2, tbl.dG_OH.max() + .2, 200)
        ax.plot(x, D.volcano_curve(x), "k--", label="OOH-OH volcano (offset 3.2 eV)")
        for sysname, g in tbl.groupby("chemical_system"):
            ax.scatter(g.dG_OH, g.U_L, s=14, alpha=.7, label=sysname if len(tbl.chemical_system.unique()) <= 12 else None)
        ax.set(xlabel="dG_OH (eV)", ylabel="U_L (V)", title="Volcano"); ax.legend(fontsize=6); ax.grid(alpha=.3)
        l.pyplot(fig); plt.close(fig)
        sc = D.fit_scaling(tbl)
        fig, ax = plt.subplots(figsize=(5.2, 3.8))
        ax.scatter(tbl.dG_OH, tbl.dG_OOH, s=14, alpha=.7); xx = np.linspace(tbl.dG_OH.min(), tbl.dG_OH.max(), 20)
        ax.plot(xx, xx + D.OOH_OH_OFFSET, "k--", label="dG_OOH = dG_OH + 3.2"); ax.plot(xx, sc["slope"] * xx + sc["intercept"], label="fit")
        ax.set(xlabel="dG_OH (eV)", ylabel="dG_OOH (eV)", title="Scaling relation"); ax.legend(fontsize=7); ax.grid(alpha=.3)
        r.pyplot(fig); plt.close(fig)
        st.write(f"Scaling fit: slope {sc['slope']:.2f}, intercept {sc['intercept']:.2f} eV, RMSE {sc['rmse']:.2f} eV, R² {sc['r2']:.2f} (n = {sc['n']}).")
        if "d_band_center" in tbl and tbl.d_band_center.notna().any():
            st.write(f"Correlation of d-band centre with dG_OH: {tbl[['d_band_center', 'dG_OH']].corr().iloc[0, 1]:.2f}")
    elif page == "AI catalyst predictor":
        c1, c2, c3 = st.columns([3, 1, 1])
        comps = [c.strip() for c in c1.text_input("Candidates (comma separated)", "Pt3Ni, Pt3Co, Pt3Ru").split(",") if c.strip()]
        facet = c2.selectbox("Facet", ["111", "100", "110"])
        structure = c3.selectbox("Structure", ["skin", "bulk"])
        try:
            res = F.dft_rank(base_ctx, comps, facet, structure)
            show = res[["rank", "composition", "pred_U_L", "pred_sd", "confidence", "outside_training_distribution", "recommendation"]].round(3)
            st.dataframe(show, hide_index=True, use_container_width=True)
            st.caption("Every number here is **ml_predicted**: a model's estimate from composition features, not a calculated or measured value.")
        except ValueError as exc:
            st.error(str(exc))
        with st.expander("How good is this model? (leave-one-system-out)"):
            from src.dft_ml import evaluate_baselines, loso_evaluate
            st.dataframe(evaluate_baselines(tbl).round(3), hide_index=True)
            st.dataframe(loso_evaluate(tbl)[0].round(3), hide_index=True)
    elif page == "Failure intelligence":
        comp = st.selectbox("Composition", sorted(tbl.composition.unique()), key="fail_comp")
        row = tbl[tbl.composition == comp].iloc[0]
        flags = D.stability_flags(pd.Series({**row.to_dict(), **D.composition_features(comp)}))
        if flags:
            for f_ in flags:
                st.warning(f"**{f_['mechanism']}**: {f_['detail']} _(evidence: {f_['evidence']})_")
        else:
            st.success("No stability heuristic fires for this composition (this is not proof of stability).")
        st.info("Link to the device: the ECSIM polarisation data show where a Pt/C cell loses voltage (kinetic, ohmic, mass transport). "
                "A good U_L addresses only the kinetic zone; see the *Diagnose a curve* tab.")
    else:
        comps = [c.strip() for c in st.text_input("Candidate pool", "Pt3Ni, Pt3Co, Pt3Fe, Pt3Ru, PtNi, PtCo").split(",") if c.strip()]
        k = st.slider("How many calculations can you afford?", 1, 5, 3)
        try:
            res = F.dft_rank(base_ctx, comps)
            st.dataframe(suggest_next_dft(res, k).round(3), hide_index=True, use_container_width=True)
            st.caption("Upper confidence bound = predicted U_L + 1 x uncertainty: the next DFT calculation should be where it can teach the model most.")
            st.markdown("**Checklist for the chosen candidate:** converge cutoff / k-points / slab, relax with the bottom layers fixed, OH, O and OOH on several sites, "
                        "the same calculation for Pt(111) as a reference, a functional check, then coverage, water and a segregation / dissolution estimate.")
        except ValueError as exc:
            st.error(str(exc))

# ----------------------------------------------------------------------------- overview
with tabs[3]:
    c = ctx.curves
    a, b, d, e = st.columns(4)
    a.metric("Polarisation curves", len(c))
    b.metric("Flagged by physics", int(c.dominant_zone.isin(["kinetic", "ohmic", "mass_transport"]).sum()))
    d.metric("Flagged by ML residual", int(c.ml_underperforms.sum()))
    e.metric("Knowledge notes", len(ctx.retriever.chunks))
    st.markdown(
        "**How it works.** Each curve is split into kinetic, ohmic and mass-transport zones and compared with "
        "curves measured at the same pressure and humidity. A gradient-boosting model that never saw the file "
        "gives a second opinion. A retrieval system (RAG) pulls explanations from curated notes, and an agent "
        "can chain these tools. All of it runs offline; the LLM only adds language and tool choice.")
    st.markdown("**Scope.** The data contain only Pt/C cathodes in oxygen, so advice on other catalysts or on air "
                "operation comes from literature notes and is labelled as untested here.")
    zc = c.dominant_zone.value_counts().rename_axis("zone").reset_index(name="curves")
    st.bar_chart(zc, x="zone", y="curves")

# ----------------------------------------------------------------------------- diagnose
with tabs[4]:
    mode = st.radio("Source", ["Measured curve from the dataset", "Upload my own curve (CSV)"], horizontal=True)
    diag, pts = None, None
    if mode.startswith("Measured"):
        fam = st.selectbox("Test family", sorted(ctx.curves.test_family.unique()))
        sub = ctx.curves[ctx.curves.test_family == fam]
        flagged_first = sub.sort_values("dominant_zone", key=lambda s: s.isin(["none", "unreliable_fit"]))
        labels = {int(r.curve_id): f"curve {int(r.curve_id)} · P={r.pressure} RH={r.relative_humidity} "
                                   f"set={r.operating_set} · {r.dominant_zone}" for r in flagged_first.itertuples()}
        cid = st.selectbox("Curve", list(labels), format_func=labels.get)
        diag, pts = F.diagnose_curve(ctx, cid), F.curve_points(ctx, cid)
    else:
        st.caption("CSV with columns `current_density` (mA/cm²) and `cell_voltage` (V). Do not upload confidential data if an LLM is on.")
        up = st.file_uploader("CSV", type="csv")
        if up is not None:
            try:
                pts = pd.read_csv(up).sort_values("current_density")
                diag = F.diagnose_uploaded_curve(ctx, pts)
                diag["guidance"] = [{"doc_id": h.doc_id, "title": h.chunk.title, "text": h.chunk.text, "evidence": h.chunk.evidence}
                                    for h in ctx.retriever.search(diag["kb_query"], k=3)] if diag["kb_query"] else []
            except Exception as exc:
                st.error(str(exc))
    if diag is not None:
        l, r = st.columns([1, 1])
        with l:
            plot_curve(pts, "Polarisation and power curve")
        with r:
            st.subheader(diag["zone_label"])
            st.write(f"**Likely component:** {diag['likely_component']}")
            if diag["parameters_to_check"]:
                st.write("**Parameters to check:** " + ", ".join(diag["parameters_to_check"]))
            lb = diag["loss_breakdown_V"]
            loss = pd.DataFrame({"loss (V)": [lb["kinetic"], lb["ohmic"], lb["mass_transport"]]},
                                index=["kinetic", "ohmic/linear", "mass transport"])
            st.bar_chart(loss)
            st.caption(f"Voltage deficit at j = {lb['j_ref']:.0f} mA/cm² splits into these three parts.")
            if "agreement" in diag:
                st.info(f"Physics vs ML: {diag['agreement']}")
            if diag.get("eis_context"):
                st.write(f"Impedance proxies for the same conditions: HFR ≈ {diag['eis_context']['hfr_proxy_median_ohm']:.3f} Ω, "
                         f"arc width ≈ {diag['eis_context']['arc_width_median_ohm']:.3f} Ω (no frequency axis in the files).")
            if "caveat" in diag:
                st.warning(diag["caveat"])
        with st.expander("Descriptors"):
            st.json(diag["descriptors"])
        st.subheader("Troubleshooting guidance (retrieved notes)")
        for g in diag.get("guidance", []):
            st.markdown(f"**[{g['doc_id']}] {g['title']}** · evidence: {g['evidence']}")
            st.write(g["text"])
        if ctx.llm is not None and getattr(ctx.llm, "available", False) and st.button("Explain with the LLM"):
            res = run_agent(ctx, f"Explain this diagnosis and what to do next: {diag['zone_label']}. "
                                 + (f"Use curve {diag['curve_id']}." if "curve_id" in diag else "Uploaded curve; descriptors: " + str(diag["descriptors"])))
            st.write(res.answer)
            show_usage(res.usage)
            for p in res.problems:
                st.warning(p)

# ----------------------------------------------------------------------------- what-if
with tabs[5]:
    s = ctx.surrogate
    st.caption(f"Surrogate trained on the Nafion standard test (one MEA design). Leave-one-curve-out error: "
               f"{s.cv_mae_:.3f} V. It interpolates inside the measured design only.")
    c1, c2, c3, c4 = st.columns(4)
    p = c1.select_slider("Pressure (psig)", s.levels["pressure"], value=s.levels["pressure"][0])
    rh = c2.select_slider("Relative humidity (%)", s.levels["relative_humidity"], value=s.levels["relative_humidity"][0])
    mc = c3.select_slider("Compression (%)", s.levels["membrane_compression"], value=s.levels["membrane_compression"][0])
    nf = c4.select_slider("Nafion (%)", s.levels["nafion_percent"], value=s.levels["nafion_percent"][-1])
    curve = s.predict_curve(p, rh, mc, nf)
    sug = F.suggest_changes(ctx, p, rh, mc, nf)
    l, r = st.columns(2)
    with l:
        plot_curve(curve, f"Predicted curve, peak {curve.power_density.max():.0f} mW/cm²")
    with r:
        st.write("**Ranked single changes (predicted peak-power gain):**")
        st.dataframe(sug.head(6).round(1), hide_index=True, use_container_width=True)
        st.caption("Best measured conditions without any model:")
        st.dataframe(F.empirical_best(ctx, 3).round(1), hide_index=True, use_container_width=True)
    st.warning("Catalyst chemistry is not varied in this data. For Pt alloys or Fe-N-C see the literature notes in the "
               "Knowledge tab; they are not tested here.")

# ----------------------------------------------------------------------------- chatbot
with tabs[6]:
    st.caption("An agent that calls diagnosis, prediction, retrieval and comment tools. Try: 'Diagnose curve 12', "
               "'At 5 psig and 30% RH what could improve power?', 'Is flooding or dehydration more likely if resistance rises?'")
    if "chat" not in st.session_state:
        st.session_state.chat = []
    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.write(m["content"])
            if m.get("trace"):
                with st.expander("Tool trace"):
                    st.json(m["trace"])
    q = st.chat_input("Ask the agent")
    if q:
        st.session_state.chat.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.write(q)
        hist = [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat[:-1][-6:]]
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                res = run_agent(ctx, q, hist)
                res_off = run_agent(ctx_off, q, hist) if side_by_side else None
            if res_off is not None:
                c_off, c_live = st.columns(2)
                c_off.markdown("**⚪ Offline router**"); c_off.write(res_off.answer)
                c_live.markdown(f"**🟢 Live · {llm.model}**"); c_live.write(res.answer)
            else:
                st.write(res.answer)
            st.caption(f"mode: {res.mode}")
            show_usage(res.usage)
            for pr in res.problems:
                st.warning(pr)
            with st.expander("Tool trace"):
                st.json(res.steps, expanded=False)
        st.session_state.chat.append({"role": "assistant", "content": res.answer, "trace": res.steps})

# ----------------------------------------------------------------------------- RAG
with tabs[7]:
    topics = ["(all)"] + sorted({c.topic for c in ctx.retriever.chunks})
    c1, c2 = st.columns([3, 1])
    question = c1.text_input("Question about cathode catalysts, failures or the dataset",
                             "Why are Fe-N-C cathodes prone to flooding?")
    topic = c2.selectbox("Topic filter", topics)
    k = st.slider("Sources (k)", 1, 6, 4)
    if question:
        a = F.ask_knowledge(ctx, question, k=k, topic=None if topic == "(all)" else topic)
        if side_by_side:
            a_off = F.ask_knowledge(ctx_off, question, k=k, topic=None if topic == "(all)" else topic)
            c_off, c_live = st.columns(2)
            c_off.markdown("**⚪ Offline (extractive)**"); c_off.write(a_off.answer)
            c_live.markdown(f"**🟢 Live · {llm.model}**"); c_live.write(a.answer)
        else:
            st.markdown(a.answer)
        st.caption(f"mode: {a.mode}" + (" · abstained: not covered by the notes" if a.abstained else "")
                   + f" · retrieval: {getattr(ctx.retriever, 'last_backend', 'local')}")
        show_usage(a.usage)
        for w in a.invalid_citations:
            st.warning(f"Answer cites [S{w}] which was not retrieved")
        for w in a.unsupported_links:
            st.warning(f"Link not found in sources: {w}")
        for i, h in enumerate(a.hits, 1):
            with st.expander(f"[S{i}] {h.doc_id} · {h.chunk.title} · evidence: {h.chunk.evidence} · score {h.score:.2f}"):
                st.write(h.chunk.text)

# ----------------------------------------------------------------------------- reviews
with tabs[8]:
    st.caption("Write a lab comment or review of a run. The NLP pipeline tags sentiment and aspect, extracts numbers "
               "and symptoms, and links it to a failure zone. Comments are stored locally (git-ignored).")
    fams = sorted(ctx.curves.test_family.unique())
    with st.form("comment"):
        text = st.text_area("Comment", "At 5 psig and 30% RH the voltage collapsed at high current and the signal was noisy.")
        c1, c2, c3 = st.columns(3)
        rating = c1.slider("Rating", 1, 5, 3)
        fam = c2.selectbox("Test family", fams)
        author = c3.text_input("Name (optional)", "")
        go = st.form_submit_button("Analyse and save")
    if go and text.strip():
        an = analyze_comment(text, ctx.review_models, ctx.llm)
        ctx.store.add(text, rating, fam, an, author or "anonymous")
        if remember and pc_store is not None:
            try:
                pc_store.save_comments(st.session_state["session_id"], [text])
            except VectorStoreUnavailable as exc:
                st.warning(f"Not saved to Pinecone: {exc}")
        l, r = st.columns(2)
        l.metric("Sentiment", an["sentiment"], f"confidence {an['sentiment_confidence']:.2f}")
        r.metric("Top aspect", an["top_aspect"], f"suspected zone: {an['suspected_zone']}")
        st.write("Extracted:", an["numbers"] or "no numbers", "· symptoms:", an["symptoms"] or "none")
        if an.get("llm_extraction"):
            st.json(an["llm_extraction"])
        if an["suspected_zone"] != "unknown":
            hits = ctx.retriever.search(F.physics.ZONE_INFO[an["suspected_zone"]]["kb_query"], k=2)
            for h in hits:
                st.markdown(f"**[{h.doc_id}] {h.chunk.title}**: {h.chunk.text}")
        show_usage(an["usage"])
    store_df = ctx.store.load()
    st.subheader(f"Stored comments ({len(store_df)})")
    st.caption("Comments are kept in this browser session only and disappear when you close it." if ctx.store.in_memory
               else "Comments are appended to a local file shared by everyone using this machine.")
    if st.button("Delete my comments"):
        ctx.store.clear()
        if pc_store is not None and pc_store.available:
            try:
                pc_store.delete_session(st.session_state["session_id"])
            except VectorStoreUnavailable as exc:
                st.warning(f"Pinecone delete failed: {exc}")
        st.rerun()
    if remember and pc_store is not None and store_df is not None and len(store_df):
        q_mem = st.text_input("Search my remembered comments (Pinecone)", "membrane dry")
        try:
            st.write(pc_store.recall_comments(st.session_state["session_id"], q_mem))
        except VectorStoreUnavailable as exc:
            st.warning(str(exc))
    if len(store_df):
        st.dataframe(store_df.tail(15), hide_index=True, use_container_width=True)
        if "sentiment" in store_df:
            st.bar_chart(store_df.groupby(["test_family", "sentiment"]).size().unstack(fill_value=0))
        if st.button("Summarise all comments"):
            summary, u = summarise_comments(store_df, ctx.llm)
            st.write(summary)
            show_usage(u)
    st.info("Classifiers are trained on a synthetic corpus (see src/reviews.py) and generalise poorly to unseen wording; "
            "treat tags as suggestions.")

# ----------------------------------------------------------------------------- R&D watch
with tabs[9]:
    st.caption("Bibliographic facts (DOI, title, authors) come from Crossref or your CSV, never from the LLM. "
               "The digest is checked for DOIs that are not in the fetched records.")
    queries = st.text_area("Queries (one per line)", "\n".join(DEFAULT_QUERIES)).splitlines()
    days = st.slider("Look back (days)", 7, 365, 60)
    csv_up = st.file_uploader("Optional watchlist CSV (doi,title,authors,year,journal,abstract)", type="csv")
    if st.button("Fetch and triage"):
        papers, errors = [], []
        for qy in [q for q in queries if q.strip()]:
            try:
                papers += fetch_crossref(qy, days=days)
            except Exception as exc:
                errors.append(f"{qy}: {type(exc).__name__}")
        if csv_up is not None:
            tmp = ROOT / "models" / "_watchlist_upload.csv"
            tmp.parent.mkdir(exist_ok=True)
            tmp.write_bytes(csv_up.getvalue())
            papers += load_watchlist(tmp)
        papers = dedupe(papers)
        if errors:
            st.warning("Some fetches failed (offline?): " + "; ".join(errors))
        if not papers:
            st.info("No records. Check your connection or upload a watchlist CSV.")
        else:
            tri, u = triage(papers, ctx.llm)
            digest = build_digest(tri)
            st.markdown(digest)
            bad = verify_digest(digest, papers)
            st.success("Digest check: every DOI comes from a fetched record.") if not bad else st.error(f"Unknown DOIs: {bad}")
            show_usage(u)
            with st.expander("All records"):
                st.dataframe(tri, hide_index=True, use_container_width=True)

# ----------------------------------------------------------------------------- evaluation
with tabs[10]:
    st.caption("Results saved by notebook 07 (reports/*.csv). Run it to refresh.")
    rep = ROOT / "reports"
    for name, title in [("retrieval_experiment.csv", "Retrieval design choices"), ("qa_eval.csv", "Question answering behaviour"),
                        ("fault_injection.csv", "Failure screening on injected faults"), ("model_comparison.csv", "Voltage models (grouped CV)")]:
        f = rep / name
        if f.exists():
            st.subheader(title)
            st.dataframe(pd.read_csv(f).round(3), hide_index=True, use_container_width=True)
    st.write(f"Evaluation questions: {len(load_questions())} (written in `eval/eval_questions.json`).")
