"""Diagnosis agent: an LLM that decides which tools to call, with an offline fallback.

Tools (all deterministic Python, all defined in ``functions.py`` / ``rag.py`` / ``reviews.py``):

    search_knowledge   retrieve literature-note passages (RAG)
    list_curves        find curves by test family / flagged only
    diagnose_curve     physics screening + ML second opinion + guidance for one curve
    predict_performance   surrogate prediction of peak power for design conditions
    suggest_changes    ranked single-variable improvements inside the measured design space
    analyze_comment    NLP on a lab comment
    summarize_comments summary of stored comments

The LLM never computes electrochemistry itself; it chooses tools and writes the explanation.
Guardrails: tool results are data, the answer is checked for invented knowledge-base ids and
for DOIs that no tool returned, and the number of steps is capped.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import pandas as pd

from . import functions as F
from .llm import LLMUnavailable, Usage
from .reviews import analyze_comment, summarise_comments

MAX_STEPS = 6
ALLOWED_DOIS = {F.DATASET_DOI}

AGENT_SYSTEM = f"""You are the PEMFC Cathode Diagnosis Agent for an electrochemistry team.
You help find WHERE a cell fails (kinetic, ohmic or mass-transport zone), WHICH parameters matter,
HOW to troubleshoot, and WHAT to improve, using the ECSIM Nafion/MEA dataset (Pt/C cathodes, H2/O2).
Rules:
- Call tools for every number, curve or literature claim. Never compute or recall numbers yourself.
- Cite knowledge-base notes as [K07] style ids and curves as 'curve 12'. Cite only ids returned by tools.
- Never invent DOIs, URLs, papers or catalyst performance numbers. The only DOI you may mention is {F.DATASET_DOI}.
- DFT rules: separate three kinds of value and name them: calculated (DFT or database), ML-predicted (with its
  uncertainty) and textbook heuristics. Never invent an adsorption energy, a limiting potential or a source. If a
  tool says recommendation_allowed is false (demo or toy data), say there is no recommendation and why. A model
  prediction outside the training chemistry is a reason to run DFT, not a result.
- A polarisation curve narrows the cause but does not prove it: say what a second measurement (EIS, CV, crossover
  test, repeat run) would confirm.
- The data contain only Pt/C cathodes in oxygen. For other catalysts or air operation, say it comes from literature
  notes and is not tested by the data.
- Be concise: a short diagnosis, the parameters to check, 2-4 concrete actions, and the uncertainty."""

TOOLS = [
    {"type": "function", "function": {
        "name": "search_knowledge", "description": "Search the curated PEMFC knowledge base.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "k": {"type": "integer"}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "list_curves", "description": "List measured polarisation curves; filter by test family substring or flagged only.",
        "parameters": {"type": "object", "properties": {"family": {"type": "string"}, "flagged_only": {"type": "boolean"}}}}},
    {"type": "function", "function": {
        "name": "diagnose_curve", "description": "Diagnose one curve by id: failure zone, parameters, ML second opinion, guidance.",
        "parameters": {"type": "object", "properties": {"curve_id": {"type": "integer"}}, "required": ["curve_id"]}}},
    {"type": "function", "function": {
        "name": "predict_performance", "description": "Surrogate-predicted peak power and current at 0.6 V for design conditions (measured ranges only).",
        "parameters": {"type": "object", "properties": {
            "pressure": {"type": "number"}, "relative_humidity": {"type": "number"},
            "membrane_compression": {"type": "number"}, "nafion_percent": {"type": "number"}},
            "required": ["pressure", "relative_humidity", "membrane_compression", "nafion_percent"]}}},
    {"type": "function", "function": {
        "name": "suggest_changes", "description": "Rank single-variable changes by predicted peak-power gain for the given conditions.",
        "parameters": {"type": "object", "properties": {
            "pressure": {"type": "number"}, "relative_humidity": {"type": "number"},
            "membrane_compression": {"type": "number"}, "nafion_percent": {"type": "number"}},
            "required": ["pressure", "relative_humidity", "membrane_compression", "nafion_percent"]}}},
    {"type": "function", "function": {
        "name": "analyze_comment", "description": "NLP analysis of a lab comment: sentiment, aspect, symptoms, suspected zone.",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}},
    {"type": "function", "function": {
        "name": "summarize_comments", "description": "Summarise the comments stored by users of the app.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "dft_rank_candidates", "description": "Rank catalyst candidates by predicted ORR limiting potential (ML on cheap features) with uncertainty, out-of-distribution flag, stability flags and next step.",
        "parameters": {"type": "object", "properties": {"compositions": {"type": "array", "items": {"type": "string"}},
                       "facet": {"type": "string"}, "structure": {"type": "string"}}, "required": ["compositions"]}}},
    {"type": "function", "function": {
        "name": "dft_descriptors", "description": "Look up DFT-table rows for a composition: adsorption energies, free energies, limiting potential, provenance.",
        "parameters": {"type": "object", "properties": {"composition": {"type": "string"}}, "required": ["composition"]}}},
    {"type": "function", "function": {
        "name": "next_dft_calculation", "description": "Which candidates to run DFT on next (highest uncertainty-aware upside), from a list of compositions.",
        "parameters": {"type": "object", "properties": {"compositions": {"type": "array", "items": {"type": "string"}}, "k": {"type": "integer"}}, "required": ["compositions"]}}},
]


# ----------------------------------------------------------------------------- tool execution
def run_tool(ctx: F.Context, name: str, args: dict) -> dict:
    """Execute one tool; errors are returned as data so the model can recover."""
    try:
        if name == "search_knowledge":
            hits = ctx.retriever.search(str(args["query"]), k=int(args.get("k", 3)))
            return {"results": [{"doc_id": h.doc_id, "title": h.chunk.title, "evidence": h.chunk.evidence,
                                 "text": h.chunk.text} for h in hits]}
        if name == "list_curves":
            t = F.list_curves(ctx, args.get("family"), bool(args.get("flagged_only", False)), limit=10)
            return {"curves": json.loads(t.to_json(orient="records"))}
        if name == "diagnose_curve":
            d = F.diagnose_curve(ctx, int(args["curve_id"]), k=2)
            d["guidance"] = [{"doc_id": g["doc_id"], "title": g["title"], "text": g["text"]} for g in d.get("guidance", [])]
            return d
        if name in ("predict_performance", "suggest_changes"):
            cond = {k: float(args[k]) for k in F.DESIGN_VARS}
            warn = ctx.surrogate.check(**cond)
            if warn:
                return {"error": "; ".join(warn)}
            if name == "predict_performance":
                c = ctx.surrogate.predict_curve(**cond)
                return {"pred_p_max_mW_cm2": round(float(c["power_density"].max()), 1),
                        "surrogate_cv_mae_V": round(ctx.surrogate.cv_mae_, 4),
                        "note": "interpolation inside the measured design of one MEA type"}
            t = F.suggest_changes(ctx, **cond)
            return {"baseline_p_max": round(t.attrs["baseline_p_max"], 1),
                    "ranked_changes": json.loads(t.head(5).round(1).to_json(orient="records"))}
        if name == "analyze_comment":
            a = analyze_comment(str(args["text"]), ctx.review_models, None)
            a.pop("usage", None)
            return a
        if name in ("dft_rank_candidates", "next_dft_calculation"):
            from .dft_ml import suggest_next_dft
            comps = [str(c) for c in args["compositions"]]
            r = F.dft_rank(ctx, comps, str(args.get("facet", "111")), str(args.get("structure", "skin")))
            src = F.get_dft(ctx)
            if name == "next_dft_calculation":
                return {"data_source": src["source"], "next": json.loads(suggest_next_dft(r, int(args.get("k", 3))).round(3).to_json(orient="records"))}
            cols = ["composition", "pred_U_L", "pred_sd", "confidence", "outside_training_distribution", "rank", "recommendation_allowed", "recommendation"]
            out = json.loads(r[cols].round(3).to_json(orient="records"))
            for o, fl in zip(out, r["stability_flags"]):
                o["stability_flags"] = fl
            return {"data_source": src["source"], "trusted_training_data": src["trusted"], "provenance_of_numbers": "ml_predicted", "candidates": out}
        if name == "dft_descriptors":
            t = F.get_dft(ctx)["table"]
            m = t[t["composition"].str.lower() == str(args["composition"]).lower()]
            if m.empty:
                return {"error": f"no DFT-table rows for {args['composition']}; do not guess values"}
            cols = ["system_id", "facet", "E_OH", "E_O", "E_OOH", "dG_OH", "U_L", "provenance", "reference"]
            return {"data_source": F.get_dft(ctx)["source"], "rows": json.loads(m[cols].round(3).to_json(orient="records"))}
        if name == "summarize_comments":
            text, _ = summarise_comments(ctx.store.load(), None)
            return {"offline_summary": text}
    except (KeyError, ValueError, TypeError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {"error": f"unknown tool {name}"}


# ----------------------------------------------------------------------------- verification
def verify_agent_answer(answer: str, ctx: F.Context, seen_doc_ids: set[str], seen_curve_ids: set[int]) -> list[str]:
    """Return a list of problems found in the final answer (empty list = clean)."""
    problems = []
    all_ids = {d["id"] for d in __import__("src.rag", fromlist=["load_documents"]).load_documents()}
    for kid in set(re.findall(r"\[(K\d{2})\]", answer)):
        if kid not in all_ids:
            problems.append(f"cites unknown knowledge note {kid}")
        elif kid not in seen_doc_ids:
            problems.append(f"cites {kid} although no tool returned it")
    for doi in re.findall(r"10\.\d{4,9}/[^\s\]\)\"'<>,;]+", answer):
        if doi.rstrip(".") not in ALLOWED_DOIS:
            problems.append(f"unsupported DOI {doi}")
    valid_curves = set(ctx.curves["curve_id"].astype(int))
    for cid in re.findall(r"curve (\d+)", answer, flags=re.I):
        if int(cid) not in valid_curves:
            problems.append(f"mentions non-existent curve {cid}")
    return problems


@dataclass
class AgentResult:
    answer: str
    mode: str                              # "llm-agent" | "offline-router"
    steps: list[dict] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)


def _collect_ids(result: dict, docs: set[str], curves: set[int]) -> None:
    for key in ("results", "guidance"):
        for r in result.get(key, []) or []:
            if "doc_id" in r:
                docs.add(r["doc_id"])
    if "curve_id" in result:
        curves.add(int(result["curve_id"]))
    for c in result.get("curves", []) or []:
        curves.add(int(c["curve_id"]))


# ----------------------------------------------------------------------------- the agent
def run_agent(ctx: F.Context, question: str, history: list[dict] | None = None) -> AgentResult:
    llm = ctx.llm
    if llm is not None and getattr(llm, "available", False):
        try:
            return _run_llm_agent(ctx, question, history or [])
        except LLMUnavailable as exc:
            res = _run_offline(ctx, question)
            res.problems.append(f"LLM call failed, answered offline: {exc}")
            return res
    return _run_offline(ctx, question)


def _run_llm_agent(ctx: F.Context, question: str, history: list[dict]) -> AgentResult:
    messages = [{"role": "system", "content": AGENT_SYSTEM}] + history + [{"role": "user", "content": question}]
    steps, usage = [], Usage()
    docs, curves = set(), set()
    for _ in range(MAX_STEPS):
        r = ctx.llm.chat(messages, tools=TOOLS)
        usage.add(r.usage)
        if not r.tool_calls:
            problems = verify_agent_answer(r.text, ctx, docs, curves)
            return AgentResult(r.text.strip(), "llm-agent", steps, problems, usage)
        messages.append(r.raw_message or {"role": "assistant", "content": r.text})
        for call in r.tool_calls:
            result = run_tool(ctx, call["name"], call["arguments"])
            _collect_ids(result, docs, curves)
            steps.append({"tool": call["name"], "arguments": call["arguments"], "result": result})
            messages.append({"role": "tool", "tool_call_id": call["id"],
                             "content": json.dumps(result, default=str)[:6000]})
    return AgentResult("I could not finish within the step limit; see the tool trace for what was found.",
                       "llm-agent", steps, ["step limit reached"], usage)


# ----------------------------------------------------------------------------- offline planner
def _run_offline(ctx: F.Context, question: str) -> AgentResult:
    """Rule-based router: the same tools, chosen by keywords, answer assembled from templates."""
    q = question.lower()
    steps, parts = [], []

    def call(name, **args):
        res = run_tool(ctx, name, args)
        steps.append({"tool": name, "arguments": args, "result": res})
        return res

    m = re.search(r"curve\s*#?\s*(\d+)", q)
    cond = {k: re.search(p, q) for k, p in {
        "pressure": r"(\d+(?:\.\d+)?)\s*psi", "relative_humidity": r"(\d+(?:\.\d+)?)\s*%\s*(?:rh|relative)"}.items()}
    if m:
        d = call("diagnose_curve", curve_id=int(m.group(1)))
        if "error" in d:
            parts.append(f"Could not diagnose that curve: {d['error']}")
        else:
            parts.append(f"Curve {d['curve_id']}: {d['zone_label']}. Likely component: {d['likely_component']}. "
                         f"Severity z = {d['severity_z']}. Agreement between methods: {d['agreement']}.")
            if d["parameters_to_check"]:
                parts.append("Parameters to check: " + ", ".join(d["parameters_to_check"]) + ".")
            for g in d.get("guidance", [])[:2]:
                parts.append(f"[{g['doc_id']}] {g['text'].split('. ')[0]}.")
    elif cond["pressure"] and cond["relative_humidity"] and re.search(r"improv|better|increase|optimi|suggest|what if|change", q):
        c = {"pressure": float(cond["pressure"].group(1)), "relative_humidity": float(cond["relative_humidity"].group(1)),
             "membrane_compression": 5.0, "nafion_percent": 25.0}
        for k_, p_ in (("membrane_compression", r"(\d+)\s*%\s*compress"), ("nafion_percent", r"(\d+)\s*%\s*nafion")):
            mm = re.search(p_, q)
            if mm:
                c[k_] = float(mm.group(1))
        res = call("suggest_changes", **c)
        if "error" in res:
            parts.append(res["error"])
        else:
            best = res["ranked_changes"][0]
            cond_txt = (f"{c['pressure']:g} psig, {c['relative_humidity']:g}% RH, {c['membrane_compression']:g}% compression, "
                        f"{c['nafion_percent']:g}% Nafion")
            parts.append(f"At {cond_txt} the surrogate predicts {res['baseline_p_max']} mW/cm2. Best single change: "
                         f"{best['change']} (predicted {best['gain_mW_cm2']:+.0f} mW/cm2). Interpolation inside "
                         "measured conditions of one MEA design; confirm with a measurement.")
    elif re.search(r"\bpt\d?[a-z]{1,2}\d?\b", q) and re.search(r"rank|dft|candidate|next|priorit|screen|compare|better", q):
        comps = sorted({m.group(0).capitalize().replace("Pt", "Pt", 1) for m in re.finditer(r"pt\d?[a-z]{1,2}\d?", q)})
        comps = [re.sub(r"^pt", "Pt", c, flags=re.I) for c in comps]
        tool = "next_dft_calculation" if re.search(r"next|which dft|run dft", q) else "dft_rank_candidates"
        res = call(tool, compositions=comps)
        if "error" in res:
            parts.append(res["error"])
        elif tool == "dft_rank_candidates":
            parts.append(f"Data source: {res['data_source']}. Predicted values are ML predictions, not DFT.")
            for c in res["candidates"]:
                parts.append(f"{c['composition']}: predicted U_L {c['pred_U_L']} V ± {c['pred_sd']} (confidence: {c['confidence']}). {c['recommendation']}")
        else:
            parts.append(f"Data source: {res['data_source']}. Highest-value DFT calculations next: " +
                         "; ".join(f"{n['composition']}({n['facet']}) predicted U_L {n['pred_U_L']} V, sd {n['pred_sd']}" for n in res["next"]))
    elif re.search(r"\bcomment|review|feedback", q) and re.search(r"summar|overview|what do", q):
        res = call("summarize_comments")
        parts.append(res["offline_summary"])
    else:
        res = call("search_knowledge", query=question, k=3)
        a = F.ask_knowledge(ctx, question, k=3)
        parts.append(a.answer if not a.abstained else a.answer)
        steps.append({"tool": "answer_from_sources", "arguments": {}, "result": {"mode": a.mode, "abstained": a.abstained}})
    docs, curves = set(), set()
    for s in steps:
        _collect_ids(s["result"], docs, curves)
    answer = " ".join(parts) + "\n\n(Offline router: no LLM key set, answers are assembled from templates and tool outputs.)"
    return AgentResult(answer, "offline-router", steps, [], Usage())
