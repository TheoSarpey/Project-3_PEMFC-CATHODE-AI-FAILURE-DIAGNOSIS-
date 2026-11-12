"""RAG, agent and guardrail tests (no API key needed: a scripted LLM replays tool calls)."""
import json

import pytest

from conftest import RAW_OK, ROOT


def test_knowledge_base_loads_and_retrieves():
    from src.rag import build_retriever, load_documents
    docs = load_documents()
    assert len(docs) >= 25 and all(d["text"] for d in docs)
    hits = build_retriever("bm25").search("how to tell flooding from dehydration", k=3)
    assert "K09" in [h.doc_id for h in hits]


def test_retrieval_quality_on_eval_set():
    from src.rag import build_retriever
    qs = [q for q in json.loads((ROOT / "eval" / "eval_questions.json").read_text()) if q["answerable"]]
    r = build_retriever("hybrid")
    hit = np_mean([any(h.doc_id in q["gold_docs"] for h in r.search(q["question"], k=4)) for q in qs])
    assert hit >= 0.85


def np_mean(xs):
    return sum(xs) / len(xs)


def test_out_of_scope_question_abstains():
    from src.rag import answer_question, build_retriever
    a = answer_question("Who won the football world cup in 2018?", build_retriever("bm25"))
    assert a.abstained and a.mode == "abstained"


def test_verify_answer_flags_fake_doi_and_bad_citation():
    from src.rag import build_retriever, verify_answer
    hits = build_retriever("bm25").search("flooding", k=2)
    cited, invalid, unsupported = verify_answer("Water blocks pores [S1][S7]. See 10.9999/made.up", hits)
    assert invalid == [7] and unsupported == ["10.9999/made.up"]


@pytest.mark.skipif(not RAW_OK, reason="needs raw data")
def test_agent_loop_with_scripted_llm_and_guardrails():
    from src.agent import run_agent
    from src.functions import get_context
    from src.llm import LLMResponse, ScriptedLLM, Usage
    ctx = get_context()
    call = {"id": "c1", "name": "diagnose_curve", "arguments": {"curve_id": 12}}
    ctx.llm = ScriptedLLM([
        LLMResponse("", [call], Usage(1, 100, 10, 0.1, None),
                    {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function",
                     "function": {"name": "diagnose_curve", "arguments": json.dumps({"curve_id": 12})}}]}),
        LLMResponse("Curve 12 may be limited. See 10.1234/invented and [K99].", [], Usage(1, 200, 30, 0.2, None))])
    res = run_agent(ctx, "Diagnose curve 12")
    assert res.mode == "llm-agent" and res.steps[0]["tool"] == "diagnose_curve"
    assert any("DOI" in p for p in res.problems) and any("K99" in p for p in res.problems)


@pytest.mark.skipif(not RAW_OK, reason="needs raw data")
def test_offline_agent_answers_and_refuses_extrapolation():
    from src.agent import run_agent, run_tool
    from src.functions import get_context
    ctx = get_context()
    assert run_agent(ctx, "Diagnose curve 12").mode == "offline-router"
    out = run_tool(ctx, "predict_performance", dict(pressure=99, relative_humidity=50, membrane_compression=5, nafion_percent=20))
    assert "error" in out
