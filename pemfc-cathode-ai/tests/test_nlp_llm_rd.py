"""Reviews NLP, LLM configuration, R&D monitor and secret hygiene."""
import os
import re
import tempfile
from pathlib import Path

from conftest import ROOT


def test_llm_settings_read_requested_variables_and_default_model():
    from src.config import get_llm_settings
    old = {k: os.environ.pop(k, None) for k in ("OPEN_AI_KEY", "OPENAI_API_KEY", "OPEN_AI_MODEL")}
    try:
        s = get_llm_settings()
        assert s.model == "GPT-6-LUNA" and not s.has_key
        os.environ["OPEN_AI_KEY"], os.environ["OPEN_AI_MODEL"] = "sk-test-not-real", "GPT-6-LUNA"
        s = get_llm_settings()
        assert s.has_key and s.model == "GPT-6-LUNA"
        from src.llm import LLMClient
        assert "sk-test" not in LLMClient(s).status()          # the key is never shown
    finally:
        for k, v in old.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v


def test_offline_client_raises_unavailable():
    from src.config import LLMSettings
    from src.llm import LLMClient, LLMUnavailable
    c = LLMClient(LLMSettings(None, "GPT-6-LUNA", "x", None, None))
    assert not c.available
    try:
        c.chat([{"role": "user", "content": "hi"}])
        raise AssertionError("should have raised")
    except LLMUnavailable:
        pass


def test_review_pipeline_and_store():
    from src.reviews import ReviewStore, analyze_comment, generate_synthetic_reviews, train_review_models
    df = generate_synthetic_reviews()
    assert df["synthetic"].all() and df["phrase_id"].nunique() == 72
    models = train_review_models(df)
    a = analyze_comment("At 5 psig and 30% RH voltage collapsed at high current, flooding, noisy signal", models)
    assert a["suspected_zone"] == "mass_transport" and a["numbers"]["pressure_psig"] == [5.0]
    with tempfile.TemporaryDirectory() as d:
        s = ReviewStore(Path(d) / "r.jsonl")
        s.add("test comment", 3, "family", a)
        assert len(s.load()) == 1


def test_rd_monitor_never_outputs_unknown_dois():
    from src.rd_monitor import Paper, build_digest, triage, verify_digest
    # UNIT-TEST FIXTURE: not a real paper, DOI prefix 10.0000 is not registered
    papers = [Paper("10.0000/fixture-not-a-real-paper", "Oxygen reduction on Pt catalysts in PEMFC cathodes",
                    "Fixture Author", 2026, "Fixture Journal", "")]
    tri, _ = triage(papers)
    digest = build_digest(tri, min_relevance=0)
    assert "No abstract available." in digest
    assert verify_digest(digest, papers) == []
    assert verify_digest(digest + " 10.1234/invented", papers) == ["10.1234/invented"]


def test_no_api_key_in_repository_files():
    bad = re.compile(r"sk-[A-Za-z0-9]{20,}")
    for p in ROOT.rglob("*"):
        if p.is_file() and p.suffix in {".py", ".md", ".txt", ".ipynb", ".toml", ".json", ".example", ".yml"} \
                and ".venv" not in p.parts and "raw" not in p.parts:
            assert not bad.search(p.read_text(errors="ignore")), f"possible key in {p}"
