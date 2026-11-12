"""Fresh-clone safety: git hygiene, first-run helpers, and (if Streamlit is installed) the app itself."""
import tempfile
from pathlib import Path

import pytest

from conftest import RAW_OK, ROOT


def test_gitignore_keeps_secrets_data_and_models_out_of_git():
    from src.bootstrap import git_hygiene
    h = git_hygiene()
    assert h["gitignore_found"] and all(h["ignored"].values()), h["ignored"]
    assert h["template_present"]                      # .env.example is committed, .env is not


def test_copy_ecsim_and_raw_check_use_the_configured_folder(monkeypatch=None):
    import src.bootstrap as b
    with tempfile.TemporaryDirectory() as d:
        src_dir, dst = Path(d) / "src", Path(d) / "raw"
        (src_dir / "fam").mkdir(parents=True)
        (src_dir / "fam" / "1.csv").write_text("a,b\n1,2\n")
        old = b.RAW_DIR
        try:
            b.RAW_DIR = dst
            assert not b.raw_data_ok()
            assert b.copy_ecsim(src_dir) == 1 and b.raw_data_ok()
        finally:
            b.RAW_DIR = old


@pytest.mark.skipif(not RAW_OK, reason="needs raw data")
def test_streamlit_app_runs_without_exceptions():
    try:
        from streamlit.testing.v1 import AppTest
    except Exception:
        pytest.skip("streamlit not installed")
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=180).run()
    assert not at.exception
