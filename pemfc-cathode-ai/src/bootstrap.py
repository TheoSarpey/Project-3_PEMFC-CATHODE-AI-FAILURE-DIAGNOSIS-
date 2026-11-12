"""First-run helpers so a fresh clone can start the app without any manual file shuffling.

The raw data (ECSIM, CC BY 4.0) is not in git.  `ensure_raw_data()` downloads it, or copies it from a
folder you already have, and says clearly when that is impossible (offline).  Keys are never handled here.
"""
from __future__ import annotations

import io
import shutil
import urllib.request
import zipfile
from pathlib import Path

from .config import PROJECT_ROOT, RAW_DIR

URL = "https://github.com/ECSIM/pem-dataset1/archive/v1.1.zip"
SKIP = {"Notebooks", "images", ".github"}


def raw_data_ok() -> bool:
    return RAW_DIR.exists() and any(RAW_DIR.rglob("*.csv"))


def download_ecsim(url: str = URL, timeout: int = 120) -> int:
    """Download and unpack the dataset into data/raw/. Returns the number of CSV files."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (pemfc-cathode-ai)"})
    payload = urllib.request.urlopen(req, timeout=timeout).read()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        for m in z.namelist():
            parts = Path(m).parts[1:]                    # drop the 'pem-dataset1-1.1/' prefix
            if not parts or m.endswith("/") or parts[0] in SKIP:
                continue
            dest = RAW_DIR.joinpath(*parts)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(z.read(m))
    return len(list(RAW_DIR.rglob("*.csv")))


def copy_ecsim(src: str | Path) -> int:
    src = Path(src).expanduser()
    if not src.exists():
        raise FileNotFoundError(f"{src} does not exist")
    shutil.copytree(src, RAW_DIR, dirs_exist_ok=True)
    return len(list(RAW_DIR.rglob("*.csv")))


def git_hygiene() -> dict:
    """Checks that secrets cannot reach git (never reads or returns a key)."""
    gi = PROJECT_ROOT / ".gitignore"
    lines = {l.strip() for l in gi.read_text().splitlines()} if gi.exists() else set()
    need = {".env": ".env" in lines, "secrets.toml": any("secrets.toml" in l for l in lines),
            "data/raw/": any(l.rstrip("/*") == "data/raw" for l in lines),
            "models/": any(l.rstrip("/*") == "models" for l in lines),
            "user comments": any("user_reviews" in l for l in lines)}
    return {"gitignore_found": gi.exists(), "ignored": need,
            ".env_file_present": (PROJECT_ROOT / ".env").exists(),
            "template_present": (PROJECT_ROOT / ".env.example").exists()}
