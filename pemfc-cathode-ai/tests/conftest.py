"""Shared helpers. Tests that need the raw dataset skip themselves if it is not downloaded."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RAW_OK = any((ROOT / "data" / "raw").rglob("*.csv")) if (ROOT / "data" / "raw").exists() else False
