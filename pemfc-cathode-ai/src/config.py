"""Central configuration: paths, random seed and the OpenAI settings.

The two settings requested for this project are read from the environment
(or from a local ``.env`` file that is git-ignored):

    OPEN_AI_KEY=sk-...            # never commit this
    OPEN_AI_MODEL=GPT-6-LUNA      # model name sent to the API as-is

The official OpenAI variable name ``OPENAI_API_KEY`` is accepted as well.
If no key is present the whole product still works in *offline mode*
(extractive answers, rule-based agent), so a grader can run it from a
fresh clone without any secret.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:  # python-dotenv is optional
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)

RAW_DIR = PROJECT_ROOT / "data" / "raw"
KNOWLEDGE_DIR = PROJECT_ROOT / "data" / "knowledge"
REVIEWS_DIR = PROJECT_ROOT / "data" / "reviews"
MODELS_DIR = PROJECT_ROOT / "models"          # git-ignored, rebuilt by notebooks
REPORTS_DIR = PROJECT_ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
EVAL_DIR = PROJECT_ROOT / "eval"

SEED = 42
DEFAULT_MODEL = "GPT-6-LUNA"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"


@dataclass(frozen=True)
class LLMSettings:
    api_key: str | None
    model: str
    embed_model: str
    price_in_per_1m: float | None    # USD per 1M input tokens (unknown unless you set it)
    price_out_per_1m: float | None   # USD per 1M output tokens

    @property
    def has_key(self) -> bool:
        return bool(self.api_key)


def _float_or_none(name: str) -> float | None:
    raw = os.getenv(name, "").strip()
    try:
        return float(raw) if raw else None
    except ValueError:
        return None


def get_llm_settings(api_key: str | None = None, model: str | None = None) -> LLMSettings:
    """Read settings at call time so the app can override them per session."""
    key = api_key or os.getenv("OPEN_AI_KEY") or os.getenv("OPENAI_API_KEY") or None
    return LLMSettings(
        api_key=key.strip() if key else None,
        model=(model or os.getenv("OPEN_AI_MODEL") or DEFAULT_MODEL).strip(),
        embed_model=os.getenv("OPEN_AI_EMBED_MODEL", DEFAULT_EMBED_MODEL).strip(),
        price_in_per_1m=_float_or_none("OPEN_AI_PRICE_IN_PER_1M"),
        price_out_per_1m=_float_or_none("OPEN_AI_PRICE_OUT_PER_1M"),
    )


def ensure_dirs() -> None:
    for d in (MODELS_DIR, FIG_DIR, REVIEWS_DIR):
        d.mkdir(parents=True, exist_ok=True)
