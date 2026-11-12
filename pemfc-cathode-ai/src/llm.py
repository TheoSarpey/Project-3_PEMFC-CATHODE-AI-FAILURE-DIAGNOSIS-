"""Thin, testable wrapper around the OpenAI API.

Design goals
------------
* One place that knows the key and the model name (``OPEN_AI_KEY`` and
  ``OPEN_AI_MODEL``, default ``GPT-6-LUNA``; see ``src/config.py``).
* Never crash the product when the key, the SDK or the model is missing:
  ``LLMClient.available`` is False and callers fall back to offline logic.
* Count tokens, seconds and (if you set prices) dollars for every call, so
  the evaluation can report a cost per question.
* Never print or log the key.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from .config import LLMSettings, get_llm_settings


class LLMUnavailable(RuntimeError):
    """Raised when an LLM call is requested but no working client exists."""


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    cost_usd: float | None = 0.0     # None = prices unknown

    def add(self, other: "Usage") -> None:
        self.calls += other.calls
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.seconds += other.seconds
        if self.cost_usd is None or other.cost_usd is None:
            self.cost_usd = None
        else:
            self.cost_usd += other.cost_usd

    def as_dict(self) -> dict:
        return {"calls": self.calls, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "seconds": round(self.seconds, 3),
                "cost_usd": None if self.cost_usd is None else round(self.cost_usd, 6)}


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[dict] = field(default_factory=list)   # [{id, name, arguments}]
    usage: Usage = field(default_factory=Usage)
    raw_message: dict | None = None                         # assistant message to append to history


class LLMClient:
    """OpenAI chat + embeddings with graceful degradation."""

    def __init__(self, settings: LLMSettings | None = None):
        self.settings = settings or get_llm_settings()
        self.total = Usage()
        self.last_error: str | None = None
        self._client = None
        if self.settings.has_key:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.settings.api_key)
            except Exception as exc:          # SDK missing or broken
                self.last_error = f"openai SDK unavailable: {type(exc).__name__}"

    # -- status ---------------------------------------------------------------
    @property
    def available(self) -> bool:
        return self._client is not None

    @property
    def model(self) -> str:
        return self.settings.model

    def status(self) -> str:
        if not self.settings.has_key:
            return "offline (no OPEN_AI_KEY set)"
        if self._client is None:
            return f"offline ({self.last_error})"
        if self.last_error:
            return f"degraded ({self.last_error})"
        return f"online · model {self.model}"

    # -- cost -------------------------------------------------------------------
    def _cost(self, p: int, c: int) -> float | None:
        s = self.settings
        if s.price_in_per_1m is None or s.price_out_per_1m is None:
            return None
        return (p * s.price_in_per_1m + c * s.price_out_per_1m) / 1e6

    def ping(self) -> tuple[bool, str]:
        """Tiny live call to confirm that the key and the model name work (costs a few tokens)."""
        if not self.available:
            return False, self.status()
        try:
            r = self.chat([{"role": "user", "content": "Reply with the single word OK."}], temperature=None)
            return True, f"live call succeeded with model {self.model} ({r.usage.prompt_tokens}+{r.usage.completion_tokens} tokens)"
        except LLMUnavailable as exc:
            return False, f"live call failed: {exc}"

    # -- chat -------------------------------------------------------------------
    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             temperature: float | None = 0.0, json_mode: bool = False) -> LLMResponse:
        if not self.available:
            raise LLMUnavailable(self.status())
        kwargs: dict = {"model": self.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if temperature is not None:
            kwargs["temperature"] = temperature
        t0 = time.perf_counter()
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            # Some models reject a custom temperature: retry once without it.
            if "temperature" in str(exc).lower() and "temperature" in kwargs:
                kwargs.pop("temperature")
                try:
                    resp = self._client.chat.completions.create(**kwargs)
                except Exception as exc2:
                    self.last_error = f"{type(exc2).__name__}: {str(exc2)[:160]}"
                    raise LLMUnavailable(self.last_error) from exc2
            else:
                self.last_error = f"{type(exc).__name__}: {str(exc)[:160]}"
                raise LLMUnavailable(self.last_error) from exc
        dt = time.perf_counter() - t0
        msg = resp.choices[0].message
        calls = []
        for tc in (getattr(msg, "tool_calls", None) or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append({"id": tc.id, "name": tc.function.name, "arguments": args})
        pt = getattr(resp.usage, "prompt_tokens", 0) or 0
        ct = getattr(resp.usage, "completion_tokens", 0) or 0
        usage = Usage(1, pt, ct, dt, self._cost(pt, ct))
        self.total.add(usage)
        raw = {"role": "assistant", "content": msg.content or ""}
        if calls:
            raw["tool_calls"] = [{"id": c["id"], "type": "function",
                                  "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
                                 for c in calls]
        return LLMResponse(msg.content or "", calls, usage, raw)

    def complete_json(self, system: str, user: str) -> tuple[dict, Usage]:
        """Ask for a JSON object; returns ({}, usage) if the reply is not valid JSON."""
        r = self.chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
                      json_mode=True)
        try:
            return json.loads(r.text), r.usage
        except json.JSONDecodeError:
            return {}, r.usage

    # -- embeddings ---------------------------------------------------------------
    def embed(self, texts: list[str]) -> list[list[float]]:
        if not self.available:
            raise LLMUnavailable(self.status())
        try:
            resp = self._client.embeddings.create(model=self.settings.embed_model, input=texts)
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {str(exc)[:160]}"
            raise LLMUnavailable(self.last_error) from exc
        pt = getattr(resp.usage, "prompt_tokens", 0) or 0
        self.total.add(Usage(1, pt, 0, 0.0, None))
        return [d.embedding for d in resp.data]


class ScriptedLLM:
    """Deterministic stand-in used by tests: replays a list of LLMResponse objects."""

    def __init__(self, responses: list[LLMResponse], model: str = "scripted-test-llm"):
        self._responses = list(responses)
        self.model = model
        self.available = True
        self.total = Usage()
        self.calls: list[list[dict]] = []

    def status(self) -> str:
        return "scripted (tests)"

    def ping(self):
        return True, "scripted"

    def chat(self, messages, tools=None, temperature=0.0, json_mode=False) -> LLMResponse:
        self.calls.append(messages)
        r = self._responses.pop(0) if self._responses else LLMResponse("(no more scripted replies)")
        self.total.add(r.usage)
        return r

    def complete_json(self, system, user):
        r = self.chat([{"role": "system", "content": system}, {"role": "user", "content": user}], json_mode=True)
        try:
            return json.loads(r.text), r.usage
        except json.JSONDecodeError:
            return {}, r.usage


def connect_llm(api_key: str | None = None, model: str | None = None, ask: bool = False) -> "LLMClient":
    """One call for notebooks: read OPEN_AI_KEY / OPEN_AI_MODEL, optionally prompt for them.

    ask=True shows a hidden-input prompt (getpass) for the key and a normal prompt for the model, so the
    key is never written into a notebook cell or its output.  Empty key = offline mode.
    """
    import getpass
    import os
    st = get_llm_settings(api_key, model)
    if ask:
        typed = getpass.getpass(f"OPEN_AI_KEY (hidden, Enter = {'keep current' if st.has_key else 'offline mode'}): ").strip()
        model_in = input(f"OPEN_AI_MODEL [{st.model}]: ").strip()
        st = get_llm_settings(typed or api_key, model_in or model)
    if st.has_key:
        os.environ["OPEN_AI_KEY"] = st.api_key          # this Python session only
    os.environ["OPEN_AI_MODEL"] = st.model
    return LLMClient(st)
