"""
LLM Factory – uses the google-genai SDK (`pip install google-genai`).
Falls back gracefully (rule-based agents) when the LLM is unavailable.

Env vars (a .env file in the project root is loaded automatically):
    GEMINI_API_KEY  – optional; without it the agents run rule-based only
    LLM_MODEL       – override model (default: gemini-3.8-flash)
"""
from __future__ import annotations
import os

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = "gemini-3.8-flash"
# `or` so a blank `LLM_MODEL=` line in .env falls back to the default
LLM_MODEL = os.getenv("LLM_MODEL") or DEFAULT_MODEL


class LLMUnavailable(Exception):
    """The LLM cannot be used (no key, disabled, or quota gone) – use rules."""


class QuotaExhausted(LLMUnavailable):
    """Raised when the Gemini free-tier daily quota is used up."""


# ── Global state, so every agent and the CLI see the same picture ────────────
_quota_exhausted: bool = False
_disabled: bool = False
_calls_ok: int = 0
_calls_failed: int = 0
_last_error: str = ""
_client = None


def is_quota_exhausted() -> bool:
    return _quota_exhausted


def disable_llm() -> None:
    """Force rule-based mode (used by --no-llm)."""
    global _disabled
    _disabled = True


def llm_snapshot() -> tuple[int, int]:
    """(successful calls, failed calls) so far – diff two snapshots for a window."""
    return _calls_ok, _calls_failed


def llm_mode_label(before: tuple[int, int] = (0, 0)) -> str:
    """Honest description of what actually produced the output since `before`."""
    ok = _calls_ok - before[0]
    failed = _calls_failed - before[1]
    if _disabled:
        return "rule-based (--no-llm)"
    if not os.getenv("GEMINI_API_KEY"):
        return "rule-based (no GEMINI_API_KEY)"
    if _quota_exhausted:
        return ("Gemini, then rule-based (quota exhausted mid-run)" if ok
                else "rule-based (Gemini quota exhausted)")
    if ok and not failed:
        return "Gemini"
    if ok:
        return f"Gemini ({failed} call(s) fell back to rules)"
    if failed:
        return f"rule-based (Gemini errors: {_last_error[:70]})"
    return "rule-based"


def _get_client():
    global _client
    if _client is None:
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise LLMUnavailable("GEMINI_API_KEY not set")
        from google import genai
        _client = genai.Client(api_key=key)
    return _client


def call_gemini(prompt: str, temperature: float = 0.1) -> str:
    """
    Call Gemini. Raises LLMUnavailable (QuotaExhausted on 429) when the LLM
    cannot be used; any other API error is re-raised as-is after being counted.
    """
    global _quota_exhausted, _calls_ok, _calls_failed, _last_error

    if _disabled:
        raise LLMUnavailable("LLM disabled")
    if _quota_exhausted:
        raise QuotaExhausted("Daily quota exhausted – using rule-based fallback")

    client = _get_client()
    from google.genai import types
    try:
        response = client.models.generate_content(
            model=LLM_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=2048,  # headroom: thinking tokens count here
            ),
        )
        text = (response.text or "").strip()
        if not text:
            raise RuntimeError("empty response from model")
        _calls_ok += 1
        return text
    except Exception as e:
        err = str(e)
        if "429" in err or "quota" in err.lower() or "RESOURCE_EXHAUSTED" in err:
            _quota_exhausted = True
            raise QuotaExhausted(f"Gemini quota hit: {err[:120]}")
        _calls_failed += 1
        _last_error = err
        raise
