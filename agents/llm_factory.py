"""
LLM Factory – uses google-genai SDK.
Falls back gracefully when quota is exceeded.

Env vars:
    GEMINI_API_KEY  – required
    LLM_MODEL       – override model (default: gemini-3.8-flash)
"""
from __future__ import annotations
import os
import google.generativeai as genai

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
LLM_MODEL      = os.getenv("LLM_MODEL", "gemini-3.8-flash")

# Track quota exhaustion globally so all agents know immediately
_quota_exhausted: bool = False

genai.configure(api_key=GEMINI_API_KEY)


def is_quota_exhausted() -> bool:
    return _quota_exhausted


def call_gemini(prompt: str, temperature: float = 0.1) -> str:
    """
    Call Gemini. Raises QuotaExhausted on 429, other exceptions as-is.
    """
    global _quota_exhausted

    if _quota_exhausted:
        raise QuotaExhausted("Daily quota exhausted – using rule-based fallback")

    model = genai.GenerativeModel(
        model_name=LLM_MODEL,
        generation_config=genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=512,
        ),
    )
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        err = str(e)
        if "429" in err or "quota" in err.lower() or "RESOURCE_EXHAUSTED" in err:
            _quota_exhausted = True
            raise QuotaExhausted(f"Gemini quota hit: {err[:120]}")
        raise


class QuotaExhausted(Exception):
    """Raised when the Gemini free-tier daily quota is used up."""
    pass
