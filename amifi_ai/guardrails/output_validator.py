# amifi_ai/guardrails/output_validator.py
"""
Structured output validator + guardrails.

Pipeline:
  1. Try direct JSON parse
  2. If fails → attempt JSON auto-repair
  3. If fails → retry with corrective prompt via LLM
  4. If fails → rule-based deterministic fallback
  5. Validate against Pydantic schema
  6. Enforce confidence threshold
"""

import json
import logging
import re

from pydantic import BaseModel, field_validator

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.50
VALID_CATEGORIES = {
    "Shopping", "Food & Dining", "Transport", "Travel",
    "Utilities", "Groceries", "Fuel", "Finance", "Miscellaneous",
}
VALID_TYPES = {"debit", "credit"}


# ── Pydantic output schema ────────────────────────────────────────────────────
class ClassificationOutput(BaseModel):
    merchant: str
    category: str
    type: str
    confidence: float

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in VALID_TYPES:
            raise ValueError(f"type must be 'debit' or 'credit', got: {v!r}")
        return v

    @field_validator("category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        if v not in VALID_CATEGORIES:
            logger.warning(f"[VALIDATOR] Unknown category '{v}' → forcing 'Miscellaneous'")
            return "Miscellaneous"
        return v

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        return round(max(0.0, min(1.0, float(v))), 2)

    @field_validator("merchant")
    @classmethod
    def validate_merchant(cls, v: str) -> str:
        v = v.strip()
        if not v or v.lower() in ("unknown", "n/a", "none", ""):
            return "Unknown"
        return v


# ── JSON Auto-Repair ──────────────────────────────────────────────────────────
def _repair_json(raw: str) -> str:
    """
    Attempt to recover valid JSON from malformed LLM output.
    Handles: trailing text, missing closing braces, unquoted values,
    markdown code fences, partial output.
    """
    text = raw.strip()

    # Strip markdown fences
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # If model output starts mid-JSON (e.g. after `{` in prompt), prepend brace
    if text and not text.startswith("{"):
        text = "{" + text

    # Extract first JSON-like object
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        text = match.group(0)

    # Fix trailing commas before closing brace: {"a": 1,}  → {"a": 1}
    text = re.sub(r",\s*\}", "}", text)

    # Fix unquoted string values: "type": debit → "type": "debit"
    text = re.sub(
        r'("(?:type|category|merchant)":\s*)([a-zA-Z &]+)([,}])',
        lambda m: f'{m.group(1)}"{m.group(2).strip()}"{m.group(3)}',
        text,
    )

    # Add missing closing brace if unclosed
    open_count  = text.count("{")
    close_count = text.count("}")
    if open_count > close_count:
        text += "}" * (open_count - close_count)

    return text


# ── Rule-based fallback ───────────────────────────────────────────────────────
def _rule_based_fallback(input_text: str) -> ClassificationOutput:
    """
    Deterministic last-resort classifier.
    Uses keyword matching on the original input text.
    Called only when LLM output is unrecoverable.
    """
    from amifi_ai.parsers.sms_parser import (
        _RE_CREDIT,
        _RE_MERCHANT_AT,
        _normalise_merchant,
    )


    text = input_text.lower()

    # Transaction type
    txn_type = "debit"
    if _RE_CREDIT.search(text):
        txn_type = "credit"

    # Merchant
    merchant = "Unknown"
    category = "Miscellaneous"
    m = _RE_MERCHANT_AT.search(input_text)
    if m:
        merchant, category, _ = _normalise_merchant(m.group(1))

    # Confidence is low since we fell back
    return ClassificationOutput(
        merchant=merchant,
        category=category,
        type=txn_type,
        confidence=0.35,
    )


# ── Core validation pipeline ──────────────────────────────────────────────────
def validate_output(
    raw_output: str,
    input_text: str,
    llm_retry_fn=None,      # Optional: callable(corrective_prompt) → str
) -> tuple[ClassificationOutput, str]:
    """
    Full validation pipeline.
    Returns (ClassificationOutput, method_used).
    method_used: "direct" | "repaired" | "llm_retry" | "rule_fallback"
    """

    # ── Step 1: Direct parse ─────────────────────────────────────────
    try:
        data = json.loads(raw_output.strip())
        result = ClassificationOutput(**data)
        logger.info("[VALIDATOR] ✅ Direct parse succeeded.")
        return result, "direct"
    except Exception as e:
        logger.warning(f"[VALIDATOR] Direct parse failed: {e}")

    # ── Step 2: JSON auto-repair ─────────────────────────────────────
    try:
        repaired = _repair_json(raw_output)
        data = json.loads(repaired)
        result = ClassificationOutput(**data)
        logger.info("[VALIDATOR] ✅ Repaired JSON parse succeeded.")
        return result, "repaired"
    except Exception as e:
        logger.warning(f"[VALIDATOR] Repair failed: {e} | Repaired text: {repr(repaired[:100])}")

    # ── Step 3: LLM retry with corrective prompt ─────────────────────
    if llm_retry_fn is not None:
        try:
            corrective_prompt = (
                f"The previous output was malformed JSON. "
                f"Classify this transaction and return ONLY valid JSON:\n"
                f"INPUT: {input_text}\n"
                f'Schema: {{"merchant": string, "category": string, "type": "debit"|"credit", "confidence": float}}\n'
                f"JSON:"
            )
            retry_output = llm_retry_fn(corrective_prompt)
            repaired_retry = _repair_json(retry_output)
            data = json.loads(repaired_retry)
            result = ClassificationOutput(**data)
            logger.info("[VALIDATOR] ✅ LLM retry succeeded.")
            return result, "llm_retry"
        except Exception as e:
            logger.error(f"[VALIDATOR] LLM retry failed: {e}")

    # ── Step 4: Rule-based deterministic fallback ────────────────────
    logger.error(
        f"[VALIDATOR] All parse attempts failed. "
        f"Using rule-based fallback for: {repr(input_text[:60])}"
    )
    result = _rule_based_fallback(input_text)
    return result, "rule_fallback"


def check_confidence_threshold(
    output: ClassificationOutput,
    threshold: float = CONFIDENCE_THRESHOLD,
) -> bool:
    """Returns True if confidence meets threshold."""
    if output.confidence < threshold:
        logger.warning(
            f"[VALIDATOR] Low confidence: {output.confidence} < {threshold}. "
            f"Consider flagging for human review."
        )
        return False
    return True
