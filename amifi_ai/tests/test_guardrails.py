# amifi_ai/tests/test_guardrails.py
"""Tests for output validator and JSON repair guardrails."""

import pytest

from amifi_ai.guardrails.output_validator import (
    ClassificationOutput,
    check_confidence_threshold,
    validate_output,
)
from amifi_ai.prompts import build_prompt, load_prompt

# ── Repair tests ──────────────────────────────────────────────────────────────

def test_repair_trailing_comma():
    raw = '{"merchant": "Amazon", "category": "Shopping", "type": "debit", "confidence": 0.9,}'
    result, method = validate_output(raw, "test input")
    assert method == "repaired"
    assert result.merchant == "Amazon"


def test_repair_markdown_fence():
    raw = '```json\n{"merchant": "Uber", "category": "Transport", "type": "debit", "confidence": 0.75}\n```'
    result, method = validate_output(raw, "test")
    assert method == "repaired"


def test_repair_missing_brace():
    raw = '{"merchant": "Flipkart", "category": "Shopping", "type": "debit", "confidence": 0.91'
    result, method = validate_output(raw, "test")
    assert method == "repaired"
    assert result.confidence == 0.91


def test_direct_parse_perfect_json():
    raw = '{"merchant": "Amazon", "category": "Shopping", "type": "debit", "confidence": 0.93}'
    result, method = validate_output(raw, "test")
    assert method == "direct"


def test_rule_fallback_on_garbage():
    raw = "I cannot process this request at all."
    result, method = validate_output(raw, "HDFC Rs 500 debited at Amazon")
    assert method == "rule_fallback"
    assert result.confidence == 0.35


# ── Schema validation ─────────────────────────────────────────────────────────

def test_invalid_type_rejected():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        ClassificationOutput(merchant="X", category="Shopping", type="spend", confidence=0.9)


def test_unknown_category_normalised():
    raw = '{"merchant": "X", "category": "Totally Unknown Cat", "type": "debit", "confidence": 0.8}'
    result, _ = validate_output(raw, "test")
    assert result.category == "Miscellaneous"


def test_confidence_clamped():
    raw = '{"merchant": "X", "category": "Shopping", "type": "debit", "confidence": 1.5}'
    result, _ = validate_output(raw, "test")
    assert result.confidence == 1.0


def test_confidence_threshold_pass():
    out = ClassificationOutput(merchant="X", category="Shopping", type="debit", confidence=0.85)
    assert check_confidence_threshold(out) is True


def test_confidence_threshold_fail():
    out = ClassificationOutput(merchant="X", category="Shopping", type="debit", confidence=0.30)
    assert check_confidence_threshold(out) is False


# ── Prompt loader ─────────────────────────────────────────────────────────────

def test_prompt_v1_loads():
    p = load_prompt("v1")
    assert p["version"] == "1.0"
    assert "{input_text}" in p["template"]


def test_prompt_v2_loads():
    p = load_prompt("v2")
    assert p["version"] == "2.0"


def test_prompt_missing_version_raises():
    with pytest.raises(FileNotFoundError, match="not found"):
        load_prompt("v99")


def test_build_prompt_substitutes():
    prompt, config = build_prompt("Rs 500 debited at Amazon", version="v1")
    assert "Rs 500 debited at Amazon" in prompt
    assert "{input_text}" not in prompt
