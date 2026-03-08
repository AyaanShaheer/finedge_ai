# amifi_ai/guardrails/schema_enforcer.py
"""
Schema enforcer + full classification pipeline.
Wires together: prompt builder → LLM → validator → schema.
"""

import logging

from amifi_ai.core.generator import GenerationConfig
from amifi_ai.guardrails.output_validator import (
    check_confidence_threshold,
    validate_output,
)
from amifi_ai.prompts import build_prompt

logger = logging.getLogger(__name__)


def classify_transaction(
    input_text: str,
    engine,                         # FinEdgeEngine instance
    prompt_version: str = "v2",
    confidence_threshold: float = 0.50,
) -> dict:
    """
    Full classification pipeline:
      1. Build structured prompt
      2. Run LLM inference
      3. Validate + repair output
      4. Schema enforce via Pydantic
      5. Confidence gate

    Returns enriched dict with metadata.
    """
    if not input_text or not input_text.strip():
        raise ValueError("[ENFORCER] Empty input text.")

    # ── Build prompt ──────────────────────────────────────────────────
    prompt, prompt_config = build_prompt(input_text, version=prompt_version)

    # Stop tokens from prompt config
    stop_token_ids = []
    tokenizer = engine.tokenizer
    for stop_str in prompt_config.get("stop_tokens", []):
        ids = tokenizer.encode(stop_str, add_special_tokens=False)
        if ids:
            stop_token_ids.append(ids[0])

    gen_config = GenerationConfig(
        max_new_tokens=prompt_config.get("max_new_tokens", 80),
        seed=prompt_config.get("seed", 42),
        stop_token_ids=stop_token_ids,
    )

    logger.info(f"[ENFORCER] Running inference (prompt_version={prompt_version})")
    result = engine.run(prompt, config=gen_config)
    raw_output = result.text

    logger.debug(f"[ENFORCER] Raw LLM output: {repr(raw_output[:200])}")

    # ── Define retry function using same engine ───────────────────────
    def llm_retry_fn(corrective_prompt: str) -> str:
        retry_result = engine.run(
            corrective_prompt,
            config=GenerationConfig(max_new_tokens=80, seed=99),
        )
        return retry_result.text

    # ── Validate + repair ─────────────────────────────────────────────
    output, method = validate_output(raw_output, input_text, llm_retry_fn)

    # ── Confidence check ──────────────────────────────────────────────
    meets_threshold = check_confidence_threshold(output, confidence_threshold)

    return {
        "merchant": output.merchant,
        "category": output.category,
        "type": output.type,
        "confidence": output.confidence,
        "meets_threshold": meets_threshold,
        "validation_method": method,
        "prompt_version": prompt_version,
        "tokens_generated": result.num_tokens_generated,
        "latency_ms": round(result.latency_ms, 2),
    }
