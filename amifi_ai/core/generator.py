# amifi_ai/core/generator.py
"""
Deterministic KV-cache-aware token-by-token generation loop.

This model uses the HuggingFace Optimum "with-past" ONNX pattern:
  - Step 0  : full prompt → input_ids=[all tokens], past_key_values=empty
  - Step N>0 : single token → input_ids=[last token], past_key_values=present from step N-1

past_key_values shape : (batch=1, num_heads=4, past_seq_len, head_dim=64)
present shape         : (batch=1, num_heads=4, past_seq_len+1, head_dim=64)
"""

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import onnxruntime as ort
from transformers import PreTrainedTokenizer

from amifi_ai.core.tokenizer import MAX_OUTPUT_TOKENS

logger = logging.getLogger(__name__)

# ── Model constants (read from session at runtime) ──────────────────
NUM_LAYERS   = 22   # past_key_values.0 … past_key_values.21
NUM_HEADS    = 4
HEAD_DIM     = 64


def _get_num_layers(session: ort.InferenceSession) -> int:
    """Auto-detect number of KV-cache layers from session inputs."""
    return sum(
        1 for inp in session.get_inputs()
        if inp.name.startswith("past_key_values.") and inp.name.endswith(".key")
    )


def _empty_past(num_layers: int) -> dict[str, np.ndarray]:
    """
    Build empty past_key_values for the first generation step.
    Shape: (batch=1, num_heads=4, past_seq_len=0, head_dim=64)
    """
    empty = np.zeros((1, NUM_HEADS, 0, HEAD_DIM), dtype=np.float32)
    past = {}
    for i in range(num_layers):
        past[f"past_key_values.{i}.key"]   = empty
        past[f"past_key_values.{i}.value"] = empty
    return past


def _present_to_past(
    outputs: list,
    session: ort.InferenceSession,
    num_layers: int,
) -> dict[str, np.ndarray]:
    """
    Extract present.*.key/value from session outputs and rename to
    past_key_values.*.key/value for the next step.

    Output tensor order: [logits, present.0.key, present.0.value, present.1.key, ...]
    """
    output_names = [o.name for o in session.get_outputs()]
    past = {}
    for i in range(num_layers):
        key_name = f"present.{i}.key"
        val_name = f"present.{i}.value"
        key_idx  = output_names.index(key_name)
        val_idx  = output_names.index(val_name)
        past[f"past_key_values.{i}.key"]   = outputs[key_idx]
        past[f"past_key_values.{i}.value"] = outputs[val_idx]
    return past


@dataclass
class GenerationConfig:
    max_new_tokens: int = MAX_OUTPUT_TOKENS
    seed: int = 42
    stop_token_ids: list[int] = field(default_factory=list)
    echo_prompt: bool = False


@dataclass
class GenerationResult:
    text: str
    token_ids: list[int]
    num_tokens_generated: int
    latency_ms: float
    stopped_by: str   # "stop_token" | "max_tokens" | "eos"


def generate(
    session: ort.InferenceSession,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    config: GenerationConfig | None = None,
) -> GenerationResult:
    """
    KV-cache autoregressive generation.
    Fully deterministic: argmax (temperature=0), seeded.
    """
    if config is None:
        config = GenerationConfig()

    np.random.seed(config.seed)

    num_layers = _get_num_layers(session)
    logger.info(f"[GEN] Detected {num_layers} KV-cache layers.")

    # Build stop token set
    stop_ids = set(config.stop_token_ids)
    if tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)
    logger.debug(f"[GEN] Stop IDs: {stop_ids}")

    # Encode full prompt
    prompt_ids: list[int] = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_len = len(prompt_ids)
    logger.info(f"[GEN] Prompt length: {prompt_len} tokens")

    generated_ids: list[int] = []
    stop_reason = "max_tokens"

    # ── Initialise empty KV cache ─────────────────────────────────────
    past = _empty_past(num_layers)

    t_start = time.perf_counter()

    # ── GENERATION LOOP ───────────────────────────────────────────────
    for step in range(config.max_new_tokens):

        if step == 0:
            # First step: feed full prompt
            current_input_ids = np.array([prompt_ids], dtype=np.int64)
        else:
            # Subsequent steps: feed ONLY the last generated token
            current_input_ids = np.array([[generated_ids[-1]]], dtype=np.int64)

        # attention_mask covers full sequence seen so far
        total_len     = prompt_len + step
        attention_mask = np.ones((1, total_len), dtype=np.int64)

        # position_ids for the current token(s)
        if step == 0:
            position_ids = np.arange(prompt_len, dtype=np.int64).reshape(1, -1)
        else:
            position_ids = np.array([[prompt_len + step - 1]], dtype=np.int64)

        feed = {
            "input_ids":      current_input_ids,
            "attention_mask": attention_mask,
            "position_ids":   position_ids,
            **past,
        }

        # ── Run inference ─────────────────────────────────────────────
        try:
            outputs = session.run(None, feed)
        except Exception as e:
            raise RuntimeError(
                f"[GEN] session.run() failed at step {step}.\n"
                f"input_ids shape   : {current_input_ids.shape}\n"
                f"attention_mask    : {attention_mask.shape}\n"
                f"position_ids      : {position_ids.shape}\n"
                f"past[0].key shape : {past['past_key_values.0.key'].shape}\n"
                f"Error: {e}"
            ) from e

        # ── Argmax over last token logits ─────────────────────────────
        logits         = outputs[0]              # (1, seq_len, vocab_size)
        last_logits    = logits[0, -1, :]        # (vocab_size,)
        next_token_id  = int(np.argmax(last_logits))

        logger.debug(
            f"[GEN] Step {step:03d}: token={next_token_id} "
            f"({repr(tokenizer.decode([next_token_id]))})"
        )

        # ── Update KV cache for next step ─────────────────────────────
        past = _present_to_past(outputs, session, num_layers)

        # ── Stop condition check ──────────────────────────────────────
        if next_token_id in stop_ids:
            stop_reason = "stop_token" if next_token_id in config.stop_token_ids else "eos"
            logger.info(f"[GEN] EOS at step {step} — reason: {stop_reason}")
            break

        generated_ids.append(next_token_id)

    # ── END LOOP ──────────────────────────────────────────────────────

    latency_ms = (time.perf_counter() - t_start) * 1000

    output_ids  = (prompt_ids + generated_ids) if config.echo_prompt else generated_ids
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)

    result = GenerationResult(
        text=output_text.strip(),
        token_ids=generated_ids,
        num_tokens_generated=len(generated_ids),
        latency_ms=latency_ms,
        stopped_by=stop_reason,
    )

    logger.info(
        f"[GEN] ✅ Done — {result.num_tokens_generated} tokens "
        f"in {result.latency_ms:.1f}ms  stopped_by={result.stopped_by}"
    )
    return result
