# amifi_ai/core/tokenizer.py
"""
Tokenizer wrapper with:
- Version validation
- Special token verification
- Input length guardrails
- Debug logging for input_ids
"""

import logging
from pathlib import Path

from transformers import AutoTokenizer, PreTrainedTokenizer

logger = logging.getLogger(__name__)

MAX_INPUT_TOKENS = 512
MAX_OUTPUT_TOKENS = 256

REQUIRED_SPECIAL_TOKENS = ["eos_token", "bos_token", "pad_token"]


def load_tokenizer(model_dir: str) -> PreTrainedTokenizer:
    """
    Load tokenizer. Searches model_dir and parent for tokenizer.json.
    """
    model_path = Path(model_dir)

    # Search for tokenizer.json in dir and one level up/down
    search_paths = [
        model_path,
        model_path.parent,
        *list(model_path.rglob("tokenizer.json"))
    ]

    tokenizer_dir = None
    for p in search_paths:
        candidate = p if p.is_dir() else p.parent
        if (candidate / "tokenizer.json").exists():
            tokenizer_dir = candidate
            break

    if tokenizer_dir is None:
        raise FileNotFoundError(
            f"[TOKENIZER] tokenizer.json not found under {model_path.resolve()}\n"
            f"Files present: {[f.name for f in model_path.rglob('*') if f.is_file()]}"
        )

    logger.info(f"[TOKENIZER] Loading from: {tokenizer_dir}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_dir),
        local_files_only=True,
        trust_remote_code=False,
    )

    # Validate special tokens
    missing_tokens = []
    for token_attr in REQUIRED_SPECIAL_TOKENS:
        token_val = getattr(tokenizer, token_attr, None)
        if token_val is None:
            missing_tokens.append(token_attr)
            logger.warning(f"[TOKENIZER] Missing special token: {token_attr}")
        else:
            logger.info(f"[TOKENIZER] {token_attr} = {repr(token_val)}")

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("[TOKENIZER] pad_token set to eos_token fallback.")

    logger.info(f"[TOKENIZER] ✅ Loaded. vocab_size={tokenizer.vocab_size}")
    return tokenizer



def encode_with_validation(
    tokenizer: PreTrainedTokenizer,
    text: str,
    debug: bool = False,
) -> list[int]:
    """
    Encode text to input_ids with length validation.
    Raises ValueError if input exceeds MAX_INPUT_TOKENS.
    """
    if not text or not text.strip():
        raise ValueError("[TOKENIZER] Empty or whitespace-only input text.")

    encoded = tokenizer.encode(text, add_special_tokens=True)

    if debug:
        logger.debug(f"[TOKENIZER] Raw text: {repr(text[:100])}")
        logger.debug(f"[TOKENIZER] input_ids ({len(encoded)} tokens): {encoded[:20]}{'...' if len(encoded) > 20 else ''}")
        # Off-by-one check: log first/last tokens
        logger.debug(
            f"[TOKENIZER] First token: id={encoded[0]} → {repr(tokenizer.decode([encoded[0]]))}"
        )
        logger.debug(
            f"[TOKENIZER] Last  token: id={encoded[-1]} → {repr(tokenizer.decode([encoded[-1]]))}"
        )

    if len(encoded) > MAX_INPUT_TOKENS:
        raise ValueError(
            f"[TOKENIZER] Input too long: {len(encoded)} tokens > MAX_INPUT_TOKENS={MAX_INPUT_TOKENS}. "
            f"Truncate your input or increase the limit."
        )

    return encoded
