# amifi_ai/tests/test_engine.py
"""Deterministic regression tests for the ONNX inference engine."""

import pytest

from amifi_ai.core.generator import GenerationConfig
from amifi_ai.core.session import find_onnx_file, load_session
from amifi_ai.core.tokenizer import encode_with_validation, load_tokenizer
from amifi_ai.inference.engine import FinEdgeEngine

MODEL_DIR = "models/tinyllama_onnx"


@pytest.fixture(scope="module")
def engine():
    return FinEdgeEngine(MODEL_DIR)


def test_onnx_file_found():
    path = find_onnx_file(MODEL_DIR)
    assert path.exists()
    assert path.suffix == ".onnx"


def test_session_loads():
    session = load_session(MODEL_DIR)
    input_names = {i.name for i in session.get_inputs()}
    assert "input_ids" in input_names
    assert "attention_mask" in input_names


def test_missing_model_dir_raises():
    with pytest.raises(FileNotFoundError, match="Model directory not found"):
        find_onnx_file("models/nonexistent_dir")


def test_tokenizer_loads():
    tok = load_tokenizer(MODEL_DIR)
    assert tok.vocab_size == 32000
    assert tok.eos_token is not None


def test_tokenizer_encode_validation():
    tok = load_tokenizer(MODEL_DIR)
    ids = encode_with_validation(tok, "Hello world")
    assert isinstance(ids, list)
    assert len(ids) > 0


def test_tokenizer_empty_input_raises():
    tok = load_tokenizer(MODEL_DIR)
    with pytest.raises(ValueError, match="Empty"):
        encode_with_validation(tok, "   ")


def test_tokenizer_too_long_raises():
    tok = load_tokenizer(MODEL_DIR)
    with pytest.raises(ValueError, match="too long"):
        encode_with_validation(tok, "word " * 600)


def test_engine_health_check(engine):
    result = engine.health_check()
    assert result["status"] == "ok"
    assert "latency_ms" in result


def test_deterministic_output(engine):
    """Same prompt + same seed must produce identical output every time."""
    config = GenerationConfig(max_new_tokens=20, seed=42)
    r1 = engine.run("Return JSON: {", config=config)
    r2 = engine.run("Return JSON: {", config=config)
    assert r1.token_ids == r2.token_ids, "Generation is not deterministic!"


def test_max_tokens_enforced(engine):
    config = GenerationConfig(max_new_tokens=5, seed=42)
    result = engine.run("Hello world", config=config)
    assert result.num_tokens_generated <= 5


def test_kv_cache_layers_detected(engine):
    from amifi_ai.core.generator import _get_num_layers
    layers = _get_num_layers(engine.session)
    assert layers == 22
