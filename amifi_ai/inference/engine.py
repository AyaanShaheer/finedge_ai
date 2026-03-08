# amifi_ai/inference/engine.py
"""
Top-level inference engine.
Loads model once, exposes .run(prompt) for all callers.
"""

import logging
from functools import lru_cache

from amifi_ai.core.generator import GenerationConfig, GenerationResult, generate
from amifi_ai.core.session import load_session
from amifi_ai.core.tokenizer import load_tokenizer

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = "models/tinyllama_onnx"


class FinEdgeEngine:
    """
    Production inference engine.
    Load once at startup — thread-safe for read operations.
    """

    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR):
        logger.info(f"[ENGINE] Initializing FinEdgeEngine from: {model_dir}")
        self.model_dir = model_dir
        self.session = load_session(model_dir)
        self.tokenizer = load_tokenizer(model_dir)
        self._healthy = True
        logger.info("[ENGINE] ✅ Ready.")

    def run(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        if not self._healthy:
            raise RuntimeError("[ENGINE] Engine is in unhealthy state. Restart required.")
        return generate(self.session, self.tokenizer, prompt, config)

    def health_check(self) -> dict:
        """Run a minimal inference to verify model is functional."""
        try:
            result = self.run(
                "Return JSON: {",
                config=GenerationConfig(max_new_tokens=10, seed=0),
            )
            return {
                "status": "ok",
                "model_dir": self.model_dir,
                "test_output": result.text[:50],
                "latency_ms": round(result.latency_ms, 2),
            }
        except Exception as e:
            self._healthy = False
            return {"status": "error", "detail": str(e)}


@lru_cache(maxsize=1)
def get_engine(model_dir: str = DEFAULT_MODEL_DIR) -> FinEdgeEngine:
    """Singleton engine — cached after first load."""
    return FinEdgeEngine(model_dir)
