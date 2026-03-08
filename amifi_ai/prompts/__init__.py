# amifi_ai/prompts/__init__.py
"""
Versioned prompt loader.
Loads prompts/v1.json, v2.json dynamically.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent
_cache: dict[str, dict] = {}


def load_prompt(version: str = "v2") -> dict:
    """
    Load a versioned prompt config by version string.
    Caches after first load.
    """
    if version in _cache:
        return _cache[version]

    prompt_file = PROMPTS_DIR / f"{version}.json"
    if not prompt_file.exists():
        available = [f.stem for f in PROMPTS_DIR.glob("*.json")]
        raise FileNotFoundError(
            f"[PROMPTS] Prompt version '{version}' not found. "
            f"Available: {available}"
        )

    with open(prompt_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    _cache[version] = config
    logger.info(f"[PROMPTS] Loaded prompt version={version} name={config.get('name')}")
    return config


def build_prompt(input_text: str, version: str = "v2") -> tuple[str, dict]:
    """
    Build a ready-to-send prompt string from template + input.
    Returns (prompt_string, prompt_config).
    """
    config = load_prompt(version)
    template = config["template"]
    prompt = template.replace("{input_text}", input_text.strip())
    return prompt, config
