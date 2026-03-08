"""
Script to download TinyLlama ONNX model weights to models/tinyllama_onnx/
Run once: python amifi_ai/scripts/download_model.py
"""
from pathlib import Path

MODEL_DIR = Path("models/tinyllama_onnx")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("[INFO] Downloading TinyLlama ONNX via huggingface_hub...")

try:
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="onnx-community/TinyLlama_v1.1-ONNX",
        local_dir=str(MODEL_DIR),
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],
    )
    print(f"[OK] Model downloaded to {MODEL_DIR}")
except ImportError:
    print("[ERROR] huggingface_hub not installed. Run: pip install huggingface_hub")
    raise
