# amifi_ai/core/session.py
"""
ONNX InferenceSession loader.
Auto-detects model.onnx or nested ONNX files within subdirectories.
Handles model.onnx + model.onnx.data external tensor pattern.
"""

import logging
from pathlib import Path

import onnxruntime as ort

logger = logging.getLogger(__name__)


def find_onnx_file(model_dir: str) -> Path:
    """
    Search for any .onnx file in model_dir and its subdirectories.
    Prefers quantized variants for CPU efficiency.
    Returns the resolved Path to the .onnx file.
    """
    model_path = Path(model_dir)

    if not model_path.exists():
        raise FileNotFoundError(
            f"[SESSION] Model directory not found: {model_path.resolve()}\n"
            f"Run: python amifi_ai/scripts/download_model.py"
        )

    # Search recursively for all .onnx files
    all_onnx = list(model_path.rglob("*.onnx"))

    if not all_onnx:
        raise FileNotFoundError(
            f"[SESSION] No .onnx file found anywhere under: {model_path.resolve()}\n"
            f"Contents: {[f.name for f in model_path.rglob('*') if f.is_file()]}"
        )

    logger.info(f"[SESSION] Found ONNX files: {[str(f) for f in all_onnx]}")

    # Priority: prefer quantized (int4/int8/q4) for CPU, else take first
    priority_keywords = ["int4", "int8", "q4", "quantized", "quant"]
    for keyword in priority_keywords:
        for f in all_onnx:
            if keyword in f.name.lower() or keyword in str(f.parent).lower():
                logger.info(f"[SESSION] Selected quantized model: {f}")
                return f

    # Fallback: just use the first one found
    selected = all_onnx[0]
    logger.info(f"[SESSION] Selected model: {selected}")
    return selected


def load_session(model_dir: str, optimization_level: int = 1) -> ort.InferenceSession:
    """
    Load ONNX InferenceSession with CPU provider only.
    Auto-detects the correct .onnx file path.
    """
    onnx_file = find_onnx_file(model_dir)
    onnx_path_str = str(onnx_file)

    # Check for external tensor data file alongside the .onnx
    data_file = onnx_file.parent / (onnx_file.name + ".data")
    if data_file.exists():
        logger.info(f"[SESSION] External tensor file found: {data_file}")
    else:
        logger.info("[SESSION] No .onnx.data sidecar — model is self-contained.")

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel(optimization_level)
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 1
    opts.log_severity_level = 2

    logger.info(f"[SESSION] Loading: {onnx_path_str}")
    logger.info("[SESSION] This may take 30-90s on first load...")

    try:
        session = ort.InferenceSession(
            onnx_path_str,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
    except Exception as e:
        raise RuntimeError(
            f"[SESSION] Failed to load ONNX model.\n"
            f"File: {onnx_path_str}\n"
            f"Error: {e}\n"
            f"If you see 'external data' errors, ensure .onnx.data is "
            f"in the same folder as the .onnx file."
        ) from e

    for inp in session.get_inputs():
        logger.info(f"[SESSION] Input  → name={inp.name}  shape={inp.shape}  type={inp.type}")
    for out in session.get_outputs():
        logger.info(f"[SESSION] Output → name={out.name}  shape={out.shape}  type={out.type}")

    logger.info("[SESSION] ✅ InferenceSession loaded successfully.")
    return session
