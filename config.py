# config.py
from dotenv import load_dotenv
import os

load_dotenv()

MODEL_DIR   = os.getenv("MODEL_DIR", "models/tinyllama_onnx")
LOG_LEVEL   = os.getenv("LOG_LEVEL", "INFO")
HOST        = os.getenv("HOST", "0.0.0.0")
PORT        = int(os.getenv("PORT", "8000"))
WORKERS     = int(os.getenv("WORKERS", "1"))
