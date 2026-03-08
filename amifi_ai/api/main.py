# amifi_ai/api/main.py
"""
FastAPI application entrypoint.
Wires middleware, routers, lifespan startup/shutdown.
"""

import logging
import sys
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from amifi_ai.api.middleware import RequestLoggingMiddleware
from amifi_ai.api.routes import router
from amifi_ai.inference.engine import get_engine

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

MODEL_DIR = "models/tinyllama_onnx"


# ── Lifespan: warm up engine on startup ──────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load model into memory at startup so first request isn't slow.
    """
    logger.info("[STARTUP] FinEdge AI starting up...")
    t_start = time.perf_counter()
    try:
        get_engine(MODEL_DIR)
        elapsed = (time.perf_counter() - t_start) * 1000
        logger.info(f"[STARTUP] ✅ Engine ready in {elapsed:.0f}ms")
    except Exception as e:
        logger.error(f"[STARTUP] ❌ Engine failed to load: {e}")
        # Don't crash — health endpoint will report degraded

    yield  # ← application runs here

    logger.info("[SHUTDOWN] FinEdge AI shutting down.")


# ── App factory ───────────────────────────────────────────────────────────────
def create_app() -> FastAPI:
    app = FastAPI(
        title="FinEdge AI",
        description="Production-ready ONNX Financial LLM Engine",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Middleware (order matters — outermost first)
    app.add_middleware(RequestLoggingMiddleware)

    # Routes
    app.include_router(router, prefix="/api/v1")

    # Root redirect
    @app.get("/", include_in_schema=False)
    async def root():
        return JSONResponse({"service": "FinEdge AI", "version": "1.0.0", "docs": "/docs"})

    return app


app = create_app()
