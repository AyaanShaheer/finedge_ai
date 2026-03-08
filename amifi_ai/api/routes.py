# amifi_ai/api/routes.py
"""
All API route handlers.
POST /classify     — single SMS/text classification
POST /parse-csv    — CSV bank statement upload
GET  /health       — model + system health check
"""

import logging
import time
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, field_validator

from amifi_ai.guardrails.schema_enforcer import classify_transaction
from amifi_ai.inference.engine import get_engine
from amifi_ai.parsers.csv_parser import parse_csv_bytes
from amifi_ai.parsers.sms_parser import parse_sms

logger = logging.getLogger(__name__)
router = APIRouter()

MODEL_DIR = "models/tinyllama_onnx"
MAX_TEXT_LENGTH = 500
MAX_CSV_SIZE_BYTES = 10 * 1024 * 1024  # 10MB


# ── Request / Response schemas ────────────────────────────────────────────────

class ClassifyRequest(BaseModel):
    text: str
    prompt_version: str = "v2"
    use_llm: bool = True   # False = regex-only fast path

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("text cannot be empty.")
        if len(v) > MAX_TEXT_LENGTH:
            raise ValueError(
                f"text too long ({len(v)} chars). Max: {MAX_TEXT_LENGTH}."
            )
        return v


class ClassifyResponse(BaseModel):
    merchant: str
    category: str
    type: str
    confidence: float
    meets_threshold: bool
    validation_method: str
    prompt_version: str
    tokens_generated: int
    latency_ms: float
    request_id: str


class SMSParseResponse(BaseModel):
    amount: float | None
    merchant: str | None
    category: str
    type: str
    confidence: float
    date: str | None
    ref_id: str | None
    account_last4: str | None
    parse_method: str
    dedup_hash: str
    is_duplicate: bool
    request_id: str


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/classify", response_model=ClassifyResponse)
async def classify_endpoint(body: ClassifyRequest, request: Request):
    """
    Classify a single financial SMS or transaction text.
    Routes through full LLM + guardrail pipeline by default.
    Set use_llm=false for fast regex-only path.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    logger.info(f"[/classify] [{request_id}] text={repr(body.text[:60])}")

    try:
        if body.use_llm:
            engine = get_engine(MODEL_DIR)
            result = classify_transaction(
                input_text=body.text,
                engine=engine,
                prompt_version=body.prompt_version,
            )
        else:
            # Fast path: regex only, no LLM
            txn = parse_sms(body.text)
            result = {
                "merchant": txn.merchant or "Unknown",
                "category": txn.category,
                "type": txn.type if txn.type != "unknown" else "debit",
                "confidence": txn.confidence,
                "meets_threshold": txn.confidence >= 0.50,
                "validation_method": "regex_only",
                "prompt_version": "none",
                "tokens_generated": 0,
                "latency_ms": 0.0,
            }

        return ClassifyResponse(**result, request_id=request_id)

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"[/classify] [{request_id}] ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/parse-csv")
async def parse_csv_endpoint(
    request: Request,
    file: Annotated[UploadFile, File(description="CSV bank statement file")],
):
    """
    Upload a CSV bank statement.
    Returns structured JSON list of validated transactions.
    Handles multiple bank formats, deduplication, schema validation.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    logger.info(f"[/parse-csv] [{request_id}] filename={file.filename}")

    # File type check
    if file.filename and not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=415,
            detail=f"Only .csv files accepted. Got: {file.filename}",
        )

    # Size check
    content = await file.read()
    if len(content) > MAX_CSV_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {len(content)} bytes. Max: {MAX_CSV_SIZE_BYTES}.",
        )
    if len(content) == 0:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")

    try:
        t_start = time.perf_counter()
        transactions = parse_csv_bytes(content, filename=file.filename or "upload.csv")
        elapsed = (time.perf_counter() - t_start) * 1000

        return {
            "request_id": request_id,
            "filename": file.filename,
            "total_rows": len(transactions),
            "parse_time_ms": round(elapsed, 2),
            "transactions": transactions,
        }

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"[/parse-csv] [{request_id}] ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_endpoint(request: Request):
    """
    Full system health check.
    Tests: model loaded, tokenizer loaded, inference functional.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    checks: dict = {}

    # Check 1: Engine / model loadable
    try:
        engine = get_engine(MODEL_DIR)
        checks["model_loaded"] = True
        checks["model_dir"] = MODEL_DIR
    except Exception as e:
        checks["model_loaded"] = False
        checks["model_error"] = str(e)

    # Check 2: Tokenizer
    try:
        vocab_size = engine.tokenizer.vocab_size
        checks["tokenizer_loaded"] = True
        checks["vocab_size"] = vocab_size
    except Exception as e:
        checks["tokenizer_loaded"] = False
        checks["tokenizer_error"] = str(e)

    # Check 3: Live inference smoke test
    try:
        hc = engine.health_check()
        checks["inference_ok"] = hc["status"] == "ok"
        checks["inference_latency_ms"] = hc.get("latency_ms")
        checks["inference_sample"] = hc.get("test_output", "")[:40]
    except Exception as e:
        checks["inference_ok"] = False
        checks["inference_error"] = str(e)

    all_ok = all([
        checks.get("model_loaded"),
        checks.get("tokenizer_loaded"),
        checks.get("inference_ok"),
    ])

    return {
        "status": "ok" if all_ok else "degraded",
        "request_id": request_id,
        "checks": checks,
    }
