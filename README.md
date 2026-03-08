# FinEdge AI 🏦  
> **Local-first financial transaction classifier powered by quantized TinyLlama ONNX runtime**  
> No cloud API keys. No internet required after setup. Fully offline inference.

---

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12-blue.svg">
  <img src="https://img.shields.io/badge/FastAPI-0.110+-green.svg">
  <img src="https://img.shields.io/badge/ONNX-Runtime-orange.svg">
  <img src="https://img.shields.io/badge/tests-50%20passing-brightgreen.svg">
  <img src="https://img.shields.io/badge/license-MIT-blue.svg">
</p>

---

# 🚀 Overview

FinEdge AI parses:

- 📱 Indian bank SMS alerts  
- 📊 CSV bank statements  

Then classifies each transaction into:

- **Merchant**
- **Category**
- **Transaction Type**
- **Confidence Score**

The system uses a **dual-path inference engine**:

| Path | Trigger | Typical Latency |
|------|---------|----------------|
| ⚡ Regex Fast Path | `use_llm: false` OR high-confidence pattern match | ~0ms |
| 🤖 LLM Path | `use_llm: true` | ~5s (int8 ONNX, CPU-only) |

The LLM path runs a **4-stage guardrail pipeline**, ensuring the system **never crashes**, even with malformed model outputs.

---

# ✨ Key Features

- ⚡ Instant regex classification for major Indian banks (HDFC, SBI, ICICI, Axis, etc.)
- 🤖 Fully offline TinyLlama 1.1B int8 ONNX inference
- 🔒 4-stage JSON guardrail system (parse → repair → retry → fallback)
- 📊 CSV parser with encoding detection + Pydantic validation
- 🧾 Prompt versioning (`v1`, `v2`) with audit trace
- 🔁 KV-cache optimized autoregressive decoding
- 🆔 Request tracing headers (`X-Request-Id`, `X-Response-Time-Ms`)
- ✅ 50 comprehensive pytest tests
- 🐳 Docker-ready

---

# 🗂 Project Structure & System Architecture

<img width="2816" height="1536" alt="Gemini_Generated_Image_qawyuxqawyuxqawy" src="https://github.com/user-attachments/assets/e698635d-8344-4481-bcf3-f84e9f6577be" />

```

finedge_ai/
├── amifi_ai/
│   ├── api/
│   │   ├── main.py
│   │   └── routes.py
│   ├── core/
│   │   ├── session.py
│   │   ├── tokenizer.py
│   │   └── generator.py
│   ├── inference/
│   │   └── engine.py
│   ├── guardrails/
│   │   ├── output_validator.py
│   │   └── schema_enforcer.py
│   ├── parsers/
│   │   ├── sms_parser.py
│   │   └── csv_parser.py
│   ├── prompts/
│   │   ├── v1.yaml
│   │   └── v2.yaml
│   ├── scripts/
│   │   ├── download_model.py
│   │   └── build.sh
│   └── tests/
│       ├── test_engine.py
│       ├── test_api.py
│       ├── test_guardrails.py
│       └── test_parsers.py
├── models/              # Not committed (see setup)
├── artifacts/           # Build output
├── Dockerfile
├── requirements.txt
└── README.md

````

---

# ⚙️ Quickstart

## 1️⃣ Clone & Setup

```bash
git clone https://github.com/your-username/finedge_ai.git
cd finedge_ai
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
````

---

## 2️⃣ Download Model (~600MB, one-time)

```bash
python amifi_ai/scripts/download_model.py
```

Downloads:

```
models/tinyllama_onnx/
```

The **int8 quantized variant** is selected automatically.

---

## 3️⃣ Run the Server

```bash
uvicorn amifi_ai.api.main:app --host 0.0.0.0 --port 8000 --reload
```

* First startup: ~21 seconds (ONNX load)
* Subsequent reloads: Instant

Swagger UI:

```
http://localhost:8000/docs
```

---

# 🌐 API Reference

---

## `GET /`

Returns service metadata.

```json
{
  "service": "FinEdge AI",
  "version": "1.0.0",
  "docs": "/docs"
}
```

---

## `GET /api/v1/health`

Performs live inference smoke test.

```json
{
  "status": "ok",
  "request_id": "8844086a",
  "checks": {
    "model_loaded": true,
    "model_dir": "models/tinyllama_onnx",
    "tokenizer_loaded": true,
    "vocab_size": 32000,
    "inference_ok": true,
    "inference_latency_ms": 653.48
  }
}
```

---

## `POST /api/v1/classify`

Classify single SMS or transaction text.

### Request

```json
{
  "text": "HDFC Bank Rs 2500 debited at AMAZON on 05-03-24",
  "use_llm": false
}
```

### Response

```json
{
  "merchant": "Amazon",
  "category": "Shopping",
  "type": "debit",
  "confidence": 0.99,
  "meets_threshold": true,
  "validation_method": "regex_only",
  "prompt_version": "none",
  "tokens_generated": 0,
  "latency_ms": 0.0,
  "request_id": "43ed91f7"
}
```

### Supported Categories

* Shopping
* Food
* Travel
* Entertainment
* Bills & Utilities
* Health
* Education
* Finance
* Miscellaneous

### Validation Methods

| Method          | Meaning                  |
| --------------- | ------------------------ |
| `regex_only`    | Fast path classification |
| `direct`        | Valid JSON from LLM      |
| `repaired`      | JSON fixed via regex     |
| `llm_retry`     | Second LLM attempt       |
| `rule_fallback` | Regex fallback           |

---

## `POST /api/v1/parse-csv`

Upload a `.csv` bank statement.

```bash
curl -X POST http://localhost:8000/api/v1/parse-csv \
  -F "file=@sample_bank.csv"
```

### Response

```json
{
  "request_id": "c7735e89",
  "filename": "sample_bank.csv",
  "total_rows": 7,
  "parse_time_ms": 240.71,
  "transactions": [...]
}
```

### Error Codes

| Code | Meaning               |
| ---- | --------------------- |
| 415  | Unsupported file type |
| 422  | Validation error      |

Duplicate rows are automatically removed via SHA-256 content hash.

---

# 📡 Response Headers

| Header               | Description                |
| -------------------- | -------------------------- |
| `X-Request-Id`       | Unique 8-char request ID   |
| `X-Response-Time-Ms` | End-to-end processing time |

---

# 🧪 Testing

## Fast Tests (~9s, no model)

```bash
pytest amifi_ai/tests/test_guardrails.py amifi_ai/tests/test_parsers.py -v
```

## Full Suite (~145s)

```bash
pytest amifi_ai/tests/ -v --tb=short
```

### Coverage Summary

| Module     | Tests                |
| ---------- | -------------------- |
| Parsers    | 15                   |
| Guardrails | 14                   |
| Engine     | 11                   |
| API        | 10                   |
| **Total**  | **50 (All Passing)** |

---

# 🐳 Docker

## Build

```bash
docker build -t finedge-ai:1.0.0 .
```

## Run

```bash
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  finedge-ai:1.0.0
```

---

# 🔧 Configuration

Defined in `pyproject.toml` under `[tool.finedge]` or via environment variables:

| Parameter              | Default                 | Description             |
| ---------------------- | ----------------------- | ----------------------- |
| `MODEL_DIR`            | `models/tinyllama_onnx` | ONNX model path         |
| `MAX_NEW_TOKENS`       | `80`                    | Max tokens per LLM call |
| `CONFIDENCE_THRESHOLD` | `0.5`                   | Review threshold        |
| `PROMPT_VERSION`       | `v2`                    | Active prompt           |

---

# 🏗 Architecture Deep Dive

## KV-Cache Generation

`core/generator.py` implements full autoregressive decoding:

* Step 0 → Entire prompt passed
* Steps 1+ → Only last token passed
* KV cache maintained across 22 transformer layers
* Linear memory growth instead of quadratic

Result: Efficient CPU inference without GPU.

---

## Guardrail Pipeline

```
LLM Output
   │
   ├── Direct JSON Parse
   │
   ├── JSON Repair (regex)
   │
   ├── LLM Retry (short prompt)
   │
   └── Rule-based Fallback
```

A structured `ClassificationOutput` is **always returned**.

---

# 📦 Build Artifact

```bash
bash amifi_ai/scripts/build.sh
```

Produces:

```
artifacts/finedge_ai_v1.0.0.tar.gz
artifacts/finedge_ai_v1.0.0.sha256
```

Models excluded (~82MB artifact).

---

# 🔐 License

MIT © 2026 Ayaan Shaheer

---

# ⭐ Why FinEdge AI?

* Fully offline
* Production-ready guardrails
* Deterministic inference
* Test-covered
* Dockerized
* Built for Indian financial data

---


If this project helped you, consider ⭐ starring the repo.
