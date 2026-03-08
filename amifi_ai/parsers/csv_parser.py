# amifi_ai/parsers/csv_parser.py
"""
CSV Bank Statement Parser.
Handles multiple bank formats, missing fields, type coercion,
schema validation, and duplicate detection.
"""

import hashlib
import io
import logging
from typing import Optional

import pandas as pd
from pydantic import BaseModel, field_validator

logger = logging.getLogger(__name__)

# ── Column alias map ──────────────────────────────────────────────────────────
# Maps various bank column names → our canonical column names
COLUMN_ALIASES: dict[str, list[str]] = {
    "date": [
        "date", "transaction date", "txn date", "value date",
        "trans date", "posting date", "transaction_date",
    ],
    "description": [
        "description", "narration", "particulars", "details",
        "transaction details", "remarks", "transaction_description",
    ],
    "amount": [
        "amount", "transaction amount", "txn amount", "trans amount",
    ],
    "debit": [
        "debit", "debit amount", "withdrawal", "withdrawals",
        "dr", "dr amount", "debit(inr)",
    ],
    "credit": [
        "credit", "credit amount", "deposit", "deposits",
        "cr", "cr amount", "credit(inr)",
    ],
    "balance": [
        "balance", "closing balance", "running balance",
        "available balance", "bal",
    ],
    "ref_id": [
        "ref", "ref no", "reference", "chq no", "cheque no",
        "transaction id", "txn id", "utr",
    ],
}


# ── Pydantic schema for a single validated row ────────────────────────────────
class TransactionRow(BaseModel):
    date: Optional[str]
    description: Optional[str]
    amount: Optional[float]
    type: str                   # "debit" | "credit" | "unknown"
    balance: Optional[float]
    ref_id: Optional[str]
    dedup_hash: str

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        if v not in ("debit", "credit", "unknown"):
            raise ValueError(f"type must be debit/credit/unknown, got: {v}")
        return v

    @field_validator("amount")
    @classmethod
    def validate_amount(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v < 0:
            raise ValueError(f"amount must be non-negative, got: {v}")
        return v


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename DataFrame columns to canonical names using COLUMN_ALIASES.
    Lowercases and strips all column headers first.
    Logs unrecognised columns.
    """
    df.columns = [str(c).strip().lower() for c in df.columns]
    rename_map: dict[str, str] = {}

    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in df.columns and canonical not in rename_map.values():
                rename_map[alias] = canonical
                break

    df = df.rename(columns=rename_map)

    unrecognised = [
        c for c in df.columns
        if c not in COLUMN_ALIASES.keys() and c not in rename_map.values()
    ]
    if unrecognised:
        logger.warning(f"[CSV_PARSER] Unrecognised columns (kept as-is): {unrecognised}")

    return df


def _coerce_amount(value) -> Optional[float]:
    """Convert various amount formats to float. Returns None on failure."""
    if pd.isna(value):
        return None
    str_val = str(value).strip().replace(",", "").replace("₹", "").replace("INR", "").strip()
    if str_val in ("", "-", "nil", "n/a"):
        return None
    try:
        return abs(float(str_val))
    except ValueError:
        logger.warning(f"[CSV_PARSER] Cannot coerce amount: {repr(value)}")
        return None


def _coerce_date(value) -> Optional[str]:
    """Parse various date formats to ISO string YYYY-MM-DD."""
    if pd.isna(value):
        return None
    try:
        return pd.to_datetime(str(value), dayfirst=True, errors="coerce").strftime("%Y-%m-%d")
    except Exception:
        return str(value).strip()


def _determine_type(row: pd.Series) -> tuple[str, Optional[float]]:
    """
    Determine transaction type and canonical amount from a DataFrame row.
    Handles: separate debit/credit columns OR single amount column.
    """
    # Case 1: separate debit / credit columns
    if "debit" in row.index and "credit" in row.index:
        debit_val  = _coerce_amount(row.get("debit"))
        credit_val = _coerce_amount(row.get("credit"))
        if debit_val and debit_val > 0:
            return "debit", debit_val
        if credit_val and credit_val > 0:
            return "credit", credit_val
        return "unknown", None

    # Case 2: single amount column — look for sign or keyword in description
    amount_val = _coerce_amount(row.get("amount"))
    desc       = str(row.get("description", "")).lower()
    if any(kw in desc for kw in ["debit", "dr", "withdrawal", "paid", "purchase"]):
        return "debit", amount_val
    if any(kw in desc for kw in ["credit", "cr", "deposit", "refund", "received"]):
        return "credit", amount_val
    return "unknown", amount_val


def _row_dedup_hash(date: Optional[str], amount: Optional[float], desc: Optional[str]) -> str:
    key = f"{date}|{amount}|{str(desc)[:30]}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _parse_dataframe(df: pd.DataFrame) -> list[dict]:
    """Convert a normalised DataFrame into validated TransactionRow list."""
    df = _normalise_columns(df)

    # Drop fully empty rows
    df = df.dropna(how="all").reset_index(drop=True)
    logger.info(f"[CSV_PARSER] {len(df)} rows after dropping blanks.")

    results: list[dict] = []
    seen_hashes: set[str] = set()
    duplicates = 0
    validation_errors = 0

    for idx, row in df.iterrows():
        txn_type, amount = _determine_type(row)
        date      = _coerce_date(row.get("date"))
        desc      = str(row.get("description", "")).strip() or None
        balance   = _coerce_amount(row.get("balance"))
        ref_id    = str(row.get("ref_id", "")).strip() or None
        dedup_h   = _row_dedup_hash(date, amount, desc)

        # Dedup check
        if dedup_h in seen_hashes:
            logger.warning(f"[CSV_PARSER] Duplicate row at index {idx}: {dedup_h}")
            duplicates += 1
            continue
        seen_hashes.add(dedup_h)

        # Pydantic validation
        try:
            validated = TransactionRow(
                date=date,
                description=desc,
                amount=amount,
                type=txn_type,
                balance=balance,
                ref_id=ref_id,
                dedup_hash=dedup_h,
            )
            results.append(validated.model_dump())
        except Exception as e:
            logger.error(f"[CSV_PARSER] Validation error at row {idx}: {e}")
            validation_errors += 1
            continue

    logger.info(
        f"[CSV_PARSER] ✅ Parsed {len(results)} rows. "
        f"Duplicates skipped: {duplicates}. "
        f"Validation errors: {validation_errors}."
    )
    return results


def parse_csv_file(filepath: str) -> list[dict]:
    """
    Parse a CSV bank statement from a file path.
    Tries multiple encodings and separators automatically.
    """
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    separators = [",", ";", "\t", "|"]

    for enc in encodings:
        for sep in separators:
            try:
                df = pd.read_csv(filepath, encoding=enc, sep=sep, skip_blank_lines=True)
                if df.shape[1] < 2:
                    continue  # not a real CSV with this separator
                logger.info(f"[CSV_PARSER] Loaded {filepath} — encoding={enc} sep={repr(sep)} shape={df.shape}")
                return _parse_dataframe(df)
            except Exception:
                continue

    raise ValueError(
        f"[CSV_PARSER] Could not parse file: {filepath}\n"
        f"Tried encodings={encodings}, separators={separators}"
    )


def parse_csv_bytes(content: bytes, filename: str = "upload.csv") -> list[dict]:
    """
    Parse a CSV bank statement from raw bytes (for FastAPI file upload).
    """
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    separators = [",", ";", "\t", "|"]

    for enc in encodings:
        for sep in separators:
            try:
                df = pd.read_csv(
                    io.BytesIO(content),
                    encoding=enc,
                    sep=sep,
                    skip_blank_lines=True,
                )
                if df.shape[1] < 2:
                    continue
                logger.info(f"[CSV_PARSER] Parsed {filename} — encoding={enc} sep={repr(sep)} shape={df.shape}")
                return _parse_dataframe(df)
            except Exception:
                continue

    raise ValueError(f"[CSV_PARSER] Could not parse uploaded file: {filename}")
