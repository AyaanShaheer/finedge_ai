# amifi_ai/parsers/sms_parser.py
"""
Financial SMS Parser.
Layer 1: Regex extraction (fast, deterministic)
Layer 2: LLM fallback for unmatched patterns
Includes merchant normalization + deduplication.
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ── Merchant normalisation map ────────────────────────────────────────────────
MERCHANT_NORMALISATION: dict[str, str] = {
    # E-commerce
    "amazon": "Amazon", "amzn": "Amazon", "amazon pay": "Amazon",
    "flipkart": "Flipkart", "myntra": "Myntra", "meesho": "Meesho",
    "nykaa": "Nykaa", "ajio": "Ajio",
    # Food
    "swiggy": "Swiggy", "zomato": "Zomato", "dominos": "Domino's",
    "mcdonalds": "McDonald's", "mcdonald": "McDonald's", "kfc": "KFC",
    "starbucks": "Starbucks",
    # Transport
    "uber": "Uber", "ola": "Ola", "rapido": "Rapido",
    "irctc": "IRCTC", "makemytrip": "MakeMyTrip", "redbus": "RedBus",
    # Utilities & Recharges
    "airtel": "Airtel", "jio": "Jio", "bsnl": "BSNL", "vi": "Vi",
    "electricity": "Electricity Board", "bescom": "BESCOM",
    # Finance
    "paytm": "Paytm", "phonepe": "PhonePe", "gpay": "Google Pay",
    "googlepay": "Google Pay", "razorpay": "Razorpay",
    # Groceries
    "bigbasket": "BigBasket", "blinkit": "Blinkit", "zepto": "Zepto",
    "dmart": "D-Mart", "reliance fresh": "Reliance Fresh",
    # Fuel
    "indian oil": "Indian Oil", "hp petrol": "HP Petrol",
    "bharat petroleum": "Bharat Petroleum", "shell": "Shell",
}

# ── Category mapping ──────────────────────────────────────────────────────────
CATEGORY_MAP: dict[str, str] = {
    "Amazon": "Shopping", "Flipkart": "Shopping", "Myntra": "Shopping",
    "Meesho": "Shopping", "Nykaa": "Shopping", "Ajio": "Shopping",
    "Swiggy": "Food & Dining", "Zomato": "Food & Dining",
    "Domino's": "Food & Dining", "McDonald's": "Food & Dining",
    "KFC": "Food & Dining", "Starbucks": "Food & Dining",
    "Uber": "Transport", "Ola": "Transport", "Rapido": "Transport",
    "IRCTC": "Travel", "MakeMyTrip": "Travel", "RedBus": "Travel",
    "Airtel": "Utilities", "Jio": "Utilities", "BSNL": "Utilities",
    "Vi": "Utilities", "Electricity Board": "Utilities", "BESCOM": "Utilities",
    "Paytm": "Finance", "PhonePe": "Finance", "Google Pay": "Finance",
    "BigBasket": "Groceries", "Blinkit": "Groceries", "Zepto": "Groceries",
    "D-Mart": "Groceries", "Reliance Fresh": "Groceries",
    "Indian Oil": "Fuel", "HP Petrol": "Fuel",
    "Bharat Petroleum": "Fuel", "Shell": "Fuel",
}

# ── Regex patterns ────────────────────────────────────────────────────────────
# Amount patterns
_RE_AMOUNT = re.compile(
    r"(?:rs\.?|inr|₹)\s*([\d,]+(?:\.\d{1,2})?)",
    re.IGNORECASE,
)
# Transaction type
_RE_DEBIT = re.compile(
    r"\b(debited|debit|spent|paid|payment|withdrawn|purchase|charged)\b",
    re.IGNORECASE,
)
_RE_CREDIT = re.compile(
    r"\b(credited|credit|received|refund|cashback|deposited|added)\b",
    re.IGNORECASE,
)
# Merchant extraction — "at MERCHANT", "to MERCHANT", "for MERCHANT"
_RE_MERCHANT_AT = re.compile(
    r"\b(?:at|to|for|@)\s+([A-Z][A-Za-z0-9\s\-&\.]{1,30}?)(?:\s+on|\s+\d|\.|\,|$)",
    re.IGNORECASE,
)
# Date patterns
_RE_DATE = re.compile(
    r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}-\d{2}-\d{2})\b"
)
# UPI/Ref ID
_RE_REF = re.compile(r"\b(?:ref|txn|utr|trans)[:\s#]*([A-Z0-9]{6,20})\b", re.IGNORECASE)
# Account last 4 digits
_RE_ACCOUNT = re.compile(r"\b(?:a/c|account|ac|card)\s*[xX*]{0,6}(\d{4})\b", re.IGNORECASE)


@dataclass
class ParsedTransaction:
    amount: Optional[float]
    merchant: Optional[str]
    category: str
    type: str               # "debit" | "credit" | "unknown"
    confidence: float
    date: Optional[str]
    ref_id: Optional[str]
    account_last4: Optional[str]
    raw_text: str
    parse_method: str       # "regex" | "llm_fallback" | "rule_fallback"
    dedup_hash: str = field(init=False)

    def __post_init__(self):
        self.dedup_hash = _compute_dedup_hash(
            self.amount, self.merchant, self.date, self.raw_text
        )

    def to_dict(self) -> dict:
        return {
            "amount": self.amount,
            "merchant": self.merchant,
            "category": self.category,
            "type": self.type,
            "confidence": self.confidence,
            "date": self.date,
            "ref_id": self.ref_id,
            "account_last4": self.account_last4,
            "parse_method": self.parse_method,
            "dedup_hash": self.dedup_hash,
        }


def _compute_dedup_hash(
    amount: Optional[float],
    merchant: Optional[str],
    date: Optional[str],
    raw: str,
) -> str:
    """Stable hash for deduplication — same transaction = same hash."""
    key = f"{amount}|{merchant}|{date}|{raw[:40]}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _normalise_merchant(raw: str) -> tuple[str, str, float]:
    """
    Normalise raw merchant string.
    Returns (normalised_name, category, confidence_boost).
    """
    cleaned = raw.strip().lower()
    for key, normalised in MERCHANT_NORMALISATION.items():
        if key in cleaned:
            category = CATEGORY_MAP.get(normalised, "Miscellaneous")
            return normalised, category, 0.15   # confidence boost for known merchant
    # Unknown merchant — title case it
    title_cased = raw.strip().title()
    return title_cased, "Miscellaneous", 0.0


def _compute_confidence(
    amount: Optional[float],
    merchant: Optional[str],
    txn_type: str,
    is_known_merchant: bool,
) -> float:
    """
    Heuristic confidence score [0.0 – 1.0].
    """
    score = 0.5
    if amount is not None:
        score += 0.20
    if merchant is not None:
        score += 0.10
    if txn_type != "unknown":
        score += 0.10
    if is_known_merchant:
        score += 0.15
    return round(min(score, 0.99), 2)


def parse_sms(text: str) -> ParsedTransaction:
    """
    Parse a financial SMS string using regex.
    Returns a ParsedTransaction with all extracted fields.
    """
    if not text or not text.strip():
        raise ValueError("[SMS_PARSER] Empty input text.")

    text = text.strip()
    logger.debug(f"[SMS_PARSER] Input: {repr(text[:100])}")

    # ── Amount ────────────────────────────────────────────────────────
    amount: Optional[float] = None
    amount_match = _RE_AMOUNT.search(text)
    if amount_match:
        raw_amt = amount_match.group(1).replace(",", "")
        try:
            amount = float(raw_amt)
        except ValueError:
            logger.warning(f"[SMS_PARSER] Could not parse amount: {amount_match.group(1)}")

    # ── Transaction type ──────────────────────────────────────────────
    txn_type = "unknown"
    if _RE_DEBIT.search(text):
        txn_type = "debit"
    elif _RE_CREDIT.search(text):
        txn_type = "credit"

    # ── Merchant ──────────────────────────────────────────────────────
    merchant_raw: Optional[str] = None
    merchant_match = _RE_MERCHANT_AT.search(text)
    if merchant_match:
        merchant_raw = merchant_match.group(1).strip()

    # ── Normalise merchant ────────────────────────────────────────────
    is_known = False
    category = "Miscellaneous"
    merchant: Optional[str] = None

    if merchant_raw:
        merchant, category, boost = _normalise_merchant(merchant_raw)
        is_known = boost > 0
    else:
        boost = 0.0

    # ── Date ──────────────────────────────────────────────────────────
    date: Optional[str] = None
    date_match = _RE_DATE.search(text)
    if date_match:
        date = date_match.group(1)

    # ── Ref ID ────────────────────────────────────────────────────────
    ref_id: Optional[str] = None
    ref_match = _RE_REF.search(text)
    if ref_match:
        ref_id = ref_match.group(1)

    # ── Account last 4 ───────────────────────────────────────────────
    account_last4: Optional[str] = None
    acc_match = _RE_ACCOUNT.search(text)
    if acc_match:
        account_last4 = acc_match.group(1)

    # ── Confidence ────────────────────────────────────────────────────
    confidence = _compute_confidence(amount, merchant, txn_type, is_known)

    txn = ParsedTransaction(
        amount=amount,
        merchant=merchant,
        category=category,
        type=txn_type,
        confidence=confidence,
        date=date,
        ref_id=ref_id,
        account_last4=account_last4,
        raw_text=text,
        parse_method="regex",
    )

    logger.info(
        f"[SMS_PARSER] ✅ amount={txn.amount} merchant={txn.merchant} "
        f"type={txn.type} confidence={txn.confidence}"
    )
    return txn


# ── Deduplication store (in-memory, session-scoped) ───────────────────────────
_seen_hashes: set[str] = set()


def deduplicate(txn: ParsedTransaction) -> bool:
    """
    Returns True if this transaction is a duplicate (already seen).
    Side-effect: registers the hash if not duplicate.
    """
    if txn.dedup_hash in _seen_hashes:
        logger.warning(f"[SMS_PARSER] Duplicate detected: hash={txn.dedup_hash}")
        return True
    _seen_hashes.add(txn.dedup_hash)
    return False


def reset_dedup_store() -> None:
    """Clear dedup store — call between batch runs."""
    _seen_hashes.clear()
