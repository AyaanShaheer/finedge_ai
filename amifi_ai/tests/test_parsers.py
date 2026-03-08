# amifi_ai/tests/test_parsers.py
"""Tests for SMS and CSV parsers."""

import pytest

from amifi_ai.parsers.csv_parser import parse_csv_bytes, parse_csv_file
from amifi_ai.parsers.sms_parser import deduplicate, parse_sms, reset_dedup_store

# ── SMS Parser ────────────────────────────────────────────────────────────────

def test_sms_debit_amazon():
    txn = parse_sms("HDFC Bank: Rs. 2500 debited at AMAZON on 05-03-24")
    assert txn.amount == 2500.0
    assert txn.merchant == "Amazon"
    assert txn.type == "debit"
    assert txn.category == "Shopping"
    assert txn.confidence >= 0.9


def test_sms_credit_salary():
    txn = parse_sms("Your a/c XX4821 credited with Rs.15000 on 04-03-24. UPI Ref: UTR992341XYZ")
    assert txn.amount == 15000.0
    assert txn.type == "credit"
    assert txn.account_last4 == "4821"
    assert txn.ref_id == "UTR992341XYZ"


def test_sms_swiggy_with_ref():
    txn = parse_sms("Paytm: Rs.350 paid to SWIGGY on 03-03-24. Txn ID: PAY77231")
    assert txn.merchant == "Swiggy"
    assert txn.category == "Food & Dining"
    assert txn.amount == 350.0


def test_sms_missing_amount():
    txn = parse_sms("Alert: Large transaction detected. Amount not specified.")
    assert txn.amount is None
    assert txn.confidence == 0.5


def test_sms_empty_raises():
    with pytest.raises(ValueError):
        parse_sms("   ")


def test_sms_deduplication():
    reset_dedup_store()
    text = "HDFC Bank: Rs. 2500 debited at AMAZON on 05-03-24"
    t1 = parse_sms(text)
    t2 = parse_sms(text)
    assert not deduplicate(t1)   # first time → not duplicate
    assert deduplicate(t2)       # second time → duplicate


def test_sms_date_extracted():
    txn = parse_sms("Kotak: INR 89.00 spent at STARBUCKS. Ref: KTK2291ABC on 06-03-24")
    assert txn.date == "06-03-24"


# ── CSV Parser ────────────────────────────────────────────────────────────────

def test_csv_file_parses():
    rows = parse_csv_file("tests/fixtures/sample_bank.csv")
    assert len(rows) == 7  # 8 rows - 1 duplicate


def test_csv_dedup_removed():
    rows = parse_csv_file("tests/fixtures/sample_bank.csv")
    hashes = [r["dedup_hash"] for r in rows]
    assert len(hashes) == len(set(hashes))  # all unique


def test_csv_amounts_positive():
    rows = parse_csv_file("tests/fixtures/sample_bank.csv")
    for row in rows:
        if row["amount"] is not None:
            assert row["amount"] >= 0


def test_csv_types_valid():
    rows = parse_csv_file("tests/fixtures/sample_bank.csv")
    for row in rows:
        assert row["type"] in ("debit", "credit", "unknown")


def test_csv_dates_iso_format():
    rows = parse_csv_file("tests/fixtures/sample_bank.csv")
    for row in rows:
        if row["date"]:
            assert len(row["date"]) == 10  # YYYY-MM-DD


def test_csv_bytes_interface():
    with open("tests/fixtures/sample_bank.csv", "rb") as f:
        content = f.read()
    rows = parse_csv_bytes(content, "test.csv")
    assert len(rows) == 7


def test_csv_missing_file_raises():
    with pytest.raises(ValueError, match="Could not parse"):
        parse_csv_file("tests/fixtures/nonexistent.csv")


def test_csv_stress_large_input():
    """500 rows with duplicates should parse without crashing."""
    header = "Date,Description,Debit,Credit,Balance,Ref No\n"
    rows = ""
    for i in range(500):
        rows += f"05-03-2024,VENDOR {i % 50},{100 + i},,{10000-i},REF{i:05d}\n"
    content = (header + rows).encode("utf-8")
    result = parse_csv_bytes(content, "stress.csv")
    assert len(result) > 0
    assert len(result) <= 500
