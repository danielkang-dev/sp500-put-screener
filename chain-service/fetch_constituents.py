#!/usr/bin/env python3
"""
fetch_constituents.py — download SPY's daily holdings file from SSGA,
extract ticker + name for every equity holding, and save the list as JSON.

Same fail-loud style as fetch_chain.py: no broad try/except swallowing
errors, HTTP failures surface via raise_for_status(), and unexpected file
structure raises a descriptive ValueError instead of silently producing a
wrong or partial result.

The header row is located dynamically (by scanning for a row containing
the literal cell "Ticker") rather than a hardcoded skiprows count, because
SSGA has changed the number of title/metadata rows at the top of this file
across revisions before — a hardcoded offset would silently break instead
of failing loudly the next time that happens.

NOTE: ssga.com is not reachable from this session's cloud container or
device sandbox (both are network-restricted), so this script has NOT been
run end-to-end against the live file. The header-detection and exclusion
logic below is written defensively for that reason — run it once locally
and sanity-check the output before relying on it.

Usage:
    python fetch_constituents.py [--out PATH] [--keep-xlsx]

Output:
    Writes <OUT> (default: ../fixtures/spy_constituents.json), a JSON list
    of {"Ticker": ..., "Name": ...} objects, one per equity holding.
    Two kinds of non-equity rows are dropped:
      - the cash/non-equity line SSGA includes for SPY's residual cash
        position
      - any row that isn't a plausible equity holding: a Ticker
        containing a digit, or a Name starting with "CONTRA" (fund-
        accounting language for a non-tradeable settlement line)
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict

import pandas as pd
import requests

SPY_HOLDINGS_URL = (
    "https://www.ssga.com/us/en/intermediary/library-content/products/"
    "fund-data/etfs/us/holdings-daily-us-en-spy.xlsx"
)


def download_holdings_xlsx(dest: Path) -> Path:
    """Download the raw SPY holdings .xlsx file to `dest`. Raises on any
    HTTP error via raise_for_status() — same style as fetch_chain.py.
    """
    headers = {"User-Agent": "Mozilla/5.0 (compatible; put-screener/1.0)"}
    response = requests.get(SPY_HOLDINGS_URL, headers=headers, timeout=30)
    response.raise_for_status()
    dest.write_bytes(response.content)
    return dest


def _find_header_row(xlsx_path: Path) -> int:
    """Return the 0-indexed row number of the real column-header row.

    SSGA's file has a few title/metadata rows before the actual table
    (fund name, as-of date, a blank row, ...). Rather than hardcoding how
    many rows to skip, scan for the row whose cells include "Ticker".
    """
    raw_df = pd.read_excel(xlsx_path, header=None, engine="openpyxl")
    for i, row in raw_df.iterrows():
        values = [str(v).strip().lower() for v in row.tolist()]
        if "ticker" in values:
            return i
    raise ValueError(
        "Could not find a header row containing 'Ticker' in the SPY "
        "holdings file — SSGA may have changed the file layout."
    )


def extract_constituents(xlsx_path: Path) -> List[Dict[str, str]]:
    """Parse Ticker + Name out of the downloaded holdings file, dropping
    non-equity rows. Every other row is kept as-is — no other filtering
    happens here.

    Two exclusion rules are applied, independently:

      1. Cash / non-equity settlement row — SSGA represents SPY's
         residual cash position as a pseudo-holding, typically ticker
         "-" and a Name like "US DOLLAR" or containing "CASH".

      2. Not a plausible equity holding:
           - Ticker contains a digit. Real US equity tickers are letters
             plus an optional "." for share class (e.g. "BRK.B") — never
             numeric.
           - Name starts with "CONTRA". That's fund-accounting language
             for a non-tradeable settlement line, not an actual holding.
    """
    header_row = _find_header_row(xlsx_path)

    df = pd.read_excel(xlsx_path, header=header_row, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]

    if "Ticker" not in df.columns or "Name" not in df.columns:
        raise ValueError(
            f"Expected 'Ticker' and 'Name' columns in the holdings file, "
            f"got: {list(df.columns)!r}"
        )

    df = df[["Ticker", "Name"]].dropna(subset=["Ticker", "Name"])
    df["Ticker"] = df["Ticker"].astype(str).str.strip()
    df["Name"] = df["Name"].astype(str).str.strip()

    name_upper = df["Name"].str.upper()

    # Rule 1: cash / non-equity settlement row.
    is_cash = (
        df["Ticker"].isin(["-", ""])
        | name_upper.str.contains("CASH", na=False)
        | (name_upper == "US DOLLAR")
    )

    # Rule 2: not a plausible equity ticker, or a CONTRA settlement line.
    has_digit_ticker = df["Ticker"].str.contains(r"\d", na=False, regex=True)
    is_contra = name_upper.str.startswith("CONTRA")
    is_non_equity = has_digit_ticker | is_contra

    dropped_cash = int(is_cash.sum())
    # Report rule 2's drop count excluding rows rule 1 already accounts
    # for, so a row matching both isn't double-counted across the two
    # warnings below.
    dropped_non_equity = int((is_non_equity & ~is_cash).sum())

    df = df[~(is_cash | is_non_equity)]

    if dropped_cash == 0:
        print(
            "Warning: no cash/non-equity row was dropped. Verify the file "
            "still contains one and that this script's detection of it "
            "(ticker '-' or name containing 'CASH'/'US DOLLAR') still matches."
        )
    elif dropped_cash > 1:
        print(
            f"Warning: dropped {dropped_cash} rows as cash/non-equity, "
            f"expected exactly 1 — double check the extra drops are correct."
        )

    if dropped_non_equity > 0:
        print(
            f"Dropped {dropped_non_equity} additional row(s) as non-equity: "
            f"digit-containing ticker or a CONTRA settlement line."
        )

    return df.to_dict(orient="records")


def main():
    parser = argparse.ArgumentParser(
        description="Download SPY's holdings file from SSGA and save ticker+name constituents as JSON."
    )
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent.parent / "fixtures" / "spy_constituents.json"),
        help="Output path for the JSON constituents list (default: fixtures/spy_constituents.json)",
    )
    parser.add_argument(
        "--keep-xlsx",
        action="store_true",
        help="Keep the downloaded raw .xlsx file next to the output JSON instead of deleting it after parsing.",
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    xlsx_path = out_path.parent / "spy_holdings_raw.xlsx"
    download_holdings_xlsx(xlsx_path)

    constituents = extract_constituents(xlsx_path)

    if not args.keep_xlsx:
        xlsx_path.unlink(missing_ok=True)

    with open(out_path, "w") as f:
        json.dump(constituents, f, indent=2)

    print(f"Saved {len(constituents)} SPY constituents to {out_path}")


if __name__ == "__main__":
    main()
