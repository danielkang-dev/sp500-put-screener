#!/usr/bin/env python3
"""
fetch_chain.py — fetch a raw options chain from Yahoo Finance (via yfinance)
for a given ticker and save the response exactly as returned.

This does NOT use yfinance's `Ticker.option_chain()`, because that method
parses the response into pandas DataFrames (filtering/reshaping fields
along the way). Instead it goes through yfinance's own authenticated
session (`Ticker._data`, which handles Yahoo's cookie/crumb requirement)
and dumps the raw JSON payload from the v7/finance/options endpoint
untouched.

Expiration selection:
    --date YYYY-MM-DD   Fetch that specific expiration directly (one request).
    (omitted)            Two requests: first with no date param (to read the
                          `expirationDates` list Yahoo returns), then a second
                          request pinned to whichever expiration is closest to
                          14 days out. The SECOND response is what gets saved.

Usage:
    python fetch_chain.py TICKER [--date YYYY-MM-DD] [--out DIR]

Examples:
    python fetch_chain.py AAPL                  # auto-picks ~14-day expiration
    python fetch_chain.py MSTR --date 2026-09-18
    python fetch_chain.py NVR --out ../fixtures

Output:
    Writes <OUT>/<TICKER>-chain.json (default OUT: ../fixtures, i.e. the
    fixtures/ directory next to this script's chain-service/ parent).
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

try:
    from yfinance.const import _BASE_URL_
except ImportError:
    # Fallback for yfinance versions that don't expose this constant.
    _BASE_URL_ = "https://query2.finance.yahoo.com"


def _closest_expiration(expiration_dates, target_days: int = 14) -> int:
    """Return the expiration epoch (seconds) closest to `target_days` from now."""
    now = time.time()
    target = now + target_days * 86400
    return min(expiration_dates, key=lambda epoch: abs(epoch - target))


def fetch_raw_chain(ticker: str, date: Optional[str] = None) -> dict:
    """Return the raw JSON payload from Yahoo Finance's options endpoint.

    No filtering, reshaping, or field selection happens here — this is
    exactly what Yahoo Finance's API returns for the request.

    If `date` is given, fetches that expiration directly (single request).
    Otherwise, makes an initial request with no date param to read the
    `expirationDates` list, picks the expiration closest to 14 days out,
    then makes a second request for that specific expiration. The second
    response is what's returned/saved.
    """
    # yfinance/Yahoo use a hyphen for share-class tickers (BRK-B), while
    # SSGA's holdings file and everything else in this repo uses a period
    # (BRK.B) — normalize for the lookup only, never for anything returned
    # or saved.
    yf_ticker = ticker.replace(".", "-")

    t = yf.Ticker(yf_ticker)

    if date:
        epoch = int(pd.Timestamp(date).timestamp())
        url = f"{_BASE_URL_}/v7/finance/options/{yf_ticker}?date={epoch}"
        response = t._data.get(url=url)
        response.raise_for_status()
        return response.json()

    # First request: no date param, just to discover expirationDates.
    first_url = f"{_BASE_URL_}/v7/finance/options/{yf_ticker}"
    first_response = t._data.get(url=first_url)
    first_response.raise_for_status()
    first_raw = first_response.json()

    results = first_raw.get("optionChain", {}).get("result", [])
    if not results:
        raise ValueError(f"No option chain data returned for {ticker!r}")

    expiration_dates = results[0].get("expirationDates", [])
    if not expiration_dates:
        raise ValueError(f"No expirationDates returned for {ticker!r}")

    target_epoch = _closest_expiration(expiration_dates, target_days=14)

    # Second request: pinned to the expiration closest to 14 days out.
    second_url = f"{_BASE_URL_}/v7/finance/options/{yf_ticker}?date={target_epoch}"
    second_response = t._data.get(url=second_url)
    second_response.raise_for_status()
    return second_response.json()


def main():
    parser = argparse.ArgumentParser(
        description="Fetch a raw options chain from Yahoo Finance via yfinance and save it unmodified as JSON."
    )
    parser.add_argument("ticker", help="Ticker symbol, e.g. AAPL")
    parser.add_argument(
        "--date",
        default=None,
        help=(
            "Specific expiration date (YYYY-MM-DD). If omitted, the script "
            "auto-selects whichever available expiration is closest to 14 "
            "days out."
        ),
    )
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent.parent / "fixtures"),
        help="Output directory for the JSON fixture (default: fixtures/ next to chain-service/)",
    )
    args = parser.parse_args()

    ticker = args.ticker.upper()
    raw = fetch_raw_chain(ticker, args.date)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ticker}-chain.json"

    with open(out_path, "w") as f:
        json.dump(raw, f, indent=2)

    print(f"Saved {ticker} options chain to {out_path}")


if __name__ == "__main__":
    main()
