#!/usr/bin/env python3
"""
diagnose_earnings.py — read-only probe answering one question: how often does
Finnhub actually return an earnings date for the tickers that reach the
ranking stage, and how often does it return nothing?

This exists because rank_shortlist.py's docstring and its code disagree. The
docstring says a ticker with no Finnhub entry is EXCLUDED ("can't confirm it's
safe" defaults to safe); the code keeps it, because the guard reads
`if earnings_date is not None and earnings_date <= expiration_date`.

Closing that gap means excluding on None. Whether that is safe to do depends
on how often None means "Finnhub genuinely has nothing scheduled" versus
"Finnhub has coverage gaps" — if the latter is common, excluding on None would
gut the shortlist. Every S&P 500 name should report within the 120-day
lookahead, so None is *expected* to be rare. This measures whether it is.

Writes nothing into the pipeline's outputs and mutates no state. It only reads
screen_results.json and calls the same endpoint rank_shortlist.py calls.

Usage:
    export FINNHUB_API_KEY=...
    python diagnose_earnings.py [--results PATH] [--out PATH] [--sleep SECONDS]

Output:
    A summary table on stdout, plus <OUT> (default
    ../output/earnings_diagnostic.json) with the per-ticker detail.
"""

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from rank_shortlist import (
    DEFAULT_RESULTS,
    DEFAULT_SLEEP,
    _env_api_key,
    get_next_earnings_date,
)

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "output" / "earnings_diagnostic.json"


def classify(earnings_date, expiration_date):
    """What the current code does vs what the docstring says it should do."""
    if earnings_date is None:
        return "no_entry", "kept", "excluded"
    if earnings_date <= expiration_date:
        return "earnings_before_expiry", "excluded", "excluded"
    return "earnings_after_expiry", "kept", "kept"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--results", default=str(DEFAULT_RESULTS))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP)
    args = parser.parse_args()

    api_key = _env_api_key()

    with open(args.results) as f:
        hits = json.load(f)

    if not hits:
        print(f"No candidates in {args.results} — nothing to diagnose.")
        return

    by_ticker = {}
    for h in hits:
        by_ticker.setdefault(h["underlyingSymbol"], set()).add(h["expirationDate"])

    rows = []
    tickers = sorted(by_ticker)
    for i, ticker in enumerate(tickers, start=1):
        expiry_epoch = min(by_ticker[ticker])
        expiry = datetime.fromtimestamp(expiry_epoch, tz=timezone.utc).date().isoformat()

        try:
            earnings = get_next_earnings_date(ticker, api_key)
        except Exception as e:
            # Deliberately does not echo the exception text: with the key now
            # sent as a header it should be clean, but this script's whole
            # purpose is diagnosing an untrusted path.
            rows.append({
                "ticker": ticker, "expiry": expiry, "earnings": None,
                "outcome": "lookup_failed", "current": "excluded", "documented": "excluded",
                "error_type": type(e).__name__,
            })
            print(f"[{i}/{len(tickers)}] {ticker}: LOOKUP FAILED ({type(e).__name__})", file=sys.stderr)
        else:
            outcome, current, documented = classify(earnings, expiry)
            rows.append({
                "ticker": ticker, "expiry": expiry, "earnings": earnings,
                "outcome": outcome, "current": current, "documented": documented,
            })
            print(f"[{i}/{len(tickers)}] {ticker}: expiry {expiry}, earnings {earnings or '—'} -> {outcome}")

        if args.sleep and i < len(tickers):
            time.sleep(args.sleep)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)

    counts = Counter(r["outcome"] for r in rows)
    kept_now = sum(1 for r in rows if r["current"] == "kept")
    kept_doc = sum(1 for r in rows if r["documented"] == "kept")

    print("\n" + "=" * 58)
    print(f"{len(rows)} ticker(s) probed")
    for outcome, n in counts.most_common():
        print(f"  {outcome:<26} {n}")
    print("-" * 58)
    print(f"  survive under current code    {kept_now}")
    print(f"  survive under documented rule {kept_doc}")
    delta = kept_now - kept_doc
    if delta:
        print(f"\n  => closing the gap would drop {delta} more ticker(s).")
        print("     Those are names kept today without their earnings verified.")
    else:
        print("\n  => no difference: the gap is not currently load-bearing,")
        print("     so implementing the documented rule is a safe no-op today.")
    print("=" * 58)
    print(f"\nDetail: {out_path}")


if __name__ == "__main__":
    main()
