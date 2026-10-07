#!/usr/bin/env python3
"""
run_filters_on_fixtures.py — smoke test: run qualifying_puts() against
every fixture in fixtures/ and print what passes for each.

Usage:
    python run_filters_on_fixtures.py
"""

import glob
import json
import os

from put_filters import qualifying_puts

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")


def main():
    paths = sorted(glob.glob(os.path.join(FIXTURES_DIR, "*.json")))
    if not paths:
        print(f"No fixtures found in {FIXTURES_DIR}")
        return

    for path in paths:
        name = os.path.basename(path)
        with open(path) as f:
            chain_json = json.load(f)

        results = chain_json.get("optionChain", {}).get("result", [])
        spot = results[0]["quote"].get("regularMarketPrice") if results else None

        print(f"=== {name} (spot={spot}) ===")
        if spot is None:
            print("  no quote/spot in fixture, skipping")
            continue

        hits = qualifying_puts(chain_json, spot)
        if not hits:
            print("  (none qualify)")
            continue

        for h in sorted(hits, key=lambda x: abs(x["delta"])):
            print(
                f"  {h['contractSymbol']:<24} strike={h['strike']:<8} "
                f"bid={h['bid']:<6} delta={h['delta']:.4f} "
                f"return={h['return_on_capital']:.4f} "
                f"OI={h['openInterest']:<6} vol={h['volume']:<6} "
                f"contracts={h['contracts']}"
            )
        print(f"  -> {len(hits)} qualifying put(s)")
        print()


if __name__ == "__main__":
    main()
