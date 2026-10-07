#!/usr/bin/env python3
"""
run_screen.py — loop over every SPY constituent, fetch its ~14-DTE options
chain via fetch_chain.py, and run it through qualifying_puts() to find
cash-secured put candidates across the whole index.

This is the orchestration layer fetch_chain.py and put_filters.py were
each missing on their own: fetch_chain.py handles one ticker, and
qualifying_puts() filters one already-fetched chain. This wires the two
together across all ~503 SPY names and, matching the fail-loud style used
elsewhere in this repo, makes sure ONE ticker's failure (an NVR-style
empty chain, a request that errors, a response missing a spot price, ...)
gets logged and skipped rather than taking down the whole run.

NOT included here (same scope note as put_filters.py's docstring):
earnings-before-expiry exclusion, dedupe-to-best-contract-per-underlying,
and the top-20-by-delta ranking CLAUDE.md's full pipeline calls for. This
script collects every qualifying contract from every ticker, unranked and
undeduped — say the word if you want those added as a final pass over the
combined results.

Usage:
    python run_screen.py [--constituents PATH] [--out DIR]
                          [--sleep SECONDS] [--limit N] [--save-raw]

Output (written to --out, default ../output/), checkpointed after every
ticker so a crash or interrupt partway through a 500+ ticker run doesn't
lose everything found so far:
    screen_results.json — every qualifying put found, across all tickers
    screen_errors.json  — every skipped ticker, with the error that skipped it
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from fetch_chain import fetch_raw_chain
from io_utils import atomic_write_json, git_commit_short
from manifest import new_manifest, write_manifest
from put_filters import qualifying_puts

DEFAULT_CONSTITUENTS = Path(__file__).resolve().parent.parent / "fixtures" / "spy_constituents.json"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent / "output"
DEFAULT_SLEEP = 1.0


def load_constituents(path: Path) -> List[Dict[str, str]]:
    with open(path) as f:
        constituents = json.load(f)
    if not constituents:
        raise ValueError(f"No constituents found in {path}")
    return constituents


def screen_ticker(ticker: str, save_raw_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Fetch one ticker's chain and return its qualifying puts.

    Raises on any failure (network error, no options chain, missing spot
    price, ...) — the caller is responsible for catching, logging, and
    moving on; this function itself doesn't swallow anything.
    """
    chain = fetch_raw_chain(ticker)

    if save_raw_dir is not None:
        save_raw_dir.mkdir(parents=True, exist_ok=True)
        with open(save_raw_dir / f"{ticker}-chain.json", "w") as f:
            json.dump(chain, f, indent=2)

    results = chain.get("optionChain", {}).get("result", [])
    if not results:
        raise ValueError(f"No option chain data returned for {ticker!r}")

    spot = results[0].get("quote", {}).get("regularMarketPrice")
    if spot is None:
        raise ValueError(f"No regularMarketPrice in quote for {ticker!r}")

    hits = qualifying_puts(chain, spot)

    # Yahoo echoes back the hyphenated symbol it was queried with (BRK-B),
    # so stamp the original dotted ticker back on. Downstream consumers key
    # off this: rank_shortlist groups by it and queries Finnhub with it, and
    # Finnhub uses periods (BRK.B), not hyphens.
    for hit in hits:
        hit["underlyingSymbol"] = ticker

    return hits


def run_screen(
    constituents_path: Union[str, Path],
    out_dir: Union[str, Path],
    sleep: float = DEFAULT_SLEEP,
    limit: Optional[int] = None,
    save_raw_dir: Optional[Path] = None,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Any]:
    """Screen every constituent in `constituents_path`, checkpointing
    results/errors/manifest to `out_dir` after every ticker, and return the
    final run manifest.

    `progress_cb(done, total, ticker)` is called once per ticker, in
    addition to (not instead of) this function's own stdout/stderr
    printing — it exists for a future caller (the FastAPI sidecar) that
    needs structured progress rather than parsed console output.
    """
    constituents = load_constituents(Path(constituents_path))
    if limit:
        constituents = constituents[:limit]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_qualifying: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []

    total = len(constituents)
    start_time = time.time()
    run_manifest = new_manifest(
        run_id=datetime.now(timezone.utc).isoformat(),
        git_commit=git_commit_short(),
        tickers_total=total,
    )

    for i, entry in enumerate(constituents, start=1):
        ticker = entry.get("Ticker")
        if not ticker:
            print(f"[{i}/{total}] <missing Ticker>: SKIPPED — entry has no Ticker field", file=sys.stderr)
            errors.append({
                "ticker": entry.get("Name", "<unknown>"),
                "error_type": "ValueError",
                "error": "missing Ticker field in constituents entry",
            })
            progress_ticker = "<missing Ticker>"
        else:
            try:
                hits = screen_ticker(ticker, save_raw_dir=save_raw_dir)
            except Exception as e:
                print(f"[{i}/{total}] {ticker}: SKIPPED — {type(e).__name__}: {e}", file=sys.stderr)
                errors.append({"ticker": ticker, "error_type": type(e).__name__, "error": str(e)})
            else:
                print(f"[{i}/{total}] {ticker}: {len(hits)} qualifying put(s)")
                all_qualifying.extend(hits)
            progress_ticker = ticker

        # Checkpoint after every ticker — success OR failure — so a crash
        # or interrupt partway through a 500+ ticker run doesn't lose
        # progress either way. (Earlier draft only wrote this inside the
        # success path, so errors from a ticker with no successful ticker
        # after it never made it to disk — caught by testing before this
        # shipped, not something to reintroduce.) Writes are atomic
        # (write-tmp-then-rename) so a kill mid-checkpoint leaves the
        # previous good file in place rather than a truncated one.
        atomic_write_json(out_dir / "screen_results.json", all_qualifying)
        atomic_write_json(out_dir / "screen_errors.json", errors)

        run_manifest["tickers_errored"] = len(errors)
        run_manifest["tickers_screened"] = i - len(errors)
        run_manifest["qualifying_contracts"] = len(all_qualifying)
        run_manifest["duration_seconds"] = round(time.time() - start_time, 2)
        write_manifest(run_manifest, out_dir / "run_manifest.json")

        if progress_cb is not None:
            progress_cb(i, total, progress_ticker)

        if sleep and i < total:
            time.sleep(sleep)

    print()
    print(f"Done: {total - len(errors)}/{total} tickers screened successfully, {len(errors)} skipped.")
    print(f"{len(all_qualifying)} qualifying put(s) found across all tickers.")
    print(f"Results:  {out_dir / 'screen_results.json'}")
    print(f"Errors:   {out_dir / 'screen_errors.json'}")
    print(f"Manifest: {out_dir / 'run_manifest.json'}")

    return run_manifest


def main():
    parser = argparse.ArgumentParser(
        description="Screen every SPY constituent for qualifying cash-secured puts."
    )
    parser.add_argument(
        "--constituents", default=str(DEFAULT_CONSTITUENTS),
        help="Path to spy_constituents.json (default: fixtures/spy_constituents.json)",
    )
    parser.add_argument(
        "--out", default=str(DEFAULT_OUT_DIR),
        help="Output directory for results/errors JSON (default: ../output)",
    )
    parser.add_argument(
        "--sleep", type=float, default=DEFAULT_SLEEP,
        help=f"Seconds to sleep between tickers, for rate-limit headroom (default: {DEFAULT_SLEEP})",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only screen the first N constituents (for a quick dry run before committing to all ~503)",
    )
    parser.add_argument(
        "--save-raw", action="store_true",
        help="Also save each ticker's raw chain JSON to fixtures/ (off by default, since a full run would write 500+ files there)",
    )
    args = parser.parse_args()

    save_raw_dir = (Path(__file__).resolve().parent.parent / "fixtures") if args.save_raw else None

    run_screen(
        constituents_path=args.constituents,
        out_dir=args.out,
        sleep=args.sleep,
        limit=args.limit,
        save_raw_dir=save_raw_dir,
    )


if __name__ == "__main__":
    main()
