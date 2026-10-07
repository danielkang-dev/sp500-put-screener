#!/usr/bin/env python3
"""
rank_shortlist.py — final pass over output/screen_results.json: apply the
three CLAUDE.md rules not yet wired into put_filters.py/run_screen.py,
plus two corrections layered on top of them:

  1. exclude any ticker with earnings before expiry
  2. drop contracts with implausible IV or thin open interest
  3. annualize return_on_capital by each contract's DTE
  4. dedupe to best contract per underlying
  5. return top 20 sorted by bucketed abs(delta) ascending, with
     annualized return descending as the tiebreak within each bucket

Writes the result to output/screen_ranked.json.

CLAUDE.md states rules 1, 4, and 5 (the last as "top 20 by abs(delta)
ascending") as single lines with no further detail. Steps 2 and 3 aren't
in CLAUDE.md at all, and the tiebreak half of step 5 isn't either. Why
each exists:

  - annualizing (step 3) exists because comparing raw return_on_capital
    across expirations silently favors whichever contracts have the
    longest DTE. A live run put 6 of 20 shortlisted contracts on a 28-DTE
    monthly expiry (tickers with no weekly options) next to 14 on a
    14-DTE expiry, and a 28-day 1% return isn't worth the same as a
    14-day 1% return — it's worth roughly half as much annualized. The
    fix is to annualize, not to filter: a strict DTE window would have
    dropped 6 of 20 real names, several of them perfectly tradeable on
    their own terms. They need a fair comparison, not exclusion.

  - the quality filters (step 2) exist because delta is DERIVED from IV
    here (put_filters.py computes it via Black-Scholes; Yahoo supplies no
    greeks). A stale or wide quote with an absurd IV therefore corrupts
    the RANKING KEY, not merely the displayed number — it manufactures a
    delta that the sort then trusts. On the same live run, 5 of 68
    contracts carried IV above 100%, topped by a custody bank at 175%,
    which is a quote artifact rather than a market view.

  - abs(delta) is bucketed (step 5) rather than compared raw because it
    is a continuous Black-Scholes output: across 68 real contracts there
    were ZERO exact ties at full float precision, so an untruncated
    primary key would make the annualized-return tiebreak unreachable
    and reduce annualization to decoration. Rounding to
    DELTA_BUCKET_PLACES groups contracts into delta bands that are
    genuinely indistinguishable in risk terms (0.12 vs 0.1201), and lets
    annualized return decide the order inside each band. Delta still
    leads; annualization still does real work.

This module never publishes an empty shortlist. If every candidate is
filtered out — or screen_results.json turns out to be empty — rank()
raises EmptyResultError instead of writing `[]`, which run_pipeline
classifies DEGRADED (exit 2). Writing an empty list would exit 0 and have
n8n publish "no trades today" as a real result, when the actual meaning is
"the filters ate everything, go look at why." With the IV/OI filters below
dropping 8 of 38 tickers on an ordinary day, that is a reachable outcome,
not a theoretical one.

Every remaining judgment call is called out below rather than left
implicit:

  - "earnings before expiry": Finnhub's earnings calendar gives a date,
    not a timestamp, and doesn't reliably distinguish before/after market
    hours on that date. Comparing at the DATE level, this script treats
    earnings ON THE SAME DAY as expiry as "before expiry" too (excludes
    it) — the conservative reading, since same-day earnings could land
    before the option expires depending on BMO/AMC timing Finnhub doesn't
    guarantee here. If Finnhub has no earnings-calendar entry for a
    ticker in the lookahead window, OR the API call itself fails, this
    script EXCLUDES that ticker rather than including it — "can't
    confirm it's safe" defaults to the safe outcome, same reasoning as
    the rule's purpose in the first place. Every exclusion is logged to
    screen_ranked_exclusions.json with a `reason` field so you can tell
    "confirmed earnings before expiry" apart from "couldn't check."

  - annualizing return_on_capital: `annualized_return = return_on_capital
    * 365 / dte`. `dte` isn't in screen_results.json — put_filters.py
    computes a time-to-expiry for its own Black-Scholes delta, but never
    writes it or the quote timestamp it used into the hit dict, so it
    doesn't survive past that one function call. This script computes
    `dte` from each hit's `expirationDate` against the RUN MANIFEST's
    `run_id` timestamp — the moment run_screen.py started — rather than
    wall-clock now. Both are approximations of "when was this chain
    captured", but the manifest's is the reproducible one: re-ranking the
    same screen_results.json a day later yields the same DTE and the same
    shortlist, where wall-clock would quietly shorten every contract.
    `dte` is floored at MIN_DTE to keep the formula from dividing by zero
    on an expiration that lands on the run date or (only in theory, given
    how fetch_chain.py picks expirations) has already passed. Raw
    `return_on_capital` is left untouched in the output —
    annualized_return is an added field, not a replacement — so anyone
    reading screen_ranked.json can still see the un-annualized number.

  - spot_price / otm_pct: read from screen_results.json, never derived
    here. Both are captured at SCREEN time by put_filters.py, against the
    exact quote its OTM filter compared each strike to. Rank fetching a
    fresh spot would break reproducibility across an n8n retry the same
    way wall-clock DTE would, and recomputing otm_pct from a stored
    spot_price would put a second copy of the formula in a second file,
    free to drift from the one that actually gated the row. So this
    module only carries them through. Files written before those fields
    existed degrade to null with a note on stderr — a display gap in an
    old file is not a reason to fail a run.

  - the IV ceiling and OI floor: both are contract-level and both run
    BEFORE dedupe, so that a ticker whose best-looking contract is best
    only because its quote is broken falls back to a real contract rather
    than being represented by the artifact. MAX_IMPLIED_VOLATILITY is a
    blunt 1.0 — a put on an S&P 500 constituent implying >100% annualized
    vol is a stale/wide quote, not a forecast — and MIN_OPEN_INTEREST is
    a flat floor applied IN ADDITION TO put_filters.py's size-aware
    `open_interest >= contracts * 10` gate, which scales with position
    size and so lets a high-strike, single-contract name through on very
    little real interest. A contract failing both tests is logged once,
    under the IV reason, since IV is checked first.

  - "best contract per underlying" for dedupe: not defined in CLAUDE.md.
    Since the final ranking key is (bucketed abs(delta) ascending,
    annualized_return descending), this script uses THE SAME key for
    dedupe — see ranking_key(), which both call — so dedupe and ranking
    can never disagree about which contract is better.

API key: reads FINNHUB_API_KEY from the environment, full stop. CLAUDE.md
explicitly bans real API keys in any file in this repo ("No real API keys
in any file"), so there's no --api-key flag and nothing here will ever
accept or persist a literal key. Set it in your shell before running:
    export FINNHUB_API_KEY=...

NOTE: like fetch_chain.py and fetch_constituents.py before it, this has
NOT been run against the live Finnhub API — neither this session's cloud
container nor its device sandbox has network access to finnhub.io. The
earnings-lookup logic is tested against a mocked Finnhub client, and the
dedupe/ranking logic against synthetic screen_results.json data — not the
real endpoint. Run it for real and sanity-check the output before relying
on it, same caveat as every other script that talks to a live API here.

Usage:
    export FINNHUB_API_KEY=...
    python rank_shortlist.py [--results PATH] [--out PATH] [--sleep SECONDS]

Output:
    <OUT> (default ../output/screen_ranked.json) — top 20 ranked candidates
    <OUT's dir>/screen_ranked_exclusions.json — every ticker dropped by the
        earnings-before-expiry rule, with why
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import requests

from io_utils import atomic_write_json
from manifest import (
    DEFAULT_MANIFEST_PATH,
    EmptyResultError,
    check_sanity_gate,
    load_manifest,
    write_manifest,
)

FINNHUB_BASE_URL = "https://finnhub.io/api/v1/calendar/earnings"
DEFAULT_RESULTS = Path(__file__).resolve().parent.parent / "output" / "screen_results.json"
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "output" / "screen_ranked.json"
DEFAULT_SLEEP = 1.0
TOP_N = 20
EARNINGS_LOOKAHEAD_DAYS = 120  # comfortably past any 14-DTE expiration
DAYS_PER_YEAR = 365
MIN_DTE = 1  # floor to keep annualized_return's division from hitting zero

# Delta is a continuous Black-Scholes output — 68 real contracts produced
# zero exact ties — so the primary sort key is rounded into bands. Without
# this the annualized-return tiebreak below could never fire. 2dp groups
# contracts that are indistinguishable in risk terms (0.12 vs 0.1201).
DELTA_BUCKET_PLACES = 2

# A put on an S&P 500 name implying >100% vol is a stale or wide quote, not
# a market view — and since delta is DERIVED from IV here, that bad number
# corrupts the sort key itself, not just the display.
MAX_IMPLIED_VOLATILITY = 1.0

# Flat floor, applied on top of put_filters.py's size-aware
# `open_interest >= contracts * 10`. That gate scales with position size,
# so a high-strike name sized at one contract clears it with only 10 open
# interest — thin enough that the quote behind it means very little.
MIN_OPEN_INTEREST = 50
EARNINGS_RETRY_BACKOFF = 2.0  # seconds, one retry for transient network/SSL errors

# requests.exceptions.SSLError subclasses ConnectionError, so this also
# covers the SSL handshake failures seen in practice — a live Finnhub probe
# hit one, and a plain retry cleared it. HTTPError (4xx/5xx responses) is
# deliberately excluded: a bad key or rate limit won't be fixed by retrying.
_RETRYABLE_ERRORS = (requests.exceptions.ConnectionError, requests.exceptions.Timeout)


def _env_api_key() -> str:
    key = os.environ.get("FINNHUB_API_KEY")
    if not key:
        raise RuntimeError(
            "FINNHUB_API_KEY is not set. Per CLAUDE.md ('No real API keys in "
            "any file'), this script only ever reads it from the environment "
            "— run `export FINNHUB_API_KEY=...` first."
        )
    return key


def get_next_earnings_date(
    symbol: str,
    api_key: str,
    lookahead_days: int = EARNINGS_LOOKAHEAD_DAYS,
    as_of: Optional[datetime] = None,
) -> Optional[str]:
    """Return the next scheduled earnings date (YYYY-MM-DD string) for
    `symbol` within `lookahead_days` of `as_of` (default: now), or None if
    Finnhub has no earnings scheduled for it in that window.

    Raises on HTTP/network failure — the caller decides what "couldn't
    check" means for the exclusion rule (see module docstring). A transient
    connection or SSL error gets one retry after EARNINGS_RETRY_BACKOFF
    seconds before that; an HTTP error response (401, 429, 5xx, ...) is not
    retried, since a bad key or a rate limit won't resolve on its own.
    """
    as_of = as_of or datetime.now(timezone.utc)
    frm = as_of.date().isoformat()
    to = (as_of + timedelta(days=lookahead_days)).date().isoformat()

    # Key goes in a header, never the query string: requests embeds the full
    # URL in its exception messages, and those get written to the exclusions
    # log and sent to alerting.
    for attempt in range(2):
        try:
            response = requests.get(
                FINNHUB_BASE_URL,
                params={"from": frm, "to": to, "symbol": symbol},
                headers={"X-Finnhub-Token": api_key},
                timeout=15,
            )
            response.raise_for_status()
            payload = response.json()
            break
        except _RETRYABLE_ERRORS:
            if attempt == 0:
                time.sleep(EARNINGS_RETRY_BACKOFF)
                continue
            raise

    entries = payload.get("earningsCalendar") or []
    dates = sorted(e["date"] for e in entries if e.get("date"))
    return dates[0] if dates else None


def apply_earnings_exclusion(
    hits: List[Dict[str, Any]],
    api_key: str,
    sleep: float = DEFAULT_SLEEP,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    """Rule 1: exclude any ticker with earnings before expiry.

    Returns (surviving_hits, exclusions). `exclusions` logs every ticker
    dropped and why — distinguishing confirmed earnings-before-expiry
    from a Finnhub lookup that failed (see module docstring for why an
    unconfirmable ticker is excluded, not kept).
    """
    by_ticker: Dict[str, List[Dict[str, Any]]] = {}
    for hit in hits:
        by_ticker.setdefault(hit["underlyingSymbol"], []).append(hit)

    surviving: List[Dict[str, Any]] = []
    exclusions: List[Dict[str, str]] = []

    tickers = sorted(by_ticker)
    for i, ticker in enumerate(tickers, start=1):
        contracts = by_ticker[ticker]
        # All contracts for a ticker share one expiration in this repo's
        # flow (one chain fetched per ticker), but don't assume it.
        expirations = {c["expirationDate"] for c in contracts}

        try:
            earnings_date = get_next_earnings_date(ticker, api_key)
        except Exception as e:
            print(
                f"[{i}/{len(tickers)}] {ticker}: earnings lookup FAILED — "
                f"{type(e).__name__}: {e} — excluding (can't confirm safe)",
                file=sys.stderr,
            )
            exclusions.append({
                "ticker": ticker,
                "reason": "earnings_lookup_failed",
                "detail": f"{type(e).__name__}: {e}",
            })
            if sleep and i < len(tickers):
                time.sleep(sleep)
            continue

        if earnings_date is None:
            # diagnose_earnings.py measured this against two real runs (54
            # tickers): 0 hit this path. Excluding here is therefore a
            # no-op against observed data, and closes the gap where a
            # ticker Finnhub doesn't resolve would otherwise pass unverified.
            exclusions.append({
                "ticker": ticker,
                "reason": "earnings_not_found",
                "detail": "no Finnhub earnings-calendar entry within lookahead window",
            })
            if sleep and i < len(tickers):
                time.sleep(sleep)
            continue

        excluded = False
        for expiration_epoch in expirations:
            expiration_date = datetime.fromtimestamp(expiration_epoch, tz=timezone.utc).date().isoformat()
            if earnings_date <= expiration_date:
                exclusions.append({
                    "ticker": ticker,
                    "reason": "earnings_before_expiry",
                    "detail": f"earnings {earnings_date} <= expiry {expiration_date}",
                })
                excluded = True
                break

        if not excluded:
            surviving.extend(contracts)

        if sleep and i < len(tickers):
            time.sleep(sleep)

    return surviving, exclusions


def apply_quality_filters(
    hits: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    """Rule 2: drop contracts whose quote is implausible enough that the
    numbers derived from it can't be trusted.

    Contract-level, not ticker-level: one bad strike doesn't condemn the
    underlying, it just shouldn't be the strike that represents it. Runs
    before dedupe so a ticker falls back to its next-best real contract.

    Returns (surviving, exclusions), with exclusions shaped like
    apply_earnings_exclusion's but carrying a `contractSymbol` too, since
    these drop individual contracts rather than whole tickers.
    """
    surviving: List[Dict[str, Any]] = []
    exclusions: List[Dict[str, str]] = []

    for hit in hits:
        ticker = hit["underlyingSymbol"]
        contract_symbol = hit.get("contractSymbol")

        iv = hit.get("impliedVolatility")
        if iv is not None and iv > MAX_IMPLIED_VOLATILITY:
            exclusions.append({
                "ticker": ticker,
                "contractSymbol": contract_symbol,
                "reason": "implied_volatility_implausible",
                "detail": f"IV {iv:.3f} > {MAX_IMPLIED_VOLATILITY:.2f}; "
                          f"delta is derived from IV, so this corrupts the ranking key",
            })
            continue

        open_interest = hit.get("openInterest", 0) or 0
        if open_interest < MIN_OPEN_INTEREST:
            exclusions.append({
                "ticker": ticker,
                "contractSymbol": contract_symbol,
                "reason": "open_interest_below_floor",
                "detail": f"open interest {open_interest} < {MIN_OPEN_INTEREST}",
            })
            continue

        surviving.append(hit)

    return surviving, exclusions


def backfill_spot_fields(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Guarantee every hit carries `spot_price` and `otm_pct`, degrading to
    None for screen_results.json files written before put_filters.py
    recorded them.

    This module never fetches a spot price and never recomputes otm_pct —
    see the module docstring for why. A missing key therefore means "this
    file predates the fields", which is a gap in what can be displayed,
    not a reason to raise KeyError partway through a run that otherwise
    produced a perfectly good shortlist.

    Returns new dicts; existing values always win over the None defaults.
    """
    missing = sum(1 for h in hits if "spot_price" not in h or "otm_pct" not in h)
    if missing:
        print(
            f"[rank] {missing} of {len(hits)} candidate(s) carry no "
            f"spot_price/otm_pct — this screen_results.json predates those "
            f"fields. Reporting them as null; rank does not fetch spot "
            f"prices. Re-run the screen to populate them.",
            file=sys.stderr,
        )
    return [{"spot_price": None, "otm_pct": None, **h} for h in hits]


def compute_dte(expiration_epoch: Union[int, float], as_of: Optional[datetime] = None) -> int:
    """Days to expiry for one contract, at the DATE level (matching how
    apply_earnings_exclusion already compares expiry dates). `as_of`
    defaults to wall-clock now; rank() passes the manifest's run_id
    instead, which is the reproducible reference — see module docstring.
    """
    as_of = as_of or datetime.now(timezone.utc)
    expiration_date = datetime.fromtimestamp(expiration_epoch, tz=timezone.utc).date()
    return (expiration_date - as_of.date()).days


def as_of_from_manifest(manifest: Dict[str, Any]) -> Optional[datetime]:
    """The run's start time, parsed out of the manifest's `run_id`, for use
    as the DTE reference point.

    Returns None — meaning "fall back to wall-clock now" — if `run_id` is
    absent or unparseable. A hand-edited or stub manifest shouldn't harden
    into a crash in the ranker: a slightly-off DTE reference degrades the
    shortlist's ordering, while raising here would produce no shortlist at
    all. A naive timestamp is read as UTC, matching every other timestamp
    in this pipeline.
    """
    raw = manifest.get("run_id")
    try:
        parsed = datetime.fromisoformat(raw)
    except (TypeError, ValueError):
        print(
            f"[rank] manifest run_id is missing or unparseable ({raw!r}); "
            f"falling back to wall-clock now for DTE. Ordering may differ "
            f"from a run ranked at screen time.",
            file=sys.stderr,
        )
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def annualize_returns(
    hits: List[Dict[str, Any]],
    as_of: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """Rule 2: add `dte` and `annualized_return` to every hit, so contracts
    on different expirations can be compared on equal footing.

    `annualized_return = return_on_capital * 365 / dte`, with `dte`
    floored at MIN_DTE. Returns new dicts — `return_on_capital` and every
    other original field are left untouched, `dte`/`annualized_return`
    are additions, not replacements. `as_of` is the DTE reference point;
    rank() passes the manifest's run_id, which is what makes a re-rank
    reproducible. See module docstring.
    """
    annotated = []
    for hit in hits:
        dte = max(compute_dte(hit["expirationDate"], as_of), MIN_DTE)
        annualized_return = hit["return_on_capital"] * DAYS_PER_YEAR / dte
        annotated.append({**hit, "dte": dte, "annualized_return": annualized_return})
    return annotated


def ranking_key(hit: Dict[str, Any]) -> Tuple[float, float]:
    """The one definition of "better", ascending: lower is better.

    Primary is abs(delta) rounded into DELTA_BUCKET_PLACES bands, so
    contracts of effectively equal risk compete on the secondary key
    instead of on float noise. Secondary is NEGATED annualized_return, so
    that within a delta band the higher annualized return sorts first
    while the tuple as a whole stays plain-ascending.

    Both dedupe_to_best_per_underlying() and rank_top_n() call this, which
    is what makes "best contract per underlying" and "top 20" structurally
    incapable of disagreeing.
    """
    return (round(abs(hit["delta"]), DELTA_BUCKET_PLACES), -hit["annualized_return"])


def dedupe_to_best_per_underlying(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rule 4: dedupe to best contract per underlying.

    "Best" isn't defined in CLAUDE.md — this uses ranking_key(), the same
    ordering the final shortlist uses. See module docstring.
    """
    best_by_ticker: Dict[str, Dict[str, Any]] = {}
    for hit in hits:
        ticker = hit["underlyingSymbol"]
        current_best = best_by_ticker.get(ticker)
        if current_best is None or ranking_key(hit) < ranking_key(current_best):
            best_by_ticker[ticker] = hit
    return list(best_by_ticker.values())


def rank_top_n(hits: List[Dict[str, Any]], n: int = TOP_N) -> List[Dict[str, Any]]:
    """Rule 5: top N by ranking_key ascending — bucketed abs(delta) first,
    annualized return descending as the tiebreak within each bucket."""
    return sorted(hits, key=ranking_key)[:n]


def rank(
    results_path: Union[str, Path],
    out_path: Union[str, Path],
    manifest_path: Union[str, Path],
    api_key: str,
    sleep: float = DEFAULT_SLEEP,
    as_of: Optional[datetime] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    """Final pass over `results_path`: sanity-gate the manifest, then apply
    earnings exclusion, quality filters, annualization, dedupe, and top-N
    ranking, in that order. Writes `out_path` and
    `<out_path's dir>/screen_ranked_exclusions.json`, updates the manifest
    at `manifest_path` with the ranking-phase counts, and returns (ranked,
    exclusions).

    `as_of` is the reference point annualize_returns() computes each
    contract's DTE against. It defaults to the manifest's `run_id` — the
    reproducible choice, see module docstring — falling back to wall-clock
    now only if that timestamp is missing or unparseable. Pass it
    explicitly to override both; the CLI and run_pipeline.py never do.

    Raises RuntimeError (from check_sanity_gate) if the screen described by
    the manifest is incomplete or produced zero qualifying contracts —
    before this function touches `results_path`, Finnhub, or any output
    file.

    Also raises EmptyResultError — the same type the gate uses, so
    run_pipeline classifies it DEGRADED without needing to know which
    check fired — if ranking ends with nothing to publish, either because
    `results_path` was empty or because the filters removed every
    candidate. In both cases `out_path` is left untouched rather than
    overwritten with `[]`: an empty shortlist written to disk is
    indistinguishable from a real one downstream, and n8n would publish
    it. This is why the function returns a non-empty `ranked` or raises,
    never an empty list.
    """
    # Sanity gate first, before touching screen_results.json: it exists to
    # stop a partial or broken screen from being ranked as if it were
    # healthy, which only works if it runs before anything downstream of
    # that screen does.
    run_manifest = load_manifest(Path(manifest_path))
    check_sanity_gate(run_manifest)

    with open(results_path) as f:
        hits = json.load(f)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    exclusions_path = out_path.parent / "screen_ranked_exclusions.json"

    if not hits:
        # The sanity gate already rules this out when the manifest's
        # qualifying_contracts is 0 for a complete screen — this remains as
        # a direct guard against results_path pointing somewhere stale or
        # hand-edited, independent of the manifest.
        raise EmptyResultError(
            f"Refusing to rank: {results_path} contains no candidates, even "
            f"though the manifest reports "
            f"{run_manifest.get('qualifying_contracts')} qualifying "
            f"contract(s). The results file is stale, truncated, or points "
            f"somewhere unexpected — no shortlist was written."
        )

    n_tickers = len({h["underlyingSymbol"] for h in hits})
    print(f"Loaded {len(hits)} candidate contract(s) across {n_tickers} ticker(s).")

    # Before any filtering, so the note below describes the FILE rather than
    # whichever contracts happened to survive. Everything downstream copies
    # dicts with {**hit, ...}, so both fields ride through untouched.
    hits = backfill_spot_fields(hits)

    after_earnings, exclusions = apply_earnings_exclusion(hits, api_key, sleep=sleep)
    print(
        f"After earnings-before-expiry exclusion: {len(after_earnings)} contract(s) "
        f"survive, {len(exclusions)} ticker(s) excluded."
    )

    after_quality, quality_exclusions = apply_quality_filters(after_earnings)
    exclusions.extend(quality_exclusions)
    print(
        f"After quality filters (IV <= {MAX_IMPLIED_VOLATILITY:.2f}, open interest "
        f">= {MIN_OPEN_INTEREST}): {len(after_quality)} contract(s) survive, "
        f"{len(quality_exclusions)} contract(s) dropped."
    )

    if as_of is None:
        as_of = as_of_from_manifest(run_manifest)
    annualized = annualize_returns(after_quality, as_of=as_of)

    deduped = dedupe_to_best_per_underlying(annualized)
    print(f"After dedupe-to-best-per-underlying: {len(deduped)} contract(s), one per surviving ticker.")

    ranked = rank_top_n(deduped, TOP_N)
    print(
        f"Top {len(ranked)} by abs(delta) ascending (bucketed to "
        f"{DELTA_BUCKET_PLACES}dp), annualized return descending within bucket."
    )

    if not ranked:
        # Everything that entered ranking was filtered back out. The run
        # itself worked — this is DEGRADED, not FAILED — but it produces no
        # shortlist, and returning normally here would exit 0 and have n8n
        # publish the empty list as today's result.
        #
        # The exclusions log IS written before raising, even though the
        # shortlist deliberately is not: it is the only record of WHY every
        # candidate vanished, which is exactly the question this exception
        # sends someone to answer. It is a diagnostic file that nothing
        # downstream publishes, so writing it cannot be mistaken for a
        # result — where an empty screen_ranked.json could be.
        atomic_write_json(exclusions_path, exclusions)
        by_reason: Dict[str, int] = {}
        for e in exclusions:
            by_reason[e["reason"]] = by_reason.get(e["reason"], 0) + 1
        breakdown = ", ".join(f"{reason}: {n}" for reason, n in sorted(by_reason.items())) or "none"
        raise EmptyResultError(
            f"Refusing to publish an empty shortlist: {len(hits)} qualifying "
            f"contract(s) across {n_tickers} ticker(s) entered ranking and "
            f"none survived. Exclusions by reason — {breakdown}. The screen "
            f"itself completed, so this is a filter or data-quality problem, "
            f"not a crash: start with {exclusions_path}. No shortlist was "
            f"written, so any existing {out_path} is from an earlier run."
        )

    atomic_write_json(out_path, ranked)
    atomic_write_json(exclusions_path, exclusions)

    run_manifest["after_earnings_exclusion"] = len(after_earnings)
    run_manifest["after_quality_filters"] = len(after_quality)
    run_manifest["after_dedupe"] = len(deduped)
    run_manifest["final_ranked"] = len(ranked)
    run_manifest["finnhub_calls_failed"] = sum(
        1 for e in exclusions if e["reason"] == "earnings_lookup_failed"
    )
    write_manifest(run_manifest, Path(manifest_path))

    print(f"Ranked shortlist: {out_path}")
    print(f"Earnings exclusions log: {exclusions_path}")
    print(f"Manifest updated: {manifest_path}")

    return ranked, exclusions


def main():
    parser = argparse.ArgumentParser(
        description="Final pass over screen_results.json: earnings exclusion, dedupe, top-20 ranking."
    )
    parser.add_argument("--results", default=str(DEFAULT_RESULTS), help="Path to screen_results.json")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Path to write the ranked shortlist")
    parser.add_argument(
        "--manifest", default=str(DEFAULT_MANIFEST_PATH),
        help="Path to the run_manifest.json written by run_screen.py (default: ../output/run_manifest.json)",
    )
    parser.add_argument(
        "--sleep", type=float, default=DEFAULT_SLEEP,
        help=f"Seconds to sleep between Finnhub calls, for rate-limit headroom (default: {DEFAULT_SLEEP})",
    )
    args = parser.parse_args()

    rank(
        results_path=args.results,
        out_path=args.out,
        manifest_path=args.manifest,
        api_key=_env_api_key(),
        sleep=args.sleep,
    )


if __name__ == "__main__":
    main()
