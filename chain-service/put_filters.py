"""
put_filters.py — filter a raw options chain JSON (as saved by
chain-service/fetch_chain.py) down to qualifying cash-secured put
candidates, per the spec in CLAUDE.md.

Filters are applied IN ORDER, each operating on the survivors of the
previous one:

    1. bid floor            — discard bid < 0.05 before computing anything
    2. OTM                  — (spot - strike) / spot >= 0.08; ALL such
                               strikes are kept, not just the nearest.
                               The distance is computed ONCE, here, and
                               carried onto the surviving contract as
                               `otm_pct` — nothing downstream recomputes
                               it, so a displayed %OTM can't disagree
                               with the filter that passed the row.
    3. strike range          — 10 <= strike <= 500
    4. return on capital     — bid / (strike - bid) >= 0.01   (bid, never mid)
    5. delta                 — |delta| <= 0.20, delta computed via
                               Black-Scholes (Yahoo's raw response gives
                               impliedVolatility per contract but no
                               greeks)
    6. size-aware liquidity  — contracts = floor(capital_per_position /
                               (strike * 100));
                               open_interest >= contracts * 10
                               volume        >= contracts * 2

Scope note: CLAUDE.md's full pipeline also calls for excluding tickers
with earnings before expiry, deduping to one contract per underlying,
and returning only the top 20 by |delta|. Those are cross-ticker /
cross-run concerns (earnings needs a decision on which timestamp field
and how "before expiry" is defined; dedupe and top-N only make sense
across multiple underlyings) that don't fit a single-chain-in,
single-spot-in function signature, so they're intentionally left out
of this function. Flagging so it's a visible choice, not a silent gap.
"""

import math
from typing import List, Dict, Any, Optional

CAPITAL_PER_POSITION = 50_000
OTM_THRESHOLD = 0.08          # strike must be >= 8% below spot
RETURN_ON_CAPITAL_MIN = 0.01
DELTA_MAX_ABS = 0.20
BID_FLOOR = 0.05
STRIKE_MIN = 10
STRIKE_MAX = 500
OI_MULTIPLE = 10
VOLUME_MULTIPLE = 2

SECONDS_PER_YEAR = 365 * 86400


def _norm_cdf(x: float) -> float:
    """Standard normal CDF, via erf (no scipy dependency)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def put_delta_bs(spot: float, strike: float, t_years: float, sigma: float, r: float = 0.0) -> Optional[float]:
    """Black-Scholes delta for a European put. Returns None if inputs are
    degenerate (t<=0, sigma<=0, spot/strike<=0) and a delta can't be
    meaningfully computed.
    """
    if t_years <= 0 or sigma <= 0 or spot <= 0 or strike <= 0:
        return None
    d1 = (math.log(spot / strike) + (r + 0.5 * sigma ** 2) * t_years) / (sigma * math.sqrt(t_years))
    return _norm_cdf(d1) - 1.0


def put_theta_bs(spot: float, strike: float, t_years: float, sigma: float, r: float = 0.0) -> Optional[float]:
    """Black-Scholes theta for a European put, per share per CALENDAR day
    (annual theta / 365). Negative for a long put, as brokers display it.
    Returns None on the same degenerate inputs as put_delta_bs.

    Same inputs, same `r`, same `t_years` as put_delta_bs, so the two greeks
    describe the same contract. With r=0 the carry term vanishes and this
    is just the time-value decay term.
    """
    if t_years <= 0 or sigma <= 0 or spot <= 0 or strike <= 0:
        return None
    sqrt_t = math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (r + 0.5 * sigma ** 2) * t_years) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    pdf_d1 = math.exp(-0.5 * d1 ** 2) / math.sqrt(2.0 * math.pi)
    annual = (-spot * pdf_d1 * sigma / (2.0 * sqrt_t)
              + r * strike * math.exp(-r * t_years) * _norm_cdf(-d2))
    return annual / 365.0


def qualifying_puts(
    chain_json: Dict[str, Any],
    spot: float,
    capital_per_position: float = CAPITAL_PER_POSITION,
    risk_free_rate: float = 0.0,
) -> List[Dict[str, Any]]:
    """Return the list of qualifying cash-secured put candidates from a raw
    Yahoo Finance options-chain response (as saved by fetch_chain.py),
    applying every CLAUDE.md filter in order.

    `chain_json` is the raw, unmodified dict Yahoo returns (i.e. what's in
    fixtures/*.json) — `{"optionChain": {"result": [...]}}`.
    `spot` is passed in explicitly rather than read from the chain, since
    the caller may want to price against a fresher quote than the one
    embedded in a saved fixture. Whatever is passed is also what gets
    RECORDED on each hit as `spot_price` — the output always reports the
    exact price the OTM filter compared against, never a re-read.

    Time-to-expiration for the Black-Scholes delta is measured from the
    quote timestamp embedded in the chain (`quote.regularMarketTime`), not
    wall-clock time — this is what makes results reproducible when re-run
    against a static fixture later; it also matches "as of when this chain
    was captured", which is the only notion of "now" the data actually
    supports.

    Returns a list of dicts (one per qualifying contract), each augmented
    with the computed fields (`spot_price`, `otm_pct`, `delta`, `theta`,
    `return_on_capital`, `contracts`), sorted by nothing in particular —
    ordering/ranking is out of scope here (see module docstring).
    """
    # The OTM filter below divides by spot, so a zero/negative/None price is
    # an argument error rather than something to quietly screen around.
    # run_screen.py already raises on a missing regularMarketPrice and logs
    # the ticker to screen_errors.json; a nonsensical one lands the same way
    # instead of silently yielding zero hits.
    if spot is None or spot <= 0:
        raise ValueError(f"spot must be a positive price, got {spot!r}")

    results = chain_json.get("optionChain", {}).get("result", [])
    if not results:
        return []

    result = results[0]
    quote = result.get("quote", {})
    as_of = quote.get("regularMarketTime")

    qualifying: List[Dict[str, Any]] = []

    for group in result.get("options", []):
        expiration = group.get("expirationDate")
        puts = group.get("puts", [])

        if as_of is not None and expiration is not None:
            t_years = (expiration - as_of) / SECONDS_PER_YEAR
        else:
            t_years = None

        for put in puts:
            strike = put.get("strike")
            if strike is None:
                continue

            # 1. bid floor — discard before computing anything else
            bid = put.get("bid", 0.0) or 0.0
            if bid < BID_FLOOR:
                continue

            # 2. OTM — strike at or below 8% under spot (keep ALL such strikes).
            # Written as a distance rather than a threshold price so the
            # number carried into the output IS the number that was tested.
            # Algebraically identical to the previous
            # `strike > spot * (1 - OTM_THRESHOLD)` for spot > 0, which the
            # guard at the top of this function now guarantees.
            otm_pct = (spot - strike) / spot
            if otm_pct < OTM_THRESHOLD:
                continue

            # 3. strike range
            if not (STRIKE_MIN <= strike <= STRIKE_MAX):
                continue

            # 4. return on capital — bid / (strike - bid), never mid
            denom = strike - bid
            if denom <= 0:
                continue
            return_on_capital = bid / denom
            if return_on_capital < RETURN_ON_CAPITAL_MIN:
                continue

            # 5. delta via Black-Scholes (Yahoo gives IV, not greeks)
            sigma = put.get("impliedVolatility")
            if t_years is None or sigma is None:
                continue
            delta = put_delta_bs(spot, strike, t_years, sigma, r=risk_free_rate)
            if delta is None or abs(delta) > DELTA_MAX_ABS:
                continue
            # Same guards as delta, so this can't be None once delta passed.
            theta = put_theta_bs(spot, strike, t_years, sigma, r=risk_free_rate)

            # 6. size-aware liquidity check
            contracts = math.floor(capital_per_position / (strike * 100))
            if contracts < 1:
                continue
            open_interest = put.get("openInterest", 0) or 0
            volume = put.get("volume", 0) or 0
            if open_interest < contracts * OI_MULTIPLE:
                continue
            if volume < contracts * VOLUME_MULTIPLE:
                continue

            qualifying.append({
                "underlyingSymbol": result.get("underlyingSymbol"),
                "contractSymbol": put.get("contractSymbol"),
                "expirationDate": expiration,
                "strike": strike,
                # Stored as a fraction (0.083), matching return_on_capital;
                # formatted as a percentage at display time.
                "spot_price": spot,
                "otm_pct": otm_pct,
                "bid": bid,
                "impliedVolatility": sigma,
                "delta": delta,
                # Informational only — never filtered on or ranked by.
                # Per share per calendar day, negative.
                "theta": theta,
                "return_on_capital": return_on_capital,
                "openInterest": open_interest,
                "volume": volume,
                "contracts": contracts,
            })

    return qualifying
