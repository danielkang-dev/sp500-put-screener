import math

import pytest
from conftest import make_chain, make_put, spot_of

from put_filters import OTM_THRESHOLD, put_delta_bs, put_theta_bs, qualifying_puts


def only_hit(puts, **chain_kwargs):
    chain = make_chain(puts, **chain_kwargs)
    return qualifying_puts(chain, spot_of(chain))


class TestBaseline:
    def test_reference_put_qualifies(self):
        """Guards every other test here: they work by breaking one field of this put."""
        hits = only_hit([make_put()])
        assert len(hits) == 1
        assert hits[0]["strike"] == 85.0
        assert hits[0]["contracts"] == 5  # floor(50000 / (85 * 100))

    def test_computed_fields_are_added(self):
        hit = only_hit([make_put()])[0]
        assert hit["return_on_capital"] == pytest.approx(1.50 / 83.50)
        assert abs(hit["delta"]) <= 0.20
        assert hit["underlyingSymbol"] == "TEST"

    def test_theta_is_added_next_to_delta_and_matches_the_function(self):
        chain = make_chain([make_put()])
        hit = qualifying_puts(chain, spot_of(chain))[0]
        keys = list(hit)
        assert keys.index("theta") == keys.index("delta") + 1
        # make_chain defaults: 14 days to expiry, r=0
        assert hit["theta"] == put_theta_bs(
            hit["spot_price"], hit["strike"], 14 / 365, hit["impliedVolatility"], r=0.0
        )
        assert hit["theta"] < 0


class TestBidFloor:
    def test_bid_below_floor_is_discarded(self):
        assert only_hit([make_put(bid=0.04)]) == []

    def test_bid_at_floor_survives_the_bid_check(self):
        # 0.05 clears the floor; it fails later on return-on-capital
        # (0.05/84.95 = 0.06%), which is what this asserts is the *only*
        # reason it drops out.
        assert only_hit([make_put(bid=0.05)]) == []
        assert only_hit([make_put(bid=0.05, strike=10.0)]) == []

    def test_missing_bid_key_treated_as_zero(self):
        """Yahoo omits `bid` entirely on illiquid contracts rather than sending null.

        Seen in fixtures/AAPL-chain.json, where puts carry no `bid` key at all.
        """
        put = make_put()
        del put["bid"]
        assert only_hit([put]) == []

    def test_null_bid_treated_as_zero(self):
        assert only_hit([make_put(bid=None)]) == []


class TestOTM:
    def test_strike_above_8pct_otm_is_rejected(self):
        # spot 100 -> threshold is 92.0
        assert only_hit([make_put(strike=92.5)]) == []

    def test_strike_exactly_at_threshold_is_kept(self):
        hits = only_hit([make_put(strike=92.0, bid=1.50)])
        assert len(hits) == 1

    def test_all_strikes_below_threshold_are_evaluated_not_just_nearest(self):
        """CLAUDE.md: evaluate ALL strikes at or below 8% OTM, not just the nearest."""
        puts = [make_put(strike=s, contractSymbol=f"T{s}") for s in (92.0, 88.0, 85.0, 80.0)]
        assert len(only_hit(puts)) == 4

    def test_distance_form_matches_threshold_price_form(self):
        """The filter was rewritten from `strike > spot * (1 - 0.08)` to
        `(spot - strike) / spot < 0.08` so the tested number could be kept.
        The two are algebraically identical; this pins that they agree in
        floating point too, across the strike grid a real chain uses."""
        for spot in (17.5, 100.0, 233.33, 512.75, 1_000.0):
            for strike in [round(x * 0.5, 2) for x in range(2, int(spot * 2) + 1)]:
                old_keeps = not (strike > spot * (1 - OTM_THRESHOLD))
                new_keeps = not ((spot - strike) / spot < OTM_THRESHOLD)
                assert old_keeps == new_keeps, (spot, strike)


class TestSpotAndOtmFields:
    """spot_price and otm_pct ride out of the filter on every hit, so the
    ranked output can show %OTM without anyone re-fetching a quote."""

    def test_hit_carries_spot_price_and_otm_pct(self):
        hits = only_hit([make_put(strike=85.0)])
        assert hits[0]["spot_price"] == 100.0
        assert hits[0]["otm_pct"] == pytest.approx(0.15)

    def test_spot_price_is_the_value_compared_against_not_the_chain_quote(self):
        """The chain says 100, the caller passes 120. The OTM filter used
        120, so 120 is what must be recorded — a row can never contradict
        the filter that passed it."""
        chain = make_chain([make_put(strike=85.0)], spot=100.0)
        hits = qualifying_puts(chain, 120.0)
        assert len(hits) == 1
        assert hits[0]["spot_price"] == 120.0
        assert hits[0]["otm_pct"] == pytest.approx((120.0 - 85.0) / 120.0)

    def test_otm_pct_is_a_fraction_not_a_percentage(self):
        """Stored like return_on_capital (0.15, not 15.0); formatted at
        display time."""
        hits = only_hit([make_put(strike=85.0)])
        assert 0 < hits[0]["otm_pct"] < 1

    def test_otm_pct_at_the_threshold_equals_the_threshold(self):
        hits = only_hit([make_put(strike=92.0, bid=1.50)])
        assert hits[0]["otm_pct"] == pytest.approx(OTM_THRESHOLD)

    @pytest.mark.parametrize("bad_spot", [0, 0.0, -5.0, None])
    def test_non_positive_spot_raises(self, bad_spot):
        """The OTM filter divides by spot now. A broken quote is an error
        the caller logs and skips, not a silent zero-hit screen."""
        chain = make_chain([make_put()])
        with pytest.raises(ValueError, match="positive price"):
            qualifying_puts(chain, bad_spot)


class TestStrikeRange:
    def test_strike_below_minimum_rejected(self):
        assert only_hit([make_put(strike=9.0, bid=0.20)], spot=100.0) == []

    def test_strike_above_maximum_rejected(self):
        # spot 1000 keeps 501 comfortably OTM, so only the strike cap can reject it
        assert only_hit([make_put(strike=501.0, bid=8.0)], spot=1000.0) == []

    def test_strike_at_bounds_accepted(self):
        assert len(only_hit([make_put(strike=10.0, bid=0.30)], spot=100.0)) == 1
        assert len(only_hit([make_put(strike=500.0, bid=8.0)], spot=1000.0)) == 1


class TestReturnOnCapital:
    def test_return_below_1pct_rejected(self):
        # 0.80 / (85 - 0.80) = 0.95%
        assert only_hit([make_put(bid=0.80)]) == []

    def test_return_at_1pct_accepted(self):
        # solve bid / (85 - bid) = 0.01  ->  bid = 0.8415841...
        bid = round(85 * 0.01 / 1.01, 4)
        hits = only_hit([make_put(bid=bid)])
        assert len(hits) == 1
        assert hits[0]["return_on_capital"] >= 0.01

    def test_return_uses_bid_never_mid(self):
        """A high ask must not rescue a contract whose bid fails the return test."""
        assert only_hit([make_put(bid=0.80, ask=50.0)]) == []


class TestDelta:
    def test_delta_above_threshold_rejected(self):
        # near-the-money + high IV pushes |delta| past 0.20
        assert only_hit([make_put(strike=92.0, impliedVolatility=1.5)]) == []

    def test_missing_iv_rejected(self):
        put = make_put()
        del put["impliedVolatility"]
        assert only_hit([put]) == []

    def test_delta_is_negative_for_puts(self):
        assert only_hit([make_put()])[0]["delta"] < 0


class TestPutTheta:
    """Reference values computed independently of put_theta_bs (plain math,
    with N() from a power series, cross-checked by Simpson integration of
    the pdf, rather than erf) for S=100, K=95, T=0.25, sigma=0.30, in
    dollars per share per calendar day."""

    REF_R0 = -0.030059865636178892
    REF_R5 = -0.024266624177913358

    def test_matches_reference_value_at_zero_rate(self):
        assert put_theta_bs(100, 95, 0.25, 0.30) == pytest.approx(self.REF_R0, rel=1e-9)

    def test_matches_reference_value_with_a_rate(self):
        """Exercises the r*K*exp(-rT)*N(-d2) carry term, which vanishes at r=0."""
        assert put_theta_bs(100, 95, 0.25, 0.30, r=0.05) == pytest.approx(self.REF_R5, rel=1e-9)

    def test_put_call_parity_of_theta(self):
        """theta_put - theta_call = r*K*exp(-rT) per year, so /365 per day.
        The call theta is written out here so it shares no code with the
        implementation under test."""
        s, k, t, sigma, r = 100.0, 95.0, 0.25, 0.30, 0.05
        d1 = (math.log(s / k) + (r + 0.5 * sigma ** 2) * t) / (sigma * math.sqrt(t))
        d2 = d1 - sigma * math.sqrt(t)
        pdf = math.exp(-0.5 * d1 ** 2) / math.sqrt(2 * math.pi)
        cdf = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
        call = (-s * pdf * sigma / (2 * math.sqrt(t)) - r * k * math.exp(-r * t) * cdf(d2)) / 365
        put = put_theta_bs(s, k, t, sigma, r=r)
        assert put - call == pytest.approx(r * k * math.exp(-r * t) / 365, rel=1e-9)

    def test_long_put_theta_is_negative_at_zero_rate(self):
        for strike in (50, 90, 100, 120):
            assert put_theta_bs(100, strike, 14 / 365, 0.35) < 0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"t_years": 0},
            {"t_years": -1},
            {"sigma": 0},
            {"sigma": -0.1},
            {"spot": 0},
            {"strike": 0},
        ],
    )
    def test_degenerate_inputs_return_none(self, kwargs):
        args = {"spot": 100, "strike": 90, "t_years": 0.1, "sigma": 0.3}
        args.update(kwargs)
        assert put_theta_bs(**args) is None
        assert put_delta_bs(**args) is None  # same guard as delta


class TestBlackScholes:
    def test_deep_otm_put_has_near_zero_delta(self):
        # Far enough OTM that N(d1) rounds to 1.0, so delta underflows to
        # exactly 0.0 rather than a small negative. Still a valid delta.
        d = put_delta_bs(spot=100, strike=50, t_years=14 / 365, sigma=0.3)
        assert -0.01 < d <= 0

    def test_atm_put_delta_near_minus_half(self):
        d = put_delta_bs(spot=100, strike=100, t_years=1.0, sigma=0.2)
        assert -0.55 < d < -0.40

    def test_delta_bounded_in_minus_one_to_zero(self):
        for strike in (20, 50, 80, 100, 120, 200):
            d = put_delta_bs(spot=100, strike=strike, t_years=0.5, sigma=0.4)
            assert -1.0 <= d <= 0.0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"t_years": 0},
            {"t_years": -1},
            {"sigma": 0},
            {"sigma": -0.1},
            {"spot": 0},
            {"strike": 0},
        ],
    )
    def test_degenerate_inputs_return_none(self, kwargs):
        args = {"spot": 100, "strike": 90, "t_years": 0.1, "sigma": 0.3}
        args.update(kwargs)
        assert put_delta_bs(**args) is None

    def test_no_scipy_dependency(self):
        """Delta uses math.erf, so the container needs no scipy."""
        import put_filters

        assert not hasattr(put_filters, "scipy")
        assert put_delta_bs(100, 90, 0.1, 0.3) == pytest.approx(
            0.5 * (1 + math.erf(
                ((math.log(100 / 90) + 0.5 * 0.09 * 0.1) / (0.3 * math.sqrt(0.1))) / math.sqrt(2)
            )) - 1.0
        )


class TestLiquidity:
    def test_open_interest_below_10x_contracts_rejected(self):
        # 5 contracts -> needs oi >= 50
        assert only_hit([make_put(openInterest=49)]) == []
        assert len(only_hit([make_put(openInterest=50)])) == 1

    def test_volume_below_2x_contracts_rejected(self):
        # 5 contracts -> needs volume >= 10
        assert only_hit([make_put(volume=9)]) == []
        assert len(only_hit([make_put(volume=10)])) == 1

    def test_missing_oi_and_volume_keys_treated_as_zero(self):
        put = make_put()
        del put["openInterest"]
        del put["volume"]
        assert only_hit([put]) == []

    def test_liquidity_floor_scales_with_position_size(self):
        """A cheaper strike buys more contracts, so it needs proportionally more liquidity."""
        # strike 20 -> floor(50000/2000) = 25 contracts -> oi >= 250, volume >= 50
        assert only_hit([make_put(strike=20.0, bid=0.30, openInterest=249, volume=100)]) == []
        assert len(only_hit([make_put(strike=20.0, bid=0.30, openInterest=250, volume=100)])) == 1

    def test_strike_too_expensive_for_one_contract_rejected(self):
        # strike 600 -> floor(50000/60000) = 0 contracts; also outside the strike cap
        assert only_hit([make_put(strike=600.0, bid=10.0)], spot=1000.0) == []


class TestEmptyAndMalformedChains:
    def test_empty_result_list_returns_empty(self):
        assert qualifying_puts({"optionChain": {"result": []}}, 100.0) == []

    def test_missing_option_chain_key_returns_empty(self):
        assert qualifying_puts({}, 100.0) == []

    def test_nvr_fixture_yields_nothing(self, nvr):
        """NVR returns a result object with zero expirationDates and no options.

        run_screen logs it as a skipped ticker; qualifying_puts must simply
        return empty rather than raise.
        """
        assert qualifying_puts(nvr, 8000.0) == []

    def test_put_without_strike_is_skipped(self):
        put = make_put()
        del put["strike"]
        assert only_hit([put]) == []

    def test_missing_quote_time_means_no_delta_and_no_hits(self):
        chain = make_chain([make_put()])
        del chain["optionChain"]["result"][0]["quote"]["regularMarketTime"]
        assert qualifying_puts(chain, 100.0) == []


class TestRealFixtures:
    """Characterization: pin current behavior against real captured chains."""

    def test_aapl_fixture_runs_and_respects_every_threshold(self, aapl):
        hits = qualifying_puts(aapl, spot_of(aapl))
        threshold = spot_of(aapl) * 0.92
        for h in hits:
            assert h["bid"] >= 0.05
            assert h["strike"] <= threshold
            assert 10 <= h["strike"] <= 500
            assert h["return_on_capital"] >= 0.01
            assert abs(h["delta"]) <= 0.20
            assert h["openInterest"] >= h["contracts"] * 10
            assert h["volume"] >= h["contracts"] * 2

    def test_results_are_deterministic_across_runs(self, aapl):
        """t_years comes from the embedded quote timestamp, not wall clock,
        so a saved fixture must produce identical output every run."""
        assert qualifying_puts(aapl, spot_of(aapl)) == qualifying_puts(aapl, spot_of(aapl))

    def test_brk_b_underlying_symbol_is_yahoos_hyphenated_form(self, brk_b):
        """qualifying_puts copies Yahoo's underlyingSymbol through verbatim.

        Yahoo echoes back the hyphenated symbol it was queried with, so this
        layer emits BRK-B. That is deliberate: put_filters only ever sees a
        chain, never the original ticker. screen_ticker owns restoring the
        dotted form -- see
        test_fetch_and_screen.py::TestOriginalTickerIsPreserved.
        """
        result = brk_b["optionChain"]["result"][0]
        assert result["underlyingSymbol"] == "BRK-B"

        hits = qualifying_puts(brk_b, spot_of(brk_b))
        for h in hits:
            assert h["underlyingSymbol"] == "BRK-B"
