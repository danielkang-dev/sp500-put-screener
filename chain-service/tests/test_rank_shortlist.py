from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
import requests

from rank_shortlist import (
    DAYS_PER_YEAR,
    DELTA_BUCKET_PLACES,
    EARNINGS_RETRY_BACKOFF,
    FINNHUB_BASE_URL,
    MAX_IMPLIED_VOLATILITY,
    MIN_DTE,
    MIN_OPEN_INTEREST,
    TOP_N,
    annualize_returns,
    apply_earnings_exclusion,
    apply_quality_filters,
    as_of_from_manifest,
    backfill_spot_fields,
    compute_dte,
    dedupe_to_best_per_underlying,
    get_next_earnings_date,
    rank_top_n,
    ranking_key,
)

AS_OF = datetime(2026, 8, 20, tzinfo=timezone.utc)


def epoch_for(date_str):
    d = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
    return int(d.timestamp())


def hit(symbol, delta, strike=100.0, expiry="2026-09-03", annualized_return=0.10,
        return_on_capital=0.01, implied_volatility=0.45, open_interest=500):
    return {
        "underlyingSymbol": symbol,
        "contractSymbol": f"{symbol}{strike}",
        "delta": delta,
        "strike": strike,
        "expirationDate": epoch_for(expiry),
        "return_on_capital": return_on_capital,
        "annualized_return": annualized_return,
        "impliedVolatility": implied_volatility,
        "openInterest": open_interest,
    }


class TestAnnualizeReturns:
    """AS_OF is 2026-08-20; expiry 2026-09-03 is 14 days out."""

    def test_dte_computed_from_expiration_minus_as_of(self):
        assert compute_dte(epoch_for("2026-09-03"), as_of=AS_OF) == 14

    def test_dte_floored_at_min_dte_when_expiry_is_today_or_past(self):
        assert compute_dte(epoch_for("2026-08-20"), as_of=AS_OF) == 0  # raw value, pre-floor
        [annotated] = annualize_returns(
            [hit("AAPL", -0.1, expiry="2026-08-20", return_on_capital=0.02)], as_of=AS_OF,
        )
        assert annotated["dte"] == MIN_DTE

    def test_formula_is_roc_times_365_over_dte(self):
        [annotated] = annualize_returns(
            [hit("AAPL", -0.1, expiry="2026-09-03", return_on_capital=0.01)], as_of=AS_OF,
        )
        assert annotated["dte"] == 14
        assert annotated["annualized_return"] == pytest.approx(0.01 * DAYS_PER_YEAR / 14)

    def test_28_day_contract_is_worth_half_of_a_14_day_contract_at_the_same_roc(self):
        """The motivating case: equal raw return_on_capital, different DTE,
        must not rank as equivalent."""
        near, far = annualize_returns(
            [
                hit("NEAR", -0.1, expiry="2026-09-03", return_on_capital=0.01),
                hit("FAR", -0.1, expiry="2026-09-17", return_on_capital=0.01),
            ],
            as_of=AS_OF,
        )
        assert far["annualized_return"] == pytest.approx(near["annualized_return"] / 2)

    def test_return_on_capital_and_other_fields_are_preserved_unchanged(self):
        original = hit("AAPL", -0.1, expiry="2026-09-03", return_on_capital=0.03)
        [annotated] = annualize_returns([original], as_of=AS_OF)
        assert annotated["return_on_capital"] == 0.03
        assert annotated["delta"] == original["delta"]
        assert annotated["underlyingSymbol"] == "AAPL"

    def test_falls_back_to_wall_clock_when_no_as_of_is_given(self):
        """rank() supplies the manifest's run_id; annualize_returns on its own
        still has to produce something sane."""
        far_future = hit("AAPL", -0.1, expiry="2099-01-01", return_on_capital=0.01)
        [annotated] = annualize_returns([far_future])
        assert annotated["dte"] > 1000  # sanity: not stuck at the MIN_DTE floor

    def test_empty_input(self):
        assert annualize_returns([]) == []


class TestQualityFilters:
    def test_iv_above_ceiling_is_dropped(self):
        """The STT case: 175% IV on a custody bank is a stale/wide quote."""
        surviving, exclusions = apply_quality_filters([hit("STT", -0.1, implied_volatility=1.754)])
        assert surviving == []
        assert exclusions[0]["reason"] == "implied_volatility_implausible"
        assert exclusions[0]["ticker"] == "STT"

    def test_iv_exactly_at_ceiling_survives(self):
        surviving, exclusions = apply_quality_filters(
            [hit("AAPL", -0.1, implied_volatility=MAX_IMPLIED_VOLATILITY)]
        )
        assert len(surviving) == 1
        assert exclusions == []

    def test_open_interest_below_floor_is_dropped(self):
        surviving, exclusions = apply_quality_filters([hit("AAPL", -0.1, open_interest=49)])
        assert surviving == []
        assert exclusions[0]["reason"] == "open_interest_below_floor"

    def test_open_interest_exactly_at_floor_survives(self):
        surviving, _ = apply_quality_filters([hit("AAPL", -0.1, open_interest=MIN_OPEN_INTEREST)])
        assert len(surviving) == 1

    def test_missing_open_interest_is_treated_as_zero_and_dropped(self):
        h = hit("AAPL", -0.1)
        del h["openInterest"]
        surviving, exclusions = apply_quality_filters([h])
        assert surviving == []
        assert exclusions[0]["reason"] == "open_interest_below_floor"

    def test_failing_both_is_logged_once_under_iv(self):
        """IV is checked first, so a doubly-bad contract yields one row, not two."""
        _, exclusions = apply_quality_filters(
            [hit("AAPL", -0.1, implied_volatility=2.0, open_interest=1)]
        )
        assert len(exclusions) == 1
        assert exclusions[0]["reason"] == "implied_volatility_implausible"

    def test_exclusions_carry_the_contract_symbol(self):
        """These drop individual contracts, not whole tickers, so the log has
        to say WHICH contract -- the ticker alone would be misleading."""
        _, exclusions = apply_quality_filters([hit("AAPL", -0.1, open_interest=1)])
        assert exclusions[0]["contractSymbol"] == "AAPL100.0"

    def test_is_contract_level_not_ticker_level(self):
        """One bad strike must not condemn the underlying -- the ticker's
        other contracts survive to represent it at dedupe."""
        surviving, _ = apply_quality_filters([
            hit("AAPL", -0.1, strike=90.0, implied_volatility=2.0),
            hit("AAPL", -0.1, strike=95.0, implied_volatility=0.40),
        ])
        assert [h["strike"] for h in surviving] == [95.0]

    def test_clean_contracts_pass_through_unchanged(self):
        hits = [hit("AAPL", -0.1), hit("MSFT", -0.2)]
        surviving, exclusions = apply_quality_filters(hits)
        assert surviving == hits
        assert exclusions == []

    def test_empty_input(self):
        assert apply_quality_filters([]) == ([], [])


class TestAsOfFromManifest:
    def test_parses_run_id(self):
        assert as_of_from_manifest({"run_id": "2026-08-21T15:13:54.174746+00:00"}) == datetime(
            2026, 8, 21, 15, 13, 54, 174746, tzinfo=timezone.utc
        )

    def test_naive_timestamp_is_read_as_utc(self):
        assert as_of_from_manifest({"run_id": "2026-08-21T15:13:54"}).tzinfo == timezone.utc

    def test_missing_run_id_falls_back_to_none(self):
        assert as_of_from_manifest({}) is None

    def test_unparseable_run_id_falls_back_to_none(self):
        """A hand-edited manifest degrades the DTE reference; it must not
        cost the whole shortlist."""
        assert as_of_from_manifest({"run_id": "not-a-timestamp"}) is None


class TestRankingKey:
    def test_delta_is_bucketed_so_the_tiebreak_can_fire(self):
        """Delta is a continuous Black-Scholes output -- 68 real contracts
        produced zero exact ties. Without bucketing, annualized_return could
        never break one."""
        a = hit("A", -0.1200, annualized_return=0.10)
        b = hit("B", -0.1201, annualized_return=0.10)
        assert ranking_key(a)[0] == ranking_key(b)[0]

    def test_lower_is_better_on_both_components(self):
        better_delta = hit("A", -0.05, annualized_return=0.10)
        worse_delta = hit("B", -0.20, annualized_return=0.10)
        assert ranking_key(better_delta) < ranking_key(worse_delta)

        better_ann = hit("C", -0.10, annualized_return=0.30)
        worse_ann = hit("D", -0.10, annualized_return=0.05)
        assert ranking_key(better_ann) < ranking_key(worse_ann)


class TestDedupe:
    def test_keeps_smallest_bucketed_abs_delta_per_underlying(self):
        hits = [
            hit("AAPL", -0.18, annualized_return=0.90),
            hit("AAPL", -0.05, strike=90.0, annualized_return=0.10),
            hit("AAPL", -0.12, annualized_return=0.50),
        ]
        out = dedupe_to_best_per_underlying(hits)
        assert len(out) == 1
        # Smallest |delta| wins outright, even on the worst annualized return.
        assert out[0]["delta"] == -0.05

    def test_annualized_return_breaks_a_delta_tie(self):
        hits = [
            hit("AAPL", -0.1201, strike=90.0, annualized_return=0.12),
            hit("AAPL", -0.1200, strike=95.0, annualized_return=0.31),
        ]
        out = dedupe_to_best_per_underlying(hits)
        assert out[0]["annualized_return"] == 0.31

    def test_one_row_per_underlying(self):
        hits = [
            hit("AAPL", -0.1), hit("MSFT", -0.2), hit("AAPL", -0.3), hit("NVDA", -0.15),
        ]
        out = dedupe_to_best_per_underlying(hits)
        assert sorted(h["underlyingSymbol"] for h in out) == ["AAPL", "MSFT", "NVDA"]

    def test_compares_magnitude_not_sign(self):
        out = dedupe_to_best_per_underlying([hit("AAPL", -0.05), hit("AAPL", 0.20)])
        assert abs(out[0]["delta"]) == 0.05

    def test_matches_the_ranking_order(self):
        """dedupe and rank_top_n share ranking_key, so the contract dedupe
        keeps must be the one ranking would have put first."""
        hits = [
            hit("AAPL", -0.1201, strike=90.0, annualized_return=0.12),
            hit("AAPL", -0.1200, strike=95.0, annualized_return=0.31),
            hit("AAPL", -0.0800, strike=99.0, annualized_return=0.01),
        ]
        assert dedupe_to_best_per_underlying(hits)[0] == rank_top_n(hits)[0]

    def test_empty_input(self):
        assert dedupe_to_best_per_underlying([]) == []


class TestRanking:
    def test_sorted_by_abs_delta_ascending(self):
        hits = [
            hit("A", -0.20, annualized_return=0.90),
            hit("B", -0.05, annualized_return=0.10),
            hit("C", -0.12, annualized_return=0.50),
        ]
        assert [h["underlyingSymbol"] for h in rank_top_n(hits)] == ["B", "C", "A"]

    def test_annualized_return_orders_within_a_delta_bucket(self):
        """The motivating case: same delta band, so the 14-DTE contract's
        better annualized return puts it above the 28-DTE one."""
        hits = [
            hit("FAR", -0.1201, annualized_return=0.143),
            hit("NEAR", -0.1200, annualized_return=0.261),
        ]
        assert [h["underlyingSymbol"] for h in rank_top_n(hits)] == ["NEAR", "FAR"]

    def test_delta_outranks_annualized_return_across_buckets(self):
        """Delta is primary: a much better annualized return does NOT rescue
        a contract from a worse delta band."""
        hits = [
            hit("HIGH_ANN", -0.20, annualized_return=5.00),
            hit("LOW_DELTA", -0.05, annualized_return=0.01),
        ]
        assert [h["underlyingSymbol"] for h in rank_top_n(hits)] == ["LOW_DELTA", "HIGH_ANN"]

    def test_caps_at_top_n(self):
        hits = [hit(f"T{i}", -0.01 * i) for i in range(1, 40)]
        assert len(rank_top_n(hits)) == TOP_N == 20

    def test_returns_fewer_than_n_when_input_is_smaller(self):
        """The real run produced 18, not 20 -- nothing may assume a full 20."""
        hits = [hit(f"T{i}", -0.01 * i) for i in range(1, 19)]
        assert len(rank_top_n(hits)) == 18

    def test_empty_input(self):
        assert rank_top_n([]) == []


class TestEarningsExclusion:
    def test_earnings_before_expiry_is_excluded(self):
        hits = [hit("AAPL", -0.1, expiry="2026-09-03")]
        with patch("rank_shortlist.get_next_earnings_date", return_value="2026-08-28"):
            surviving, exclusions = apply_earnings_exclusion(hits, "k", sleep=0)
        assert surviving == []
        assert exclusions[0]["reason"] == "earnings_before_expiry"

    def test_earnings_after_expiry_survives(self):
        hits = [hit("AAPL", -0.1, expiry="2026-09-03")]
        with patch("rank_shortlist.get_next_earnings_date", return_value="2026-10-15"):
            surviving, exclusions = apply_earnings_exclusion(hits, "k", sleep=0)
        assert len(surviving) == 1
        assert exclusions == []

    def test_earnings_same_day_as_expiry_is_excluded(self):
        """Finnhub gives a date with no BMO/AMC guarantee, so same-day counts
        as before-expiry -- the conservative reading, per the module docstring."""
        hits = [hit("AAPL", -0.1, expiry="2026-09-03")]
        with patch("rank_shortlist.get_next_earnings_date", return_value="2026-09-03"):
            surviving, exclusions = apply_earnings_exclusion(hits, "k", sleep=0)
        assert surviving == []
        assert exclusions[0]["reason"] == "earnings_before_expiry"

    def test_lookup_failure_excludes_the_ticker(self):
        hits = [hit("AAPL", -0.1)]
        with patch("rank_shortlist.get_next_earnings_date", side_effect=RuntimeError("boom")):
            surviving, exclusions = apply_earnings_exclusion(hits, "k", sleep=0)
        assert surviving == []
        assert exclusions[0]["reason"] == "earnings_lookup_failed"

    def test_all_contracts_for_an_excluded_ticker_are_dropped(self):
        hits = [hit("AAPL", -0.1), hit("AAPL", -0.2, strike=90.0), hit("MSFT", -0.1)]

        def fake(symbol, *a, **kw):
            # AAPL: earnings before expiry -> excluded.
            # MSFT: earnings well after expiry -> survives.
            return "2026-08-28" if symbol == "AAPL" else "2026-12-01"

        with patch("rank_shortlist.get_next_earnings_date", side_effect=fake):
            surviving, _ = apply_earnings_exclusion(hits, "k", sleep=0)
        assert {h["underlyingSymbol"] for h in surviving} == {"MSFT"}


class TestEarningsNotFoundIsExcluded:
    """A ticker Finnhub doesn't resolve is excluded, matching the module
    docstring's 'can't confirm it's safe' rule. This used to be a gap: the
    guard read `if earnings_date is not None and ...`, so None silently kept
    the ticker. diagnose_earnings.py measured 0/54 real tickers hitting this
    path across two runs, so closing it changes no observed outcome — it
    only removes the case where a future unresolved symbol would pass the
    earnings filter unverified.
    """

    def test_no_entry_excludes_the_ticker(self):
        hits = [hit("BRK.B", -0.1)]
        with patch("rank_shortlist.get_next_earnings_date", return_value=None):
            surviving, exclusions = apply_earnings_exclusion(hits, "k", sleep=0)
        assert surviving == []
        assert exclusions[0]["reason"] == "earnings_not_found"


class TestFinnhubRequest:
    """Request construction, including where the API key is allowed to appear."""

    def test_returns_earliest_scheduled_date(self):
        payload = {"earningsCalendar": [{"date": "2026-10-01"}, {"date": "2026-09-15"}]}
        with patch("rank_shortlist.requests.get") as g:
            g.return_value.json.return_value = payload
            g.return_value.raise_for_status.return_value = None
            assert get_next_earnings_date("AAPL", "k", as_of=AS_OF) == "2026-09-15"

    def test_returns_none_when_calendar_empty(self):
        with patch("rank_shortlist.requests.get") as g:
            g.return_value.json.return_value = {"earningsCalendar": []}
            g.return_value.raise_for_status.return_value = None
            assert get_next_earnings_date("AAPL", "k", as_of=AS_OF) is None

    def test_lookahead_window_spans_requested_days(self):
        with patch("rank_shortlist.requests.get") as g:
            g.return_value.json.return_value = {"earningsCalendar": []}
            g.return_value.raise_for_status.return_value = None
            get_next_earnings_date("AAPL", "k", lookahead_days=120, as_of=AS_OF)
            params = g.call_args.kwargs["params"]
        assert params["from"] == "2026-08-20"
        assert params["to"] == (AS_OF + timedelta(days=120)).date().isoformat()

    def test_api_key_travels_in_a_header_never_the_query_string(self):
        """requests puts the full URL in its exception messages, and those land
        in the exclusions log and outbound alerts. A key in the query string
        would leak through that path."""
        with patch("rank_shortlist.requests.get") as g:
            g.return_value.json.return_value = {"earningsCalendar": []}
            g.return_value.raise_for_status.return_value = None
            get_next_earnings_date("AAPL", "sekret", as_of=AS_OF)
            kwargs = g.call_args.kwargs

        assert kwargs["headers"]["X-Finnhub-Token"] == "sekret"
        assert "token" not in kwargs["params"]
        assert "sekret" not in str(kwargs["params"])

    def test_http_error_message_carries_no_key(self):
        """The end-to-end property the header fix exists to guarantee."""
        import requests as _rq

        with patch("rank_shortlist.requests.get") as g:
            g.return_value.url = f"{FINNHUB_BASE_URL}?symbol=AAPL&from=2026-08-20"
            g.return_value.raise_for_status.side_effect = _rq.HTTPError(
                f"401 Client Error for url: {g.return_value.url}"
            )
            with pytest.raises(_rq.HTTPError) as exc:
                get_next_earnings_date("AAPL", "sekret", as_of=AS_OF)

        assert "sekret" not in str(exc.value)


class TestFinnhubRetry:
    """A live probe hit requests.exceptions.SSLError against Finnhub and
    succeeded cleanly on a plain retry, so transient connection/SSL errors
    get one retry with a short backoff. HTTP error responses (bad key, rate
    limit) do not retry -- see test_http_error_is_not_retried below."""

    def _ok_response(self):
        r = MagicMock()
        r.raise_for_status.return_value = None
        r.json.return_value = {"earningsCalendar": [{"date": "2026-09-15"}]}
        return r

    def test_connection_error_is_retried_once_then_succeeds(self):
        with patch("rank_shortlist.requests.get") as g, patch("rank_shortlist.time.sleep") as sleep:
            g.side_effect = [requests.exceptions.ConnectionError("boom"), self._ok_response()]
            assert get_next_earnings_date("AAPL", "k", as_of=AS_OF) == "2026-09-15"
        assert g.call_count == 2
        sleep.assert_called_once_with(EARNINGS_RETRY_BACKOFF)

    def test_ssl_error_is_retried_once_then_succeeds(self):
        """requests.exceptions.SSLError subclasses ConnectionError, which is
        what the live PLTR failure actually raised."""
        with patch("rank_shortlist.requests.get") as g, patch("rank_shortlist.time.sleep"):
            g.side_effect = [requests.exceptions.SSLError("handshake failed"), self._ok_response()]
            assert get_next_earnings_date("AAPL", "k", as_of=AS_OF) == "2026-09-15"
        assert g.call_count == 2

    def test_second_consecutive_failure_raises(self):
        with patch("rank_shortlist.requests.get") as g, patch("rank_shortlist.time.sleep"):
            g.side_effect = requests.exceptions.ConnectionError("still down")
            with pytest.raises(requests.exceptions.ConnectionError):
                get_next_earnings_date("AAPL", "k", as_of=AS_OF)
        assert g.call_count == 2

    def test_http_error_is_not_retried(self):
        """A 401/429/5xx response means the server answered -- retrying a bad
        key or a rate limit wastes the pipeline's time budget for nothing."""
        with patch("rank_shortlist.requests.get") as g, patch("rank_shortlist.time.sleep") as sleep:
            g.return_value.raise_for_status.side_effect = requests.exceptions.HTTPError("401")
            with pytest.raises(requests.exceptions.HTTPError):
                get_next_earnings_date("AAPL", "k", as_of=AS_OF)
        assert g.call_count == 1
        sleep.assert_not_called()


class TestBackfillSpotFields:
    """Rank reads spot_price/otm_pct; it never fetches or recomputes them."""

    def test_existing_values_pass_through_untouched(self):
        h = {**hit("AAPL", -0.12), "spot_price": 231.4, "otm_pct": 0.083}
        out = backfill_spot_fields([h])
        assert out[0]["spot_price"] == 231.4
        assert out[0]["otm_pct"] == 0.083

    def test_missing_fields_degrade_to_null_not_keyerror(self):
        """An older screen_results.json predates both fields. That's a
        display gap, not a reason to fail an otherwise good run."""
        out = backfill_spot_fields([hit("AAPL", -0.12)])
        assert out[0]["spot_price"] is None
        assert out[0]["otm_pct"] is None

    def test_missing_fields_are_reported_on_stderr(self, capsys):
        backfill_spot_fields([hit("AAPL", -0.12), hit("MSFT", -0.15)])
        err = capsys.readouterr().err
        assert "2 of 2" in err
        assert "spot_price/otm_pct" in err

    def test_silent_when_every_hit_has_both_fields(self, capsys):
        h = {**hit("AAPL", -0.12), "spot_price": 231.4, "otm_pct": 0.083}
        backfill_spot_fields([h])
        assert capsys.readouterr().err == ""

    def test_does_not_mutate_the_input(self):
        h = hit("AAPL", -0.12)
        backfill_spot_fields([h])
        assert "spot_price" not in h

    def test_partial_file_only_backfills_the_missing_side(self):
        h = {**hit("AAPL", -0.12), "spot_price": 231.4}
        out = backfill_spot_fields([h])
        assert out[0]["spot_price"] == 231.4
        assert out[0]["otm_pct"] is None

    def test_fields_survive_annualization(self):
        """annualize_returns copies with {**hit, ...}; this pins that the
        carry-through actually reaches the ranked output."""
        h = {**hit("AAPL", -0.12), "spot_price": 231.4, "otm_pct": 0.083}
        out = annualize_returns(backfill_spot_fields([h]), as_of=AS_OF)
        assert out[0]["spot_price"] == 231.4
        assert out[0]["otm_pct"] == 0.083
