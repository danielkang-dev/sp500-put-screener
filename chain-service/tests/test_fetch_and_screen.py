import time
from unittest.mock import MagicMock, patch

import pytest
from conftest import FIXTURES, make_chain, make_put

from fetch_chain import _closest_expiration, fetch_raw_chain
from run_screen import load_constituents, screen_ticker

DAY = 86400


class TestClosestExpiration:
    def test_picks_expiration_nearest_to_14_days(self):
        now = time.time()
        candidates = [int(now + d * DAY) for d in (1, 7, 13, 21, 45)]
        assert _closest_expiration(candidates) == int(now + 13 * DAY)

    def test_prefers_nearest_even_when_it_is_further_out(self):
        # 16 days is 2 off the target; 2 days is 12 off.
        now = time.time()
        candidates = [int(now + d * DAY) for d in (2, 16)]
        assert _closest_expiration(candidates) == int(now + 16 * DAY)

    def test_honors_a_custom_target(self):
        # Avoid equidistant candidates: _closest_expiration reads time.time()
        # itself, so a tie is broken by clock drift between here and the call.
        now = time.time()
        candidates = [int(now + d * DAY) for d in (7, 30, 60)]
        assert _closest_expiration(candidates, target_days=35) == int(now + 30 * DAY)

    def test_single_candidate_is_returned(self):
        only = int(time.time() + 99 * DAY)
        assert _closest_expiration([only]) == only


class TestTickerNormalization:
    """CLAUDE.md: '.' -> '-' for the yfinance lookup ONLY."""

    def _fake_yf(self, payload):
        ticker = MagicMock()
        ticker._data.get.return_value.json.return_value = payload
        ticker._data.get.return_value.raise_for_status.return_value = None
        return ticker

    def test_dotted_ticker_is_hyphenated_for_the_lookup(self):
        payload = {"optionChain": {"result": [{"expirationDates": [int(time.time() + 14 * DAY)]}]}}
        fake = self._fake_yf(payload)
        with patch("fetch_chain.yf.Ticker", return_value=fake) as mk:
            fetch_raw_chain("BRK.B")
        mk.assert_called_once_with("BRK-B")
        assert all("BRK-B" in c.kwargs["url"] for c in fake._data.get.call_args_list)

    def test_plain_ticker_is_untouched(self):
        payload = {"optionChain": {"result": [{"expirationDates": [int(time.time() + 14 * DAY)]}]}}
        with patch("fetch_chain.yf.Ticker", return_value=self._fake_yf(payload)) as mk:
            fetch_raw_chain("AAPL")
        mk.assert_called_once_with("AAPL")

    def test_raises_when_no_expiration_dates(self):
        """The real NVR/VMRK failure mode, and the only two errors in the last run."""
        payload = {"optionChain": {"result": [{"expirationDates": []}]}}
        with patch("fetch_chain.yf.Ticker", return_value=self._fake_yf(payload)):
            with pytest.raises(ValueError, match="No expirationDates"):
                fetch_raw_chain("NVR")

    def test_raises_when_result_list_empty(self):
        payload = {"optionChain": {"result": []}}
        with patch("fetch_chain.yf.Ticker", return_value=self._fake_yf(payload)):
            with pytest.raises(ValueError, match="No option chain data"):
                fetch_raw_chain("BOGUS")


class TestScreenTicker:
    def test_returns_qualifying_puts_for_a_healthy_chain(self):
        chain = make_chain([make_put()], symbol="TEST")
        with patch("run_screen.fetch_raw_chain", return_value=chain):
            assert len(screen_ticker("TEST")) == 1

    def test_raises_on_empty_result_so_caller_can_skip(self):
        with patch("run_screen.fetch_raw_chain", return_value={"optionChain": {"result": []}}):
            with pytest.raises(ValueError, match="No option chain data"):
                screen_ticker("NVR")

    def test_raises_when_spot_price_missing(self):
        chain = make_chain([make_put()])
        del chain["optionChain"]["result"][0]["quote"]["regularMarketPrice"]
        with patch("run_screen.fetch_raw_chain", return_value=chain):
            with pytest.raises(ValueError, match="regularMarketPrice"):
                screen_ticker("TEST")

    def test_network_errors_propagate_for_the_caller_to_log(self):
        """screen_ticker swallows nothing; run_screen's loop owns skip-and-log."""
        with patch("run_screen.fetch_raw_chain", side_effect=ConnectionError("down")):
            with pytest.raises(ConnectionError):
                screen_ticker("AAPL")


class TestOriginalTickerIsPreserved:
    """CLAUDE.md: the original dotted ticker stays unchanged in output labels
    and dict keys. Yahoo echoes back the hyphenated form it was queried with,
    so screen_ticker stamps the original back on."""

    def test_dotted_ticker_is_restored_on_every_hit(self):
        chain = make_chain([make_put()], symbol="BRK-B")
        with patch("run_screen.fetch_raw_chain", return_value=chain):
            hits = screen_ticker("BRK.B")
        assert hits
        assert all(h["underlyingSymbol"] == "BRK.B" for h in hits)

    def test_plain_ticker_is_unaffected(self):
        chain = make_chain([make_put()], symbol="AAPL")
        with patch("run_screen.fetch_raw_chain", return_value=chain):
            hits = screen_ticker("AAPL")
        assert all(h["underlyingSymbol"] == "AAPL" for h in hits)

    def test_finnhub_would_be_queried_with_the_dotted_form(self):
        """The reason this matters: rank_shortlist groups by underlyingSymbol
        and passes it straight to Finnhub, which uses periods, not hyphens."""
        chain = make_chain([make_put()], symbol="BF-B")
        with patch("run_screen.fetch_raw_chain", return_value=chain):
            hits = screen_ticker("BF.B")
        assert {h["underlyingSymbol"] for h in hits} == {"BF.B"}


class TestConstituents:
    CONSTITUENTS = FIXTURES / "spy_constituents.json"

    def test_loads_all_503_spy_names(self):
        rows = load_constituents(self.CONSTITUENTS)
        assert len(rows) == 503
        assert all("Ticker" in r for r in rows)

    def test_dotted_tickers_survive_in_the_constituents_file(self):
        rows = load_constituents(self.CONSTITUENTS)
        dotted = {r["Ticker"] for r in rows if "." in r["Ticker"]}
        assert dotted == {"BRK.B", "BF.B"}

    def test_empty_constituents_file_raises(self, tmp_path):
        p = tmp_path / "empty.json"
        p.write_text("[]")
        with pytest.raises(ValueError, match="No constituents"):
            load_constituents(p)
