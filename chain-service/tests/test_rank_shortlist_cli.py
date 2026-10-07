"""End-to-end wiring of the sanity gate and manifest into rank_shortlist's
CLI entrypoint -- the unit tests in test_manifest.py cover the gate logic
itself in isolation; these confirm main() actually calls it, in the right
order, with the right file.
"""

import json
import sys
from unittest.mock import patch

import pytest

import rank_shortlist
from manifest import (
    MIN_TICKERS_SCREENED,
    EmptyResultError,
    load_manifest,
    new_manifest,
    write_manifest,
)


# A real ISO timestamp, because rank() now derives the DTE reference point
# from the manifest's run_id. Expiry epoch 2_000_000_000 is 2033-05-18, so
# every hit below sits exactly 2462 days out from this -- clear of the
# MIN_DTE floor, and fixed regardless of when the suite runs.
RUN_ID = "2026-08-21T15:13:54.174746+00:00"


def write_results(path, hits):
    with open(path, "w") as f:
        json.dump(hits, f)


def a_hit(symbol="AAPL", strike=100.0, delta=-0.1, expiry_epoch=2_000_000_000,
          return_on_capital=0.01, implied_volatility=0.45, open_interest=500):
    return {
        "underlyingSymbol": symbol, "contractSymbol": f"{symbol}T",
        "delta": delta, "strike": strike, "expirationDate": expiry_epoch,
        "return_on_capital": return_on_capital,
        "impliedVolatility": implied_volatility, "openInterest": open_interest,
    }


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    monkeypatch.setenv("FINNHUB_API_KEY", "test-key")
    results_path = tmp_path / "screen_results.json"
    manifest_path = tmp_path / "run_manifest.json"
    out_path = tmp_path / "screen_ranked.json"
    return {"results": results_path, "manifest": manifest_path, "out": out_path}


def run_cli(env):
    argv = [
        "rank_shortlist.py",
        "--results", str(env["results"]),
        "--manifest", str(env["manifest"]),
        "--out", str(env["out"]),
        "--sleep", "0",
    ]
    with patch.object(sys, "argv", argv):
        rank_shortlist.main()


class TestGateBlocksBeforeAnyWork:
    def test_incomplete_screen_raises_and_writes_nothing(self, cli_env):
        write_results(cli_env["results"], [a_hit()])
        write_manifest(
            new_manifest(RUN_ID, "abc", 503)
            | {"tickers_screened": 3, "tickers_errored": 0, "qualifying_contracts": 1},
            cli_env["manifest"],
        )

        with patch("rank_shortlist.apply_earnings_exclusion") as mock_exclusion:
            with pytest.raises(RuntimeError, match="Refusing to rank"):
                run_cli(cli_env)
            mock_exclusion.assert_not_called()

        assert not cli_env["out"].exists()

    def test_zero_qualifying_with_complete_screen_raises(self, cli_env):
        write_results(cli_env["results"], [])
        write_manifest(
            new_manifest(RUN_ID, "abc", 503)
            | {"tickers_screened": 503, "tickers_errored": 0, "qualifying_contracts": 0},
            cli_env["manifest"],
        )

        with patch("rank_shortlist.apply_earnings_exclusion") as mock_exclusion:
            with pytest.raises(RuntimeError, match="broken filter"):
                run_cli(cli_env)
            mock_exclusion.assert_not_called()


class TestGatePassesOnHealthyRun:
    def test_healthy_run_completes_and_updates_manifest(self, cli_env):
        hits = [
            a_hit("AAPL", delta=-0.15, return_on_capital=0.010),
            a_hit("MSFT", delta=-0.05, return_on_capital=0.015),
        ]
        write_results(cli_env["results"], hits)
        write_manifest(
            new_manifest(RUN_ID, "abc", 503)
            | {"tickers_screened": 501, "tickers_errored": 2, "qualifying_contracts": 2},
            cli_env["manifest"],
        )

        # earnings date well after expiry (epoch 2_000_000_000 -> 2033-05-18) for both
        with patch("rank_shortlist.get_next_earnings_date", return_value="2033-12-31"):
            run_cli(cli_env)

        ranked = json.loads(cli_env["out"].read_text())
        # Different delta buckets (0.05 vs 0.15), so delta alone decides:
        # MSFT's smaller |delta| wins regardless of return_on_capital.
        assert [h["underlyingSymbol"] for h in ranked] == ["MSFT", "AAPL"]
        # dte/annualized_return are computed against the manifest's run_id.
        assert all("dte" in h and "annualized_return" in h for h in ranked)

        updated = load_manifest(cli_env["manifest"])
        assert updated["after_earnings_exclusion"] == 2
        assert updated["after_quality_filters"] == 2
        assert updated["after_dedupe"] == 2
        assert updated["final_ranked"] == 2
        assert updated["finnhub_calls_failed"] == 0
        # Screening-phase fields from run_screen.py must survive untouched.
        assert updated["tickers_screened"] == 501
        assert updated["git_commit"] == "abc"

    def test_finnhub_calls_failed_is_counted_not_just_present(self, cli_env):
        hits = [a_hit("AAPL"), a_hit("MSFT")]
        write_results(cli_env["results"], hits)
        write_manifest(
            new_manifest(RUN_ID, "abc", 503)
            | {"tickers_screened": 501, "tickers_errored": 2, "qualifying_contracts": 2},
            cli_env["manifest"],
        )

        def flaky(symbol, *a, **kw):
            if symbol == "AAPL":
                raise RuntimeError("simulated Finnhub failure")
            return "2033-12-31"  # after the 2033-05-18 expiry -> MSFT survives

        with patch("rank_shortlist.get_next_earnings_date", side_effect=flaky):
            run_cli(cli_env)

        updated = load_manifest(cli_env["manifest"])
        assert updated["finnhub_calls_failed"] == 1
        assert updated["final_ranked"] == 1  # only MSFT survives


class TestQualityFiltersEndToEnd:
    def _seed(self, env, hits):
        write_results(env["results"], hits)
        write_manifest(
            new_manifest(RUN_ID, "abc", 503)
            | {"tickers_screened": 501, "tickers_errored": 2,
               "qualifying_contracts": len(hits)},
            env["manifest"],
        )

    def test_bad_iv_and_thin_oi_are_dropped_and_logged(self, cli_env):
        self._seed(cli_env, [
            a_hit("GOOD", delta=-0.10),
            a_hit("STT", delta=-0.05, implied_volatility=1.754),
            a_hit("THIN", delta=-0.05, open_interest=12),
        ])
        with patch("rank_shortlist.get_next_earnings_date", return_value="2033-12-31"):
            run_cli(cli_env)

        ranked = json.loads(cli_env["out"].read_text())
        assert [h["underlyingSymbol"] for h in ranked] == ["GOOD"]

        exclusions = json.loads((cli_env["out"].parent / "screen_ranked_exclusions.json").read_text())
        by_reason = {e["ticker"]: e["reason"] for e in exclusions}
        assert by_reason["STT"] == "implied_volatility_implausible"
        assert by_reason["THIN"] == "open_interest_below_floor"

        updated = load_manifest(cli_env["manifest"])
        assert updated["after_earnings_exclusion"] == 3
        assert updated["after_quality_filters"] == 1
        assert updated["final_ranked"] == 1

    def test_quality_filters_run_before_dedupe(self, cli_env):
        """A ticker whose lowest-delta contract is only lowest because its IV
        is garbage must fall back to its real contract, not vanish."""
        self._seed(cli_env, [
            a_hit("AAPL", strike=90.0, delta=-0.02, implied_volatility=1.9),
            a_hit("AAPL", strike=95.0, delta=-0.11, implied_volatility=0.40),
        ])
        with patch("rank_shortlist.get_next_earnings_date", return_value="2033-12-31"):
            run_cli(cli_env)

        ranked = json.loads(cli_env["out"].read_text())
        assert [h["strike"] for h in ranked] == [95.0]

    def test_finnhub_calls_failed_ignores_the_new_reason_codes(self, cli_env):
        """The count filters on an exact reason string, so contract-level
        exclusions sharing the log must not inflate it."""
        self._seed(cli_env, [a_hit("STT", implied_volatility=1.754), a_hit("GOOD")])
        with patch("rank_shortlist.get_next_earnings_date", return_value="2033-12-31"):
            run_cli(cli_env)
        assert load_manifest(cli_env["manifest"])["finnhub_calls_failed"] == 0


class TestEmptyShortlistIsRefused:
    """The hole this closes: quality filters drop 8 of 38 tickers on an
    ordinary day, so a run filtering down to nothing is reachable. Writing
    [] and returning would exit 0 as SUCCEEDED, and Phase 9 runs unattended
    -- n8n would publish an empty shortlist as today's real result.
    EmptyResultError is the type the sanity gate already uses for "ran
    fine, produced nothing plausible", so run_pipeline classifies it
    DEGRADED (exit 2) with no change on its side.
    """

    def _seed(self, env, hits):
        write_results(env["results"], hits)
        write_manifest(
            new_manifest(RUN_ID, "abc", 503)
            | {"tickers_screened": 501, "tickers_errored": 2,
               "qualifying_contracts": len(hits)},
            env["manifest"],
        )

    def test_all_candidates_filtered_out_raises(self, cli_env):
        self._seed(cli_env, [
            a_hit("STT", implied_volatility=1.754),
            a_hit("THIN", open_interest=3),
        ])
        with patch("rank_shortlist.get_next_earnings_date", return_value="2033-12-31"):
            with pytest.raises(EmptyResultError, match="Refusing to publish an empty shortlist"):
                run_cli(cli_env)

    def test_it_is_a_runtimeerror_so_existing_handlers_still_catch_it(self, cli_env):
        """Phase 8a's contract: callers written before these subclasses
        existed catch RuntimeError and must keep working."""
        self._seed(cli_env, [a_hit("STT", implied_volatility=1.754)])
        with patch("rank_shortlist.get_next_earnings_date", return_value="2033-12-31"):
            with pytest.raises(RuntimeError):
                run_cli(cli_env)

    def test_no_shortlist_is_written(self, cli_env):
        """The whole point: an earlier run's screen_ranked.json must survive
        untouched rather than be replaced with []."""
        cli_env["out"].write_text(json.dumps([{"underlyingSymbol": "YESTERDAY"}]))
        self._seed(cli_env, [a_hit("STT", implied_volatility=1.754)])
        with patch("rank_shortlist.get_next_earnings_date", return_value="2033-12-31"):
            with pytest.raises(EmptyResultError):
                run_cli(cli_env)

        assert json.loads(cli_env["out"].read_text()) == [{"underlyingSymbol": "YESTERDAY"}]

    def test_exclusions_log_is_still_written_for_diagnosis(self, cli_env):
        """The shortlist is withheld, but the record of WHY everything went
        is exactly what the exception sends someone to read."""
        self._seed(cli_env, [
            a_hit("STT", implied_volatility=1.754),
            a_hit("THIN", open_interest=3),
        ])
        with patch("rank_shortlist.get_next_earnings_date", return_value="2033-12-31"):
            with pytest.raises(EmptyResultError):
                run_cli(cli_env)

        exclusions = json.loads((cli_env["out"].parent / "screen_ranked_exclusions.json").read_text())
        assert {e["reason"] for e in exclusions} == {
            "implied_volatility_implausible", "open_interest_below_floor",
        }

    def test_message_carries_the_reason_breakdown(self, cli_env):
        self._seed(cli_env, [
            a_hit("STT", implied_volatility=1.754),
            a_hit("THIN", open_interest=3),
        ])
        with patch("rank_shortlist.get_next_earnings_date", return_value="2033-12-31"):
            with pytest.raises(EmptyResultError) as exc:
                run_cli(cli_env)

        assert "implied_volatility_implausible: 1" in str(exc.value)
        assert "open_interest_below_floor: 1" in str(exc.value)

    def test_everything_dropped_on_earnings_also_raises(self, cli_env):
        """Not specific to the new filters -- an all-earnings wipeout had the
        same hole and takes the same exit."""
        self._seed(cli_env, [a_hit("AAPL"), a_hit("MSFT")])
        with patch("rank_shortlist.get_next_earnings_date", return_value="2033-01-01"):
            with pytest.raises(EmptyResultError, match="earnings_before_expiry: 2"):
                run_cli(cli_env)

    def test_empty_results_file_raises_instead_of_returning(self, cli_env):
        """Reachable when the manifest and the results file disagree -- a
        stale or truncated screen_results.json. The gate passes (the manifest
        claims candidates), so this guard is the only thing standing between
        a stale file and an empty published shortlist."""
        write_results(cli_env["results"], [])
        write_manifest(
            new_manifest(RUN_ID, "abc", 503)
            | {"tickers_screened": 501, "tickers_errored": 2, "qualifying_contracts": 7},
            cli_env["manifest"],
        )
        with pytest.raises(EmptyResultError, match="contains no candidates"):
            run_cli(cli_env)

        assert not cli_env["out"].exists()

    def test_a_single_survivor_still_publishes(self, cli_env):
        """The guard fires on empty, not on small."""
        self._seed(cli_env, [a_hit("GOOD"), a_hit("STT", implied_volatility=1.754)])
        with patch("rank_shortlist.get_next_earnings_date", return_value="2033-12-31"):
            run_cli(cli_env)

        ranked = json.loads(cli_env["out"].read_text())
        assert [h["underlyingSymbol"] for h in ranked] == ["GOOD"]


class TestDteReferenceIsTheManifest:
    def test_dte_is_measured_from_run_id_not_wall_clock(self, cli_env):
        """Expiry 2033-05-18 against a run_id of 2026-08-21 is 2462 days --
        a fixed number, where a wall-clock reference would give a different
        one every day the suite runs."""
        write_results(cli_env["results"], [a_hit("AAPL")])
        write_manifest(
            new_manifest(RUN_ID, "abc", 503)
            | {"tickers_screened": 501, "tickers_errored": 2, "qualifying_contracts": 1},
            cli_env["manifest"],
        )
        with patch("rank_shortlist.get_next_earnings_date", return_value="2033-12-31"):
            run_cli(cli_env)

        [ranked] = json.loads(cli_env["out"].read_text())
        assert ranked["dte"] == 2462
        assert ranked["annualized_return"] == pytest.approx(0.01 * 365 / 2462)

    def test_unparseable_run_id_still_produces_a_shortlist(self, cli_env):
        """Degraded DTE reference, not a lost run -- see as_of_from_manifest."""
        write_results(cli_env["results"], [a_hit("AAPL")])
        write_manifest(
            new_manifest("not-a-timestamp", "abc", 503)
            | {"tickers_screened": 501, "tickers_errored": 2, "qualifying_contracts": 1},
            cli_env["manifest"],
        )
        with patch("rank_shortlist.get_next_earnings_date", return_value="2033-12-31"):
            run_cli(cli_env)

        [ranked] = json.loads(cli_env["out"].read_text())
        assert ranked["dte"] > 0


def _load_rank_outputs(env):
    ranked = json.loads(env["out"].read_text())
    exclusions = json.loads((env["out"].parent / "screen_ranked_exclusions.json").read_text())
    manifest = load_manifest(env["manifest"])
    manifest = {k: v for k, v in manifest.items() if k not in ("run_id", "duration_seconds")}
    return ranked, exclusions, manifest


class TestCliIsAThinWrapper:
    """Both invocations below target the SAME paths, run one after the
    other -- not separate directories, so the printed file-path text
    matches exactly and the comparison isolates real behavioral
    differences rather than incidental path strings. Files are read into
    memory between runs since the second invocation overwrites the
    first's."""

    def _seed(self, env):
        hits = [a_hit("AAPL", delta=-0.15), a_hit("MSFT", delta=-0.05)]
        write_results(env["results"], hits)
        write_manifest(
            new_manifest(RUN_ID, "abc", 503)
            | {"tickers_screened": 501, "tickers_errored": 2, "qualifying_contracts": 2},
            env["manifest"],
        )

    def test_stdout_is_byte_identical_to_a_direct_call(self, cli_env, capsys):
        self._seed(cli_env)

        with patch("rank_shortlist.get_next_earnings_date", return_value="2033-12-31"):
            run_cli(cli_env)
        cli_stdout = capsys.readouterr().out

        self._seed(cli_env)  # reset: rank() already overwrote results/manifest above
        with patch("rank_shortlist.get_next_earnings_date", return_value="2033-12-31"):
            rank_shortlist.rank(
                results_path=cli_env["results"],
                out_path=cli_env["out"],
                manifest_path=cli_env["manifest"],
                api_key="test-key",
                sleep=0,
            )
        direct_stdout = capsys.readouterr().out

        assert cli_stdout == direct_stdout
        assert cli_stdout != ""

    def test_output_files_match_a_direct_call(self, cli_env):
        self._seed(cli_env)
        with patch("rank_shortlist.get_next_earnings_date", return_value="2033-12-31"):
            run_cli(cli_env)
        cli_ranked, cli_excl, cli_manifest = _load_rank_outputs(cli_env)

        self._seed(cli_env)
        with patch("rank_shortlist.get_next_earnings_date", return_value="2033-12-31"):
            rank_shortlist.rank(
                results_path=cli_env["results"],
                out_path=cli_env["out"],
                manifest_path=cli_env["manifest"],
                api_key="test-key",
                sleep=0,
            )
        direct_ranked, direct_excl, direct_manifest = _load_rank_outputs(cli_env)

        assert cli_ranked == direct_ranked
        assert cli_excl == direct_excl
        assert cli_manifest == direct_manifest

    def test_every_flag_is_forwarded_unchanged(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FINNHUB_API_KEY", "the-real-key")
        env = {
            "results": tmp_path / "screen_results.json",
            "manifest": tmp_path / "run_manifest.json",
            "out": tmp_path / "screen_ranked.json",
        }
        argv = [
            "rank_shortlist.py",
            "--results", str(env["results"]),
            "--manifest", str(env["manifest"]),
            "--out", str(env["out"]),
            "--sleep", "3.5",
        ]
        with patch("rank_shortlist.rank") as mock_rank:
            with patch.object(sys, "argv", argv):
                rank_shortlist.main()

        mock_rank.assert_called_once()
        kwargs = mock_rank.call_args.kwargs
        assert str(kwargs["results_path"]) == str(env["results"])
        assert str(kwargs["manifest_path"]) == str(env["manifest"])
        assert str(kwargs["out_path"]) == str(env["out"])
        assert kwargs["sleep"] == 3.5
        assert kwargs["api_key"] == "the-real-key"  # read via _env_api_key(), forwarded unchanged

    def test_main_calls_rank_exactly_once(self, cli_env):
        """rank is mocked entirely, so its manifest/results files never need
        to exist -- main() must delegate once and touch nothing else itself."""
        with patch("rank_shortlist.rank") as mock_rank:
            run_cli(cli_env)
        assert mock_rank.call_count == 1
