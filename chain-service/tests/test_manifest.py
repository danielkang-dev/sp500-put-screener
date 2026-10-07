import pytest

from manifest import (
    MIN_TICKERS_SCREENED,
    STATUS_RUNNING,
    EmptyResultError,
    IncompleteScreenError,
    SanityGateError,
    check_sanity_gate,
    load_manifest,
    new_manifest,
    write_manifest,
)


def healthy_manifest(**overrides):
    """Shaped like the real baseline run: 501/503 screened, 64 qualifying."""
    m = new_manifest(run_id="2026-08-20T00:00:00+00:00", git_commit="abc1234", tickers_total=503)
    m.update(tickers_screened=501, tickers_errored=2, qualifying_contracts=64)
    m.update(overrides)
    return m


class TestNewManifest:
    def test_shape_matches_spec_fields(self):
        m = new_manifest(run_id="r1", git_commit="abc123", tickers_total=503)
        assert set(m.keys()) == {
            "run_id", "status", "git_commit", "tickers_total", "tickers_screened",
            "tickers_errored", "qualifying_contracts", "after_earnings_exclusion",
            "after_quality_filters", "after_dedupe", "final_ranked",
            "finnhub_calls_failed", "duration_seconds",
        }

    def test_status_starts_running(self):
        """A fresh manifest describes a run in flight. Only run_pipeline.py
        ever moves it off this value -- a manual run_screen -> rank_shortlist
        sequence legitimately leaves it here, since nothing in that path
        classifies the outcome."""
        m = new_manifest(run_id="r1", git_commit="abc123", tickers_total=503)
        assert m["status"] == STATUS_RUNNING

    def test_ranking_phase_fields_start_none(self):
        m = new_manifest(run_id="r1", git_commit=None, tickers_total=503)
        assert m["after_earnings_exclusion"] is None
        assert m["after_quality_filters"] is None
        assert m["after_dedupe"] is None
        assert m["final_ranked"] is None
        assert m["finnhub_calls_failed"] is None

    def test_accepts_missing_git_commit(self):
        m = new_manifest(run_id="r1", git_commit=None, tickers_total=503)
        assert m["git_commit"] is None


class TestManifestRoundTrip:
    def test_write_then_load_is_lossless(self, tmp_path):
        path = tmp_path / "run_manifest.json"
        m = healthy_manifest()
        write_manifest(m, path)
        assert load_manifest(path) == m

    def test_write_is_atomic(self, tmp_path):
        """Reuses the same guarantee as screen_results.json / screen_errors.json."""
        path = tmp_path / "run_manifest.json"
        write_manifest(healthy_manifest(), path)
        assert not (tmp_path / "run_manifest.json.tmp").exists()


class TestSanityGateIncompleteScreen:
    def test_fires_just_below_threshold(self):
        m = healthy_manifest(tickers_screened=MIN_TICKERS_SCREENED - 1)
        with pytest.raises(RuntimeError, match="Refusing to rank"):
            check_sanity_gate(m)

    def test_fires_on_a_severely_truncated_screen(self):
        m = healthy_manifest(tickers_screened=3, tickers_errored=0, qualifying_contracts=0)
        with pytest.raises(RuntimeError, match="incomplete"):
            check_sanity_gate(m)

    def test_error_names_the_threshold_and_actual_count(self):
        m = healthy_manifest(tickers_screened=200)
        with pytest.raises(RuntimeError, match=r"200/503.*minimum 490"):
            check_sanity_gate(m)

    def test_passes_exactly_at_threshold(self):
        """490 is the floor, not the cutoff -- 490 itself must be accepted."""
        m = healthy_manifest(tickers_screened=MIN_TICKERS_SCREENED, qualifying_contracts=1)
        check_sanity_gate(m)  # must not raise


class TestSanityGateZeroQualifying:
    def test_fires_when_screen_completed_but_nothing_qualified(self):
        m = healthy_manifest(tickers_screened=503, tickers_errored=0, qualifying_contracts=0)
        with pytest.raises(RuntimeError, match="broken filter"):
            check_sanity_gate(m)

    def test_fires_at_exactly_the_completeness_threshold_too(self):
        """The two conditions are independent -- clearing the completeness
        bar does not exempt a run from the zero-qualifying check."""
        m = healthy_manifest(tickers_screened=MIN_TICKERS_SCREENED, qualifying_contracts=0)
        with pytest.raises(RuntimeError, match="broken filter"):
            check_sanity_gate(m)


class TestSanityGateHealthyRun:
    def test_passes_on_the_real_baseline_shape(self):
        check_sanity_gate(healthy_manifest())  # 501/503, 64 qualifying -- must not raise

    def test_passes_with_zero_errors_and_full_completion(self):
        m = healthy_manifest(tickers_screened=503, tickers_errored=0, qualifying_contracts=1)
        check_sanity_gate(m)

    def test_gate_only_inspects_screening_fields(self):
        """The gate must not depend on ranking-phase fields, since it runs
        before they exist -- rank_shortlist.py's own manifest, pre-ranking,
        still has them as None."""
        m = healthy_manifest(
            after_earnings_exclusion=None, after_dedupe=None,
            final_ranked=None, finnhub_calls_failed=None,
        )
        check_sanity_gate(m)  # must not raise despite the Nones


class TestSanityGateExceptionTypes:
    """The gate distinguishes its two conditions by TYPE, not just by
    message, so run_pipeline.py can classify one as FAILED and the other
    as DEGRADED. Both remain RuntimeError subclasses -- the tests above
    (and any caller predating these types) catch RuntimeError and must
    keep passing untouched."""

    def test_incomplete_screen_raises_its_own_type(self):
        m = healthy_manifest(tickers_screened=200)
        with pytest.raises(IncompleteScreenError):
            check_sanity_gate(m)

    def test_zero_qualifying_raises_its_own_type(self):
        m = healthy_manifest(tickers_screened=503, qualifying_contracts=0)
        with pytest.raises(EmptyResultError):
            check_sanity_gate(m)

    def test_the_two_conditions_are_not_interchangeable(self):
        """The whole point of the split: catching one must not catch the
        other, or run_pipeline.py would misclassify."""
        incomplete = healthy_manifest(tickers_screened=200)
        with pytest.raises(IncompleteScreenError):
            check_sanity_gate(incomplete)
        assert not isinstance(
            IncompleteScreenError("x"), EmptyResultError
        ), "IncompleteScreenError must not be a subclass of EmptyResultError"
        assert not isinstance(
            EmptyResultError("x"), IncompleteScreenError
        ), "EmptyResultError must not be a subclass of IncompleteScreenError"

    def test_incomplete_screen_wins_when_both_conditions_hold(self):
        """A truncated screen usually has zero qualifying contracts too.
        Order matters: the incomplete screen is the actionable cause, the
        empty result is its symptom."""
        m = healthy_manifest(tickers_screened=3, qualifying_contracts=0)
        with pytest.raises(IncompleteScreenError):
            check_sanity_gate(m)

    @pytest.mark.parametrize("exc_type", [IncompleteScreenError, EmptyResultError])
    def test_both_are_runtime_errors(self, exc_type):
        assert issubclass(exc_type, SanityGateError)
        assert issubclass(exc_type, RuntimeError)

    @pytest.mark.parametrize(
        "manifest_kwargs",
        [
            {"tickers_screened": 200},
            {"tickers_screened": 503, "qualifying_contracts": 0},
        ],
    )
    def test_either_condition_is_catchable_as_the_shared_base(self, manifest_kwargs):
        """A caller that only cares THAT the gate tripped, not which
        condition tripped it, catches SanityGateError once."""
        with pytest.raises(SanityGateError):
            check_sanity_gate(healthy_manifest(**manifest_kwargs))
