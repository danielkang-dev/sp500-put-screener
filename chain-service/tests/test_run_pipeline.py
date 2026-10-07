"""run_pipeline.py is the only place in the pipeline that decides what an
outcome MEANS, so these tests are almost entirely about classification:
which event type, which phase label, which status on disk, which exit
code, for each of the five ways a run can end.

The two phases themselves (run_screen, rank) are patched out throughout.
They have their own tests; what matters here is what the orchestrator does
with what they return or raise.
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

import run_pipeline as pipeline_module
from manifest import (
    STATUS_DEGRADED,
    STATUS_FAILED,
    STATUS_SUCCEEDED,
    EmptyResultError,
    IncompleteScreenError,
    new_manifest,
    write_manifest,
)
from notify import RunDegraded, RunFailed, RunSucceeded
from run_pipeline import (
    EXIT_DEGRADED,
    EXIT_FAILED,
    EXIT_SUCCEEDED,
    PHASE_PREFLIGHT,
    PHASE_RANK,
    PHASE_SCREEN,
    EXIT_CODES,
    NullNotifier,
    preflight,
    run_pipeline,
)


class RecordingNotifier:
    """Stands in for DiscordNotifier. Records instead of sending, so a
    test can assert both WHAT was sent and that exactly one thing was."""

    def __init__(self):
        self.events = []

    def send(self, event):
        self.events.append(event)


class RaisingNotifier:
    def send(self, event):
        raise RuntimeError("notifier exploded")


RANKED = [
    {"underlyingSymbol": "AAPL", "strike": 190.0, "delta": -0.08,
     "return_on_capital": 0.012, "bid": 2.30},
    {"underlyingSymbol": "MSFT", "strike": 380.0, "delta": -0.11,
     "return_on_capital": 0.015, "bid": 4.10},
]


def healthy_manifest(**overrides):
    """The real baseline run's shape: 501/503 screened, 64 qualifying."""
    m = new_manifest(run_id="2026-08-20T00:00:00+00:00", git_commit="abc1234", tickers_total=503)
    m.update(tickers_screened=501, tickers_errored=2, qualifying_contracts=64, duration_seconds=612.4)
    m.update(overrides)
    return m


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    """Every test gets a key by default, so only the test that is ABOUT a
    missing key has to think about it.

    DISCORD_WEBHOOK_URL stays UNSET on purpose, and that is safe because
    every test here injects its own Notifier -- which also switches off
    the webhook preflight check, since that check only describes what the
    default DiscordNotifier needs. The tests that are about the webhook
    check inject nothing, and say so."""
    monkeypatch.setenv("FINNHUB_API_KEY", "test-key-not-real")
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)


@pytest.fixture
def out_dir(tmp_path):
    d = tmp_path / "out"
    d.mkdir()
    return d


def fake_screen(manifest, out_dir):
    """A run_screen stand-in that also does what the real one does to
    disk: leave a checkpointed manifest behind. The orchestrator re-reads
    that file on the success path, so a fake that skips the write would
    test a pipeline the real one never runs."""
    def _screen(**kwargs):
        write_manifest(manifest, Path(out_dir) / "run_manifest.json")
        return manifest
    return _screen


def read_manifest(out_dir):
    return json.loads((Path(out_dir) / "run_manifest.json").read_text())


def _run(out_dir, notifier, screen_fn, rank_fn, **kwargs):
    with patch.object(pipeline_module, "run_screen", side_effect=screen_fn):
        with patch.object(pipeline_module, "rank", side_effect=rank_fn):
            return run_pipeline(out_dir=out_dir, sleep=0, finnhub_sleep=0,
                                notifier=notifier, **kwargs)


class TestSucceeded:
    def test_clean_run_is_succeeded(self, out_dir):
        m = healthy_manifest()
        notifier = RecordingNotifier()

        def rank_fn(**kwargs):
            # The real rank() rewrites the manifest with ranking-phase
            # counts before returning; mirror that here.
            ranked_manifest = dict(m, after_earnings_exclusion=40, after_dedupe=25, final_ranked=2,
                                   finnhub_calls_failed=0)
            write_manifest(ranked_manifest, Path(out_dir) / "run_manifest.json")
            return RANKED, []

        status, manifest = _run(out_dir, notifier, fake_screen(m, out_dir), rank_fn)

        assert status == STATUS_SUCCEEDED
        assert manifest["status"] == STATUS_SUCCEEDED
        assert read_manifest(out_dir)["status"] == STATUS_SUCCEEDED

    def test_sends_exactly_one_succeeded_event_carrying_the_shortlist(self, out_dir):
        m = healthy_manifest()
        notifier = RecordingNotifier()
        _run(out_dir, notifier, fake_screen(m, out_dir), lambda **kw: (RANKED, []))

        assert len(notifier.events) == 1
        event = notifier.events[0]
        assert isinstance(event, RunSucceeded)
        assert event.ranked == RANKED

    def test_success_event_carries_the_ranking_phase_counts(self, out_dir):
        """rank() rewrites the manifest on its way out, so the orchestrator
        must re-read it rather than reporting the pre-ranking copy
        run_screen handed back -- otherwise the success embed's
        'after earnings exclusion' and 'final ranked' fields are None."""
        m = healthy_manifest()
        notifier = RecordingNotifier()

        def rank_fn(**kwargs):
            write_manifest(
                dict(m, after_earnings_exclusion=40, after_dedupe=25, final_ranked=2,
                     finnhub_calls_failed=1),
                Path(out_dir) / "run_manifest.json",
            )
            return RANKED, []

        _run(out_dir, notifier, fake_screen(m, out_dir), rank_fn)

        reported = notifier.events[0].manifest
        assert reported["after_earnings_exclusion"] == 40
        assert reported["final_ranked"] == 2
        assert reported["finnhub_calls_failed"] == 1


class TestRealRankFilteringToNothingIsDegraded:
    """End-to-end with the REAL rank(), not a stand-in.

    Every other test in this file patches rank out, which means none of
    them prove that rank_shortlist actually RAISES when its filters empty
    the shortlist -- only that the orchestrator classifies the exception
    correctly if it arrives. This is the test that connects the two, and
    the regression guard for the hole it closes: before this, a run whose
    filters removed every candidate wrote [] and exited 0 as SUCCEEDED,
    and Phase 9 runs unattended.
    """

    def _hit(self, symbol, implied_volatility=0.45, open_interest=500):
        return {
            "underlyingSymbol": symbol, "contractSymbol": f"{symbol}x",
            "delta": -0.1, "strike": 100.0, "expirationDate": 2_000_000_000,
            "return_on_capital": 0.01, "impliedVolatility": implied_volatility,
            "openInterest": open_interest,
        }

    def _run_for_real(self, out_dir, hits, notifier):
        """Patches run_screen only -- rank() is the real one."""
        m = healthy_manifest(tickers_screened=501, tickers_errored=2,
                             qualifying_contracts=len(hits))
        (Path(out_dir) / "screen_results.json").write_text(json.dumps(hits))
        with patch.object(pipeline_module, "run_screen", side_effect=fake_screen(m, out_dir)):
            with patch("rank_shortlist.get_next_earnings_date", return_value="2033-12-31"):
                return run_pipeline(out_dir=out_dir, sleep=0, finnhub_sleep=0, notifier=notifier)

    def test_filtered_to_nothing_is_degraded_with_exit_code_two(self, out_dir):
        notifier = RecordingNotifier()
        status, manifest = self._run_for_real(
            out_dir,
            [self._hit("STT", implied_volatility=1.754), self._hit("THIN", open_interest=3)],
            notifier,
        )

        assert status == STATUS_DEGRADED
        assert manifest["status"] == STATUS_DEGRADED
        assert EXIT_CODES[status] == EXIT_DEGRADED, (
            "exit 0 here would have n8n publish an empty shortlist as today's result"
        )
        assert isinstance(notifier.events[0], RunDegraded)

    def test_no_shortlist_file_is_written(self, out_dir):
        self._run_for_real(out_dir, [self._hit("STT", implied_volatility=1.754)],
                           RecordingNotifier())
        assert not (Path(out_dir) / "screen_ranked.json").exists()

    def test_a_survivor_still_succeeds_with_exit_zero(self, out_dir):
        """Positive control: the guard must fire on empty, not on thin."""
        notifier = RecordingNotifier()
        status, _ = self._run_for_real(
            out_dir,
            [self._hit("GOOD"), self._hit("STT", implied_volatility=1.754)],
            notifier,
        )

        assert status == STATUS_SUCCEEDED
        assert EXIT_CODES[status] == EXIT_SUCCEEDED
        ranked = json.loads((Path(out_dir) / "screen_ranked.json").read_text())
        assert [h["underlyingSymbol"] for h in ranked] == ["GOOD"]


class TestIncompleteScreenIsBlamedOnTheScreen:
    """The headline classification rule. The sanity gate fires inside
    rank(), but an incomplete screen is run_screen's fault -- reporting it
    as phase='rank' would point whoever reads the alert at the wrong
    script."""

    def _run_incomplete(self, out_dir, notifier):
        m = healthy_manifest(tickers_screened=200, qualifying_contracts=0)

        def rank_fn(**kwargs):
            raise IncompleteScreenError(
                "Refusing to rank: only 200/503 tickers were screened (minimum 490)."
            )

        return _run(out_dir, notifier, fake_screen(m, out_dir), rank_fn)

    def test_status_is_failed_not_degraded(self, out_dir):
        status, manifest = self._run_incomplete(out_dir, RecordingNotifier())
        assert status == STATUS_FAILED
        assert manifest["status"] == STATUS_FAILED

    def test_event_is_run_failed_with_phase_screen(self, out_dir):
        notifier = RecordingNotifier()
        self._run_incomplete(out_dir, notifier)

        assert len(notifier.events) == 1
        event = notifier.events[0]
        assert isinstance(event, RunFailed)
        assert event.phase == PHASE_SCREEN, (
            "an incomplete screen is the screen's fault, even though the gate "
            "only detects it at rank time"
        )

    def test_gate_message_reaches_the_alert(self, out_dir):
        notifier = RecordingNotifier()
        self._run_incomplete(out_dir, notifier)
        assert "200/503" in notifier.events[0].error

    def test_partial_counts_are_reported_not_discarded(self, out_dir):
        notifier = RecordingNotifier()
        self._run_incomplete(out_dir, notifier)
        assert notifier.events[0].manifest["tickers_screened"] == 200

    def test_disk_records_failed(self, out_dir):
        self._run_incomplete(out_dir, RecordingNotifier())
        assert read_manifest(out_dir)["status"] == STATUS_FAILED


class TestZeroQualifyingIsDegraded:
    """The other half of the split: the run completed and did what it was
    told, it just returned something implausible. Different reaction, so
    different event and different exit code."""

    def _run_empty(self, out_dir, notifier):
        m = healthy_manifest(tickers_screened=503, tickers_errored=0, qualifying_contracts=0)

        def rank_fn(**kwargs):
            raise EmptyResultError(
                "Refusing to rank: 503/503 tickers were screened successfully but "
                "zero qualifying contracts were found."
            )

        return _run(out_dir, notifier, fake_screen(m, out_dir), rank_fn)

    def test_status_is_degraded_not_failed(self, out_dir):
        status, manifest = self._run_empty(out_dir, RecordingNotifier())
        assert status == STATUS_DEGRADED
        assert manifest["status"] == STATUS_DEGRADED

    def test_event_is_run_degraded(self, out_dir):
        notifier = RecordingNotifier()
        self._run_empty(out_dir, notifier)

        assert len(notifier.events) == 1
        assert isinstance(notifier.events[0], RunDegraded)

    def test_gate_message_becomes_the_reason(self, out_dir):
        notifier = RecordingNotifier()
        self._run_empty(out_dir, notifier)
        assert "zero qualifying contracts" in notifier.events[0].reason

    def test_disk_records_degraded(self, out_dir):
        self._run_empty(out_dir, RecordingNotifier())
        assert read_manifest(out_dir)["status"] == STATUS_DEGRADED


class TestScreenPhaseFailure:
    def test_run_screen_raising_is_failed_at_phase_screen(self, out_dir):
        notifier = RecordingNotifier()

        def screen_fn(**kwargs):
            raise OSError("constituents file is unreadable")

        status, _ = _run(out_dir, notifier, screen_fn, lambda **kw: (RANKED, []))

        assert status == STATUS_FAILED
        assert len(notifier.events) == 1
        assert isinstance(notifier.events[0], RunFailed)
        assert notifier.events[0].phase == PHASE_SCREEN

    def test_error_text_names_the_exception_type(self, out_dir):
        """str(exception) alone loses the type, and 'connection reset' reads
        very differently from 'KeyError: connection reset'."""
        notifier = RecordingNotifier()

        def screen_fn(**kwargs):
            raise OSError("constituents file is unreadable")

        _run(out_dir, notifier, screen_fn, lambda **kw: (RANKED, []))
        assert "OSError" in notifier.events[0].error

    def test_reports_the_checkpoint_the_screen_left_behind(self, out_dir):
        """run_screen checkpoints after every ticker, so a mid-run crash
        still has a manifest on disk saying how far it got. That is the
        most useful thing the alert can carry."""
        write_manifest(healthy_manifest(tickers_screened=317), out_dir / "run_manifest.json")
        notifier = RecordingNotifier()

        def screen_fn(**kwargs):
            raise ConnectionError("yahoo stopped responding")

        _run(out_dir, notifier, screen_fn, lambda **kw: (RANKED, []))
        assert notifier.events[0].manifest["tickers_screened"] == 317

    def test_survives_a_crash_before_the_first_checkpoint(self, out_dir):
        """No manifest on disk at all -- died on the constituents load. The
        alert must still send rather than the orchestrator dying trying to
        describe the failure."""
        notifier = RecordingNotifier()

        def screen_fn(**kwargs):
            raise ValueError("No constituents found")

        with patch.object(pipeline_module, "git_commit_short", return_value=None):
            status, manifest = _run(out_dir, notifier, screen_fn, lambda **kw: (RANKED, []))

        assert status == STATUS_FAILED
        assert len(notifier.events) == 1
        assert notifier.events[0].manifest["tickers_screened"] == 0

    def test_rank_is_never_reached(self, out_dir):
        def screen_fn(**kwargs):
            raise OSError("boom")

        rank_calls = []

        def rank_fn(**kwargs):
            rank_calls.append(kwargs)
            return RANKED, []

        _run(out_dir, RecordingNotifier(), screen_fn, rank_fn)
        assert rank_calls == []


class TestRankPhaseFailure:
    def test_a_non_gate_exception_is_blamed_on_rank(self, out_dir):
        """Anything that is not the sanity gate genuinely did happen during
        ranking -- a Finnhub outage, a malformed screen_results.json."""
        m = healthy_manifest()
        notifier = RecordingNotifier()

        def rank_fn(**kwargs):
            raise ConnectionError("finnhub.io unreachable")

        status, _ = _run(out_dir, notifier, fake_screen(m, out_dir), rank_fn)

        assert status == STATUS_FAILED
        assert len(notifier.events) == 1
        assert isinstance(notifier.events[0], RunFailed)
        assert notifier.events[0].phase == PHASE_RANK
        assert "ConnectionError" in notifier.events[0].error

    def test_screening_counts_still_reported(self, out_dir):
        m = healthy_manifest()
        notifier = RecordingNotifier()
        _run(out_dir, notifier, fake_screen(m, out_dir),
             lambda **kw: (_ for _ in ()).throw(ConnectionError("down")))
        assert notifier.events[0].manifest["qualifying_contracts"] == 64


class TestPreflightMissingApiKey:
    """A missing FINNHUB_API_KEY is only needed at rank time, but it is
    checked first: a full screen is ~10 minutes and 500+ requests, and one
    missing line in screener.env is the likeliest production failure by
    some distance. It is reported as phase="preflight", NOT phase="rank" --
    an unset environment variable is not a bug in the ranking logic, and
    sending someone into rank_shortlist.py to look for one wastes their
    time."""

    def test_fails_before_the_screen_runs(self, out_dir, monkeypatch):
        monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
        screen_calls = []

        def screen_fn(**kwargs):
            screen_calls.append(kwargs)
            return healthy_manifest()

        with patch.object(pipeline_module, "git_commit_short", return_value=None):
            status, _ = _run(out_dir, RecordingNotifier(), screen_fn, lambda **kw: (RANKED, []))

        assert status == STATUS_FAILED
        assert screen_calls == [], "the screen must not run without the key ranking will need"

    def test_alert_names_the_preflight_phase_and_the_variable(self, out_dir, monkeypatch):
        monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
        notifier = RecordingNotifier()

        with patch.object(pipeline_module, "git_commit_short", return_value=None):
            _run(out_dir, notifier, lambda **kw: healthy_manifest(), lambda **kw: (RANKED, []))

        assert len(notifier.events) == 1
        event = notifier.events[0]
        assert isinstance(event, RunFailed)
        assert event.phase == PHASE_PREFLIGHT
        assert "FINNHUB_API_KEY" in event.error

    def test_phase_is_not_rank(self, out_dir, monkeypatch):
        """Pinned separately from the assertion above, because this is the
        specific regression: the missing key is consumed by the ranking
        phase, which makes 'rank' the tempting and wrong label."""
        monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
        notifier = RecordingNotifier()

        with patch.object(pipeline_module, "git_commit_short", return_value=None):
            _run(out_dir, notifier, lambda **kw: healthy_manifest(), lambda **kw: (RANKED, []))

        assert notifier.events[0].phase != PHASE_RANK

    def test_an_explicit_key_bypasses_the_environment(self, out_dir, monkeypatch):
        """The sidecar may hold the key itself rather than exporting it."""
        monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
        m = healthy_manifest()
        received = {}

        def rank_fn(**kwargs):
            received.update(kwargs)
            return RANKED, []

        status, _ = _run(out_dir, RecordingNotifier(), fake_screen(m, out_dir), rank_fn,
                         api_key="passed-in-key")

        assert status == STATUS_SUCCEEDED
        assert received["api_key"] == "passed-in-key"


class TestProgressHook:
    def test_progress_cb_is_forwarded_to_run_screen(self, out_dir):
        """Phase 6 added run_screen's progress_cb for exactly this. The
        sidecar reports progress through it -- there must not be a second
        progress mechanism."""
        m = healthy_manifest()
        received = {}

        def screen_fn(**kwargs):
            received.update(kwargs)
            write_manifest(m, out_dir / "run_manifest.json")
            return m

        def cb(done, total, ticker):
            pass

        _run(out_dir, RecordingNotifier(), screen_fn, lambda **kw: (RANKED, []), progress_cb=cb)
        assert received["progress_cb"] is cb

    def test_progress_cb_defaults_to_none(self, out_dir):
        m = healthy_manifest()
        received = {}

        def screen_fn(**kwargs):
            received.update(kwargs)
            write_manifest(m, out_dir / "run_manifest.json")
            return m

        _run(out_dir, RecordingNotifier(), screen_fn, lambda **kw: (RANKED, []))
        assert received["progress_cb"] is None


class TestOrchestratorNeverRaises:
    """A pipeline that dies while reporting that it died is the failure
    this module exists to prevent."""

    def test_a_raising_notifier_does_not_break_the_run(self, out_dir):
        m = healthy_manifest()
        status, manifest = _run(out_dir, RaisingNotifier(), fake_screen(m, out_dir),
                                lambda **kw: (RANKED, []))
        assert status == STATUS_SUCCEEDED
        assert manifest["status"] == STATUS_SUCCEEDED

    def test_an_unwritable_manifest_still_sends_the_alert(self, out_dir):
        """The failure paths are exactly where the disk might be the
        problem. The notification is the only thing that reaches a human,
        so it outranks the manifest write."""
        m = healthy_manifest()
        notifier = RecordingNotifier()

        with patch.object(pipeline_module, "write_manifest",
                          side_effect=OSError("read-only filesystem")):
            status, _ = _run(out_dir, notifier, fake_screen(m, out_dir),
                             lambda **kw: (RANKED, []))

        assert status == STATUS_SUCCEEDED
        assert len(notifier.events) == 1

    def test_an_unreadable_manifest_after_ranking_still_succeeds(self, out_dir):
        """The post-rank re-read is an enrichment step, not a gate: ranking
        already succeeded and the shortlist is already written."""
        m = healthy_manifest()
        notifier = RecordingNotifier()

        def rank_fn(**kwargs):
            (out_dir / "run_manifest.json").write_text("{ truncated")
            return RANKED, []

        status, _ = _run(out_dir, notifier, fake_screen(m, out_dir), rank_fn)

        assert status == STATUS_SUCCEEDED
        assert isinstance(notifier.events[0], RunSucceeded)
        assert notifier.events[0].ranked == RANKED


class TestExactlyOneNotification:
    @pytest.mark.parametrize(
        "rank_fn, expected_status",
        [
            (lambda **kw: (RANKED, []), STATUS_SUCCEEDED),
            (lambda **kw: (_ for _ in ()).throw(IncompleteScreenError("x")), STATUS_FAILED),
            (lambda **kw: (_ for _ in ()).throw(EmptyResultError("x")), STATUS_DEGRADED),
            (lambda **kw: (_ for _ in ()).throw(ConnectionError("x")), STATUS_FAILED),
        ],
    )
    def test_one_event_per_run_on_every_branch(self, out_dir, rank_fn, expected_status):
        """Two alerts for one run trains whoever reads them to ignore
        both."""
        m = healthy_manifest()
        notifier = RecordingNotifier()
        status, _ = _run(out_dir, notifier, fake_screen(m, out_dir), rank_fn)

        assert status == expected_status
        assert len(notifier.events) == 1


class TestExitCodes:
    def test_every_status_maps_to_a_code(self):
        assert EXIT_CODES == {STATUS_SUCCEEDED: 0, STATUS_FAILED: 1, STATUS_DEGRADED: 2}

    def test_degraded_is_distinguishable_from_failed(self):
        """n8n branches on the exit code, and 'investigate the filters' is
        not the same instruction as 'the run broke'."""
        assert EXIT_CODES[STATUS_DEGRADED] != EXIT_CODES[STATUS_FAILED]
        assert EXIT_CODES[STATUS_DEGRADED] != EXIT_CODES[STATUS_SUCCEEDED]

    @pytest.mark.parametrize(
        "status, expected_code",
        [(STATUS_SUCCEEDED, 0), (STATUS_FAILED, 1), (STATUS_DEGRADED, 2)],
    )
    def test_main_exits_with_the_status_code(self, tmp_path, status, expected_code):
        argv = ["run_pipeline.py", "--out", str(tmp_path / "out"), "--sleep", "0"]
        with patch.object(pipeline_module, "run_pipeline",
                          return_value=(status, healthy_manifest())):
            with patch.object(sys, "argv", argv):
                with pytest.raises(SystemExit) as excinfo:
                    pipeline_module.main()
        assert excinfo.value.code == expected_code


class TestCliIsAThinWrapper:
    """Same property the run_screen and rank_shortlist CLIs are held to:
    main() forwards, it does not decide. The one thing it adds on top is
    the status-to-exit-code translation, covered above."""

    def test_every_flag_is_forwarded_unchanged(self, tmp_path):
        constituents = tmp_path / "constituents.json"
        constituents.write_text("[]")
        out_dir = tmp_path / "out"
        argv = [
            "run_pipeline.py",
            "--constituents", str(constituents),
            "--out", str(out_dir),
            "--sleep", "2.5",
            "--finnhub-sleep", "0.5",
            "--limit", "7",
            "--save-raw",
        ]
        with patch.object(pipeline_module, "run_pipeline",
                          return_value=(STATUS_SUCCEEDED, healthy_manifest())) as mock_run:
            with patch.object(sys, "argv", argv):
                with pytest.raises(SystemExit):
                    pipeline_module.main()

        kwargs = mock_run.call_args.kwargs
        assert str(kwargs["constituents_path"]) == str(constituents)
        assert str(kwargs["out_dir"]) == str(out_dir)
        assert kwargs["sleep"] == 2.5
        assert kwargs["finnhub_sleep"] == 0.5
        assert kwargs["limit"] == 7
        assert kwargs["save_raw_dir"] is not None

    def test_save_raw_dir_is_none_when_flag_omitted(self, tmp_path):
        argv = ["run_pipeline.py", "--out", str(tmp_path / "out")]
        with patch.object(pipeline_module, "run_pipeline",
                          return_value=(STATUS_SUCCEEDED, healthy_manifest())) as mock_run:
            with patch.object(sys, "argv", argv):
                with pytest.raises(SystemExit):
                    pipeline_module.main()
        assert mock_run.call_args.kwargs["save_raw_dir"] is None

    def test_main_calls_run_pipeline_exactly_once(self, tmp_path):
        argv = ["run_pipeline.py", "--out", str(tmp_path / "out")]
        with patch.object(pipeline_module, "run_pipeline",
                          return_value=(STATUS_SUCCEEDED, healthy_manifest())) as mock_run:
            with patch.object(sys, "argv", argv):
                with pytest.raises(SystemExit):
                    pipeline_module.main()
        assert mock_run.call_count == 1

    def test_notifier_is_none_by_default(self, tmp_path):
        """None is what makes the real run build a DiscordNotifier reading
        DISCORD_WEBHOOK_URL -- and, just as importantly, what leaves the
        webhook preflight check switched ON."""
        argv = ["run_pipeline.py", "--out", str(tmp_path / "out")]
        with patch.object(pipeline_module, "run_pipeline",
                          return_value=(STATUS_SUCCEEDED, healthy_manifest())) as mock_run:
            with patch.object(sys, "argv", argv):
                with pytest.raises(SystemExit):
                    pipeline_module.main()
        assert mock_run.call_args.kwargs["notifier"] is None

    def test_no_notify_passes_a_null_notifier(self, tmp_path):
        argv = ["run_pipeline.py", "--out", str(tmp_path / "out"), "--no-notify"]
        with patch.object(pipeline_module, "run_pipeline",
                          return_value=(STATUS_SUCCEEDED, healthy_manifest())) as mock_run:
            with patch.object(sys, "argv", argv):
                with pytest.raises(SystemExit):
                    pipeline_module.main()
        assert isinstance(mock_run.call_args.kwargs["notifier"], NullNotifier)


def _run_without_injecting_a_notifier(out_dir, screen_fn, rank_fn, recorder, **kwargs):
    """Exercise the real default path -- notifier=None, so run_pipeline
    builds its own and the DISCORD_WEBHOOK_URL preflight check is live.
    DiscordNotifier is swapped for a recorder at the class, not passed in,
    because passing one is exactly what would switch the check off."""
    with patch.object(pipeline_module, "DiscordNotifier", return_value=recorder):
        with patch.object(pipeline_module, "run_screen", side_effect=screen_fn):
            with patch.object(pipeline_module, "rank", side_effect=rank_fn):
                return run_pipeline(out_dir=out_dir, sleep=0, finnhub_sleep=0, **kwargs)


class TestPreflightMissingWebhook:
    """notify.py never raises, by design. The cost of that design is that a
    missing DISCORD_WEBHOOK_URL turns every alert into a line on stderr --
    and on the droplet, unattended, nothing reads stderr. A broken run
    would look exactly like a working one. This check converts that
    silence into the one signal that does survive: a non-zero exit code."""

    def test_missing_webhook_fails_preflight(self, out_dir, monkeypatch):
        monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
        recorder = RecordingNotifier()

        with patch.object(pipeline_module, "git_commit_short", return_value=None):
            status, manifest = _run_without_injecting_a_notifier(
                out_dir, lambda **kw: healthy_manifest(), lambda **kw: (RANKED, []), recorder,
            )

        assert status == STATUS_FAILED
        assert manifest["status"] == STATUS_FAILED

    def test_screen_never_runs(self, out_dir, monkeypatch):
        """The whole point of preflight: catch it before the ~10 minutes,
        not after."""
        monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
        screen_calls = []

        def screen_fn(**kwargs):
            screen_calls.append(kwargs)
            return healthy_manifest()

        with patch.object(pipeline_module, "git_commit_short", return_value=None):
            _run_without_injecting_a_notifier(
                out_dir, screen_fn, lambda **kw: (RANKED, []), RecordingNotifier(),
            )

        assert screen_calls == []

    def test_phase_is_preflight(self, out_dir, monkeypatch):
        monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
        recorder = RecordingNotifier()

        with patch.object(pipeline_module, "git_commit_short", return_value=None):
            _run_without_injecting_a_notifier(
                out_dir, lambda **kw: healthy_manifest(), lambda **kw: (RANKED, []), recorder,
            )

        assert len(recorder.events) == 1
        event = recorder.events[0]
        assert isinstance(event, RunFailed)
        assert event.phase == PHASE_PREFLIGHT
        assert "DISCORD_WEBHOOK_URL" in event.error

    def test_stderr_says_the_alert_cannot_be_delivered(self, out_dir, monkeypatch, capsys):
        """The alert about the missing webhook is the one alert the missing
        webhook guarantees nobody will see. Whoever IS at a terminal should
        be told that the exit code is the only report."""
        monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)

        with patch.object(pipeline_module, "git_commit_short", return_value=None):
            _run_without_injecting_a_notifier(
                out_dir, lambda **kw: healthy_manifest(), lambda **kw: (RANKED, []),
                RecordingNotifier(),
            )

        err = capsys.readouterr().err
        assert "PREFLIGHT FAILED" in err
        assert f"exit code {EXIT_FAILED}" in err

    def test_a_set_webhook_passes_preflight(self, out_dir, monkeypatch):
        monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/1/abc")
        m = healthy_manifest()
        recorder = RecordingNotifier()

        status, _ = _run_without_injecting_a_notifier(
            out_dir, fake_screen(m, out_dir), lambda **kw: (RANKED, []), recorder,
        )

        assert status == STATUS_SUCCEEDED
        assert isinstance(recorder.events[0], RunSucceeded)


class TestPreflightReportsEverythingAtOnce:
    def test_both_missing_secrets_appear_in_one_alert(self, out_dir, monkeypatch):
        """One round trip to fix a misconfigured screener.env, not two."""
        monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
        monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
        recorder = RecordingNotifier()

        with patch.object(pipeline_module, "git_commit_short", return_value=None):
            status, _ = _run_without_injecting_a_notifier(
                out_dir, lambda **kw: healthy_manifest(), lambda **kw: (RANKED, []), recorder,
            )

        assert status == STATUS_FAILED
        error = recorder.events[0].error
        assert "FINNHUB_API_KEY" in error
        assert "DISCORD_WEBHOOK_URL" in error


class TestPreflightFunction:
    """preflight() is separable from the run, so the sidecar can check the
    environment on startup rather than discovering it at 3am."""

    def test_clean_environment_reports_no_problems(self, monkeypatch):
        monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/1/abc")
        api_key, problems = preflight()
        assert problems == []
        assert api_key == "test-key-not-real"

    def test_explicit_api_key_skips_the_env_lookup(self, monkeypatch):
        monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
        monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/1/abc")
        api_key, problems = preflight(api_key="explicit")
        assert problems == []
        assert api_key == "explicit"

    def test_check_webhook_false_ignores_an_unset_webhook(self, monkeypatch):
        """An injected notifier may not be a DiscordNotifier at all, so
        DISCORD_WEBHOOK_URL is none of its business."""
        monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
        _api_key, problems = preflight(check_webhook=False)
        assert problems == []

    def test_an_empty_webhook_counts_as_missing(self, monkeypatch):
        """DISCORD_WEBHOOK_URL= in screener.env sets it to the empty string,
        which is exactly as useless as not setting it at all."""
        monkeypatch.setenv("DISCORD_WEBHOOK_URL", "")
        _api_key, problems = preflight()
        assert len(problems) == 1
        assert "DISCORD_WEBHOOK_URL" in problems[0]

    def test_both_checks_run_even_when_the_first_fails(self, monkeypatch):
        monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
        monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
        _api_key, problems = preflight()
        assert len(problems) == 2


class TestInjectedNotifierSkipsTheWebhookCheck:
    def test_recording_notifier_runs_without_a_webhook(self, out_dir, monkeypatch):
        monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
        m = healthy_manifest()
        status, _ = _run(out_dir, RecordingNotifier(), fake_screen(m, out_dir),
                         lambda **kw: (RANKED, []))
        assert status == STATUS_SUCCEEDED

    def test_null_notifier_runs_without_a_webhook(self, out_dir, monkeypatch):
        """The --no-notify path. It disables alerting outright rather than
        merely muting the check, so there is no half-notifying state."""
        monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
        m = healthy_manifest()
        status, _ = _run(out_dir, NullNotifier(), fake_screen(m, out_dir),
                         lambda **kw: (RANKED, []))
        assert status == STATUS_SUCCEEDED


class TestExitCodeContract:
    """n8n branches on these numbers. They are the module's public API as
    much as any function here, so they are pinned by value, not derived."""

    def test_succeeded_is_zero(self):
        assert EXIT_SUCCEEDED == 0
        assert EXIT_CODES[STATUS_SUCCEEDED] == 0

    def test_failed_is_one(self):
        assert EXIT_FAILED == 1
        assert EXIT_CODES[STATUS_FAILED] == 1

    def test_degraded_is_two(self):
        assert EXIT_DEGRADED == 2
        assert EXIT_CODES[STATUS_DEGRADED] == 2

    def test_degraded_is_non_zero(self):
        """A degraded run writes no shortlist. If it exited 0, n8n would
        take the success branch and republish the previous run's file as
        though it were today's."""
        assert EXIT_DEGRADED != 0

    def test_only_success_is_zero(self):
        non_success = [code for status, code in EXIT_CODES.items() if status != STATUS_SUCCEEDED]
        assert all(code != 0 for code in non_success)

    def test_all_three_codes_are_distinct(self):
        assert len(set(EXIT_CODES.values())) == 3

    def test_help_text_documents_the_codes(self, capsys):
        """Whoever wires the n8n branch reads --help, not the source."""
        with patch.object(sys, "argv", ["run_pipeline.py", "--help"]):
            with pytest.raises(SystemExit):
                pipeline_module.main()
        out = capsys.readouterr().out
        assert "exit codes" in out
        assert "succeeded" in out and "failed" in out and "degraded" in out


class TestOutputDirIsCreatedBeforePreflight:
    def test_preflight_failure_still_records_status_on_disk(self, tmp_path, monkeypatch):
        """run_screen creates out_dir, but preflight runs before it -- and
        on a first deploy, when a missing screener.env is most likely, the
        directory does not exist yet. The failure manifest still has to
        land somewhere."""
        monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
        fresh = tmp_path / "never-created"

        with patch.object(pipeline_module, "git_commit_short", return_value=None):
            status, _ = _run(fresh, RecordingNotifier(), lambda **kw: healthy_manifest(),
                             lambda **kw: (RANKED, []))

        assert status == STATUS_FAILED
        assert json.loads((fresh / "run_manifest.json").read_text())["status"] == STATUS_FAILED

    def test_an_uncreatable_out_dir_does_not_crash_the_run(self, tmp_path, monkeypatch):
        """Same principle as the manifest write: a broken disk must still
        produce a status and an alert, not a traceback."""
        monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
        blocker = tmp_path / "blocker"
        blocker.write_text("I am a file, not a directory")
        notifier = RecordingNotifier()

        with patch.object(pipeline_module, "git_commit_short", return_value=None):
            status, _ = _run(blocker / "out", notifier, lambda **kw: healthy_manifest(),
                             lambda **kw: (RANKED, []))

        assert status == STATUS_FAILED
        assert len(notifier.events) == 1


class TestNoNotifyIsTheProductionPath:
    """The webhook lives in n8n's credential store, not the environment.
    n8n alerts by branching on the exit code, so a production run passes
    --no-notify and the pipeline stays quiet. This inverts the flag's
    original meaning -- it was 'never pass this in n8n' -- so the help
    text is pinned rather than left to drift back."""

    def _help_text(self, capsys):
        with patch.object(sys, "argv", ["run_pipeline.py", "--help"]):
            with pytest.raises(SystemExit):
                pipeline_module.main()
        return " ".join(capsys.readouterr().out.split())

    def test_help_says_the_flag_is_required_for_n8n(self, capsys):
        help_text = self._help_text(capsys)
        assert "REQUIRED for the n8n path" in help_text

    def test_help_does_not_still_forbid_the_flag_in_n8n(self, capsys):
        """The exact wording that is now wrong."""
        help_text = self._help_text(capsys)
        assert "Never pass this in n8n" not in help_text

    def test_help_says_n8n_owns_alerting(self, capsys):
        help_text = self._help_text(capsys)
        assert "n8n owns alerting" in help_text

    def test_the_flag_still_disables_alerting_outright(self, out_dir, monkeypatch):
        """Its behaviour is unchanged -- only its intended audience moved.
        NullNotifier sends nothing and skips the webhook check together."""
        monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
        m = healthy_manifest()
        notifier = NullNotifier()
        status, _ = _run(out_dir, notifier, fake_screen(m, out_dir), lambda **kw: (RANKED, []))
        assert status == STATUS_SUCCEEDED
