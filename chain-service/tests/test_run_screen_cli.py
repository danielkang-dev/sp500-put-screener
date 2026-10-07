"""main() must be a thin wrapper: calling it through the CLI must produce
exactly the same stdout and output files as calling run_screen() directly
with the equivalent arguments. These tests prove that property rather than
assuming it -- if main() ever grows printing or logic of its own beyond
forwarding to run_screen(), they catch the drift.
"""

import json
import sys
from unittest.mock import patch

import run_screen
from run_screen import run_screen as run_screen_fn

FAKE_CONSTITUENTS = [
    {"Ticker": "AAPL", "Name": "Apple"},
    {"Ticker": "MSFT", "Name": "Microsoft"},
    {"Ticker": "NVR", "Name": "NVR Inc"},
]


def fake_screen_ticker(ticker, save_raw_dir=None):
    if ticker == "NVR":
        raise ValueError(f"No option chain data returned for {ticker!r}")
    return [{"underlyingSymbol": ticker, "strike": 90.0, "delta": -0.1, "contractSymbol": f"{ticker}T"}]


def _write_constituents(tmp_path, entries=FAKE_CONSTITUENTS):
    p = tmp_path / "constituents.json"
    p.write_text(json.dumps(entries))
    return p


def _load_outputs(out_dir):
    results = json.loads((out_dir / "screen_results.json").read_text())
    errors = json.loads((out_dir / "screen_errors.json").read_text())
    manifest = json.loads((out_dir / "run_manifest.json").read_text())
    # run_id and duration_seconds are wall-clock and legitimately differ
    # between two separate invocations even with identical inputs.
    manifest = {k: v for k, v in manifest.items() if k not in ("run_id", "duration_seconds")}
    return results, errors, manifest


class TestCliIsAThinWrapper:
    """Both invocations below target the SAME output paths, run one after
    the other -- not two separate directories. That makes the comparison
    meaningful: with different directories, the printed file paths would
    differ as text even if the underlying behavior were identical, which
    would be a false failure unrelated to what this test is actually
    checking. Output files are read into memory between runs since the
    second invocation overwrites the first's."""

    def test_stdout_is_byte_identical_to_a_direct_call(self, tmp_path, capsys):
        constituents_path = _write_constituents(tmp_path)
        out_dir = tmp_path / "out"
        argv = [
            "run_screen.py",
            "--constituents", str(constituents_path),
            "--out", str(out_dir),
            "--sleep", "0",
        ]
        with patch("run_screen.screen_ticker", side_effect=fake_screen_ticker):
            with patch.object(sys, "argv", argv):
                run_screen.main()
        cli_stdout = capsys.readouterr().out

        with patch("run_screen.screen_ticker", side_effect=fake_screen_ticker):
            run_screen_fn(
                constituents_path=constituents_path,
                out_dir=out_dir,
                sleep=0,
                limit=None,
                save_raw_dir=None,
            )
        direct_stdout = capsys.readouterr().out

        assert cli_stdout == direct_stdout
        assert cli_stdout != ""  # guard against both sides being empty by accident

    def test_output_files_match_a_direct_call(self, tmp_path):
        constituents_path = _write_constituents(tmp_path)
        out_dir = tmp_path / "out"
        argv = [
            "run_screen.py",
            "--constituents", str(constituents_path),
            "--out", str(out_dir),
            "--sleep", "0",
        ]
        with patch("run_screen.screen_ticker", side_effect=fake_screen_ticker):
            with patch.object(sys, "argv", argv):
                run_screen.main()
        cli_results, cli_errors, cli_manifest = _load_outputs(out_dir)

        with patch("run_screen.screen_ticker", side_effect=fake_screen_ticker):
            run_screen_fn(
                constituents_path=constituents_path,
                out_dir=out_dir,
                sleep=0,
                limit=None,
                save_raw_dir=None,
            )
        direct_results, direct_errors, direct_manifest = _load_outputs(out_dir)

        assert cli_results == direct_results
        assert cli_errors == direct_errors
        assert cli_manifest == direct_manifest

    def test_every_flag_is_forwarded_unchanged(self, tmp_path):
        """Pins the forwarding itself, not just the no-flags case: --limit,
        --sleep, and --save-raw must reach run_screen() as-is."""
        constituents_path = _write_constituents(tmp_path)
        out_dir = tmp_path / "out"
        argv = [
            "run_screen.py",
            "--constituents", str(constituents_path),
            "--out", str(out_dir),
            "--sleep", "2.5",
            "--limit", "2",
            "--save-raw",
        ]
        with patch("run_screen.run_screen") as mock_run:
            with patch.object(sys, "argv", argv):
                run_screen.main()

        mock_run.assert_called_once()
        kwargs = mock_run.call_args.kwargs
        assert str(kwargs["constituents_path"]) == str(constituents_path)
        assert str(kwargs["out_dir"]) == str(out_dir)
        assert kwargs["sleep"] == 2.5
        assert kwargs["limit"] == 2
        assert kwargs["save_raw_dir"] is not None  # --save-raw was passed

    def test_save_raw_dir_is_none_when_flag_omitted(self, tmp_path):
        constituents_path = _write_constituents(tmp_path)
        argv = [
            "run_screen.py",
            "--constituents", str(constituents_path),
            "--out", str(tmp_path / "out"),
            "--sleep", "0",
        ]
        with patch("run_screen.run_screen") as mock_run:
            with patch.object(sys, "argv", argv):
                run_screen.main()

        assert mock_run.call_args.kwargs["save_raw_dir"] is None

    def test_main_calls_run_screen_exactly_once(self, tmp_path):
        """A thin wrapper delegates once -- it must not loop, retry, or
        duplicate the call itself."""
        constituents_path = _write_constituents(tmp_path)
        argv = [
            "run_screen.py",
            "--constituents", str(constituents_path),
            "--out", str(tmp_path / "out"),
        ]
        with patch("run_screen.run_screen") as mock_run:
            with patch.object(sys, "argv", argv):
                run_screen.main()
        assert mock_run.call_count == 1
