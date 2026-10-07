#!/usr/bin/env python3
"""
run_pipeline.py — one command that runs a screen end to end: check the
environment, screen the index, rank the survivors, then say out loud what
happened.

Everything this calls already existed and already worked. What was missing
was the layer that decides what a given outcome MEANS. run_screen.py and
rank_shortlist.py each report their own mechanical result (files written,
counts printed, exceptions raised); notify.py can render any of three
outcomes to Discord but, by its own docstring, deliberately refuses to
classify. Nothing owned the judgment in between. This does, and it is the
only module in the pipeline that does:

    preflight check fails              -> FAILED,    phase="preflight"
    run_screen raises                  -> FAILED,    phase="screen"
    IncompleteScreenError from gate    -> FAILED,    phase="screen"   (*)
    EmptyResultError                   -> DEGRADED                    (**)
    rank raises anything else          -> FAILED,    phase="rank"
    rank returns                       -> SUCCEEDED

  (*) The gate detects an incomplete screen at RANK time — it is the
      ranker that loads the manifest and refuses — but the fault is in the
      screen: run_screen.py was interrupted before it finished the index.
      Labelling that alert phase="rank" would send whoever reads it to
      the wrong script, so it is reported against the phase where the
      fault originated, not the phase that noticed it. `phase` answers
      "which file do I open", nothing else.

 (**) EmptyResultError reaches here from two places, deliberately sharing
      one type because they want one classification. The sanity gate
      raises it when the SCREEN found zero qualifying contracts;
      rank_shortlist raises it when RANKING filtered every candidate back
      out (earnings, implausible IV, thin open interest) or found
      screen_results.json empty. Either way the pipeline ran correctly and
      produced nothing publishable, and either way no shortlist is
      written — so both are DEGRADED, and this module does not need to
      tell them apart. The exception message says which.

The failed/degraded split follows the same reasoning. An incomplete screen
means the pipeline broke; a run that completes but yields no publishable
shortlist means the pipeline ran fine and returned something implausible
(see put_filters.py, and rank_shortlist.py's filters). Those want
different reactions from whoever is paged, so they get different colors,
and different exit codes.

EXIT CODES — the contract n8n branches on:

    0   SUCCEEDED   ranked shortlist written to screen_ranked.json
    1   FAILED      no shortlist; something broke
    2   DEGRADED    no shortlist; the run completed but the result is
                    implausible enough that ranking was refused

Degraded is deliberately NON-ZERO. It produces no shortlist, so treating
it as success would leave n8n publishing yesterday's file as if it were
today's. It is a separate code from FAILED because the two want different
handling: 1 means "the pipeline is broken, go read the traceback", 2 means
"the pipeline is fine, go look at put_filters.py".

PREFLIGHT — both secrets are checked BEFORE the screen starts:

  - FINNHUB_API_KEY is only needed at rank time, but a full-index screen
    costs ~10 minutes of wall clock and 500+ Yahoo requests. Discovering
    an unset variable after paying that is avoidable.
  - DISCORD_WEBHOOK_URL is checked because notify.py never raises. Without
    it, every alert this module sends degrades to a line on stderr, and a
    run that alerts nobody is indistinguishable from one that worked. The
    check converts that silence into a non-zero exit code.

    This check applies to LOCAL runs only. In production the webhook is
    not an environment variable at all — it lives in n8n's credential
    store, and n8n does the alerting by branching on the exit code. So the
    n8n path always passes --no-notify (and the sidecar in api.py always
    injects NullNotifier), which switches both the alerting and this check
    off together. Leaving the check on for local runs is still worth it:
    that is where a silently unalerted run is actually possible.

Both are reported together rather than one at a time, so a misconfigured
environment takes one round trip to fix instead of two.

run_screen.py and rank_shortlist.py are untouched by this and stay
notification-free — importing them here does not change what either does
when run by hand, in the manual order CLAUDE.md documents.

Places no trades. Same as everything else here, this only ever produces a
ranked list and a Discord message about it.

Usage:
    export FINNHUB_API_KEY=...
    export DISCORD_WEBHOOK_URL=...
    python run_pipeline.py [--constituents PATH] [--out DIR] [--sleep SECONDS]
                           [--finnhub-sleep SECONDS] [--limit N] [--save-raw]
                           [--no-notify]
"""

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from io_utils import git_commit_short
from manifest import (
    STATUS_DEGRADED,
    STATUS_FAILED,
    STATUS_SUCCEEDED,
    EmptyResultError,
    IncompleteScreenError,
    load_manifest,
    new_manifest,
    write_manifest,
)
from notify import DiscordNotifier, Notifier, RunDegraded, RunFailed, RunSucceeded
from rank_shortlist import DEFAULT_SLEEP as DEFAULT_FINNHUB_SLEEP
# _env_api_key is private to rank_shortlist, and crossed here on purpose:
# CLAUDE.md's "no real API keys in any file" rule is enforced by that one
# function reading FINNHUB_API_KEY from the environment and nowhere else.
# Re-implementing the same check here would mean two places to get it
# wrong. Importing it means there is still exactly one.
from rank_shortlist import _env_api_key, rank
from run_screen import DEFAULT_CONSTITUENTS, DEFAULT_OUT_DIR, DEFAULT_SLEEP, run_screen

# `phase` on a RunFailed alert answers exactly one question: which file
# should the person woken up by it open first.
PHASE_PREFLIGHT = "preflight"
PHASE_SCREEN = "screen"
PHASE_RANK = "rank"

EXIT_SUCCEEDED = 0
EXIT_FAILED = 1
EXIT_DEGRADED = 2

EXIT_CODES = {
    STATUS_SUCCEEDED: EXIT_SUCCEEDED,
    STATUS_FAILED: EXIT_FAILED,
    STATUS_DEGRADED: EXIT_DEGRADED,
}

_MISSING_WEBHOOK_MESSAGE = (
    "DISCORD_WEBHOOK_URL is not set. notify.py never raises, so without it "
    "every alert this run would send degrades to a line on stderr — and a "
    "failed run that alerts nobody is indistinguishable from a successful "
    "one. Either export it, or pass --no-notify to run without alerting. "
    "The n8n path always passes --no-notify: there the webhook lives in "
    "n8n's credential store and n8n alerts by branching on the exit code, "
    "so this variable is deliberately absent on the droplet."
)


class NullNotifier:
    """Sends nothing, on purpose.

    Selected by --no-notify, and by the sidecar unconditionally. This is
    the PRODUCTION path, not a local convenience: in n8n, DISCORD_WEBHOOK_URL
    lives in n8n's credential store rather than the environment, and n8n
    does the alerting itself by branching on the exit code. A pipeline that
    also posted to Discord would double-report every run.

    It disables notification outright rather than merely skipping the
    preflight check — a run that half-notifies is worse than one that
    clearly does not, and the exit code still reports the outcome either
    way.
    """

    def send(self, event: Any) -> None:
        pass


def _stub_manifest(tickers_total: int = 0) -> Dict[str, Any]:
    """A manifest for a run that died before run_screen wrote its first
    checkpoint — including one that never started, because preflight
    stopped it. Every count is zero and honest about it; the alert's
    summary fields render as 0/0 rather than crashing the notifier on a
    missing key."""
    return new_manifest(
        run_id=datetime.now(timezone.utc).isoformat(),
        git_commit=git_commit_short(),
        tickers_total=tickers_total,
    )


def preflight(api_key: Optional[str] = None, check_webhook: bool = True) -> Tuple[Optional[str], List[str]]:
    """Check everything the run will need before it costs anything.

    Returns (api_key, problems). An empty `problems` list means go. Both
    checks run even when the first one fails, so one round trip surfaces
    everything wrong with a misconfigured environment.

    `check_webhook` is False whenever the caller injected its own
    Notifier: DISCORD_WEBHOOK_URL is what the DEFAULT notifier reads, and
    a caller sending somewhere else entirely — a test, a future sidecar —
    is not misconfigured for leaving it unset.
    """
    problems: List[str] = []

    if api_key is None:
        try:
            api_key = _env_api_key()
        except RuntimeError as e:
            problems.append(str(e))

    if check_webhook and not os.environ.get("DISCORD_WEBHOOK_URL"):
        problems.append(_MISSING_WEBHOOK_MESSAGE)

    return api_key, problems


def _manifest_after_screen_failure(manifest_path: Path) -> Dict[str, Any]:
    """The best available manifest once run_screen has already blown up.

    run_screen checkpoints after every ticker, so in the usual case (it
    died 200 tickers in) there is a real, current manifest on disk and the
    alert can report exactly how far the run got. If it died before the
    first checkpoint — an unreadable constituents file, a bad --out path —
    there is nothing there, and a stub keeps the alert sendable.
    """
    try:
        return load_manifest(manifest_path)
    except (OSError, ValueError):
        return _stub_manifest()


def _write_manifest_best_effort(manifest: Dict[str, Any], manifest_path: Path) -> None:
    """Persist the final status, but never at the cost of the alert.

    This runs on the failure paths, where the disk is exactly what might
    be wrong. If the write fails, the notification is still the more
    important of the two — it is the only thing that reaches a human — so
    this swallows and reports rather than raising over it.
    """
    try:
        write_manifest(manifest, manifest_path)
    except OSError as e:
        print(f"[pipeline] could not write manifest to {manifest_path}: "
              f"{type(e).__name__}: {e}", file=sys.stderr)


def _terminal(
    notifier: Notifier,
    manifest: Dict[str, Any],
    manifest_path: Path,
    status: str,
    make_event: Callable[[Dict[str, Any]], Any],
) -> Tuple[str, Dict[str, Any]]:
    """End the run: stamp `status`, persist, notify, return.

    The stamp happens before the event is built so the embed and the file
    on disk can never disagree about how the run ended — they are built
    from the same dict, in that order, at one place.
    """
    manifest["status"] = status
    _write_manifest_best_effort(manifest, manifest_path)

    # DiscordNotifier.send() already guarantees it never raises. This
    # guards the injected-notifier case instead: a caller's own Notifier
    # must not be able to destroy an otherwise-complete run's exit code
    # on its way out the door.
    try:
        notifier.send(make_event(manifest))
    except Exception as e:
        print(f"[pipeline] notifier raised on {status} event: "
              f"{type(e).__name__}: {e}", file=sys.stderr)

    print(f"\nPipeline status: {status.upper()}")
    return status, manifest


def run_pipeline(
    constituents_path: Union[str, Path] = DEFAULT_CONSTITUENTS,
    out_dir: Union[str, Path] = DEFAULT_OUT_DIR,
    sleep: float = DEFAULT_SLEEP,
    limit: Optional[int] = None,
    save_raw_dir: Optional[Path] = None,
    api_key: Optional[str] = None,
    finnhub_sleep: float = DEFAULT_FINNHUB_SLEEP,
    notifier: Optional[Notifier] = None,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Preflight, screen, rank, classify, notify. Returns (status, manifest).

    Never raises: every outcome, including a crash in either phase, is a
    status and a notification. That is the whole point of the module — a
    pipeline that dies silently at 3am is the failure this exists to make
    impossible.

    `progress_cb(done, total, ticker)` is forwarded straight to
    run_screen(), whose progress hook has been sitting unused since Phase
    6 waiting for exactly this. The FastAPI sidecar reports progress
    through this parameter; there is deliberately no second progress
    mechanism anywhere in the pipeline.

    `notifier` defaults to DiscordNotifier() reading DISCORD_WEBHOOK_URL
    from the environment. Inject one to test, or to send somewhere else —
    doing so also skips the DISCORD_WEBHOOK_URL preflight check, which
    only describes the default notifier's needs.
    """
    out_dir = Path(out_dir)
    manifest_path = out_dir / "run_manifest.json"
    results_path = out_dir / "screen_results.json"
    ranked_path = out_dir / "screen_ranked.json"

    # run_screen creates this itself, but preflight can fail before
    # run_screen is ever called -- and on a first deploy, when preflight
    # failing is most likely, the directory does not exist yet. Without
    # this the failure manifest silently has nowhere to land.
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"[pipeline] could not create {out_dir}: {type(e).__name__}: {e}",
              file=sys.stderr)

    # Resolved before the notifier is built, since whether the caller
    # brought its own is exactly what decides if the webhook check applies.
    check_webhook = notifier is None
    notifier = notifier if notifier is not None else DiscordNotifier()

    api_key, problems = preflight(api_key=api_key, check_webhook=check_webhook)
    if problems:
        if check_webhook and not os.environ.get("DISCORD_WEBHOOK_URL"):
            # The alert about the missing webhook cannot itself be
            # delivered by the missing webhook. Say so on stderr, and lean
            # on the exit code, which is the only signal that survives an
            # unattended run.
            print("[pipeline] PREFLIGHT FAILED and DISCORD_WEBHOOK_URL is unset, so "
                  "no alert can be sent about it — this run reports itself only "
                  f"through exit code {EXIT_FAILED}.", file=sys.stderr)
        return _terminal(
            notifier, _stub_manifest(), manifest_path, STATUS_FAILED,
            lambda m: RunFailed(manifest=m, error="\n".join(problems), phase=PHASE_PREFLIGHT),
        )

    try:
        manifest = run_screen(
            constituents_path=constituents_path,
            out_dir=out_dir,
            sleep=sleep,
            limit=limit,
            save_raw_dir=save_raw_dir,
            progress_cb=progress_cb,
        )
    except Exception as e:
        return _terminal(
            notifier, _manifest_after_screen_failure(manifest_path), manifest_path,
            STATUS_FAILED,
            lambda m: RunFailed(manifest=m, error=f"{type(e).__name__}: {e}", phase=PHASE_SCREEN),
        )

    # The manifest run_screen returned is the same one it just checkpointed
    # to disk, and rank() only rewrites the file AFTER the gate and the
    # ranking both succeed — so on every branch below except the last, this
    # in-memory copy is current and no re-read is needed.
    try:
        ranked, _exclusions = rank(
            results_path=results_path,
            out_path=ranked_path,
            manifest_path=manifest_path,
            api_key=api_key,
            sleep=finnhub_sleep,
        )
    except IncompleteScreenError as e:
        return _terminal(
            notifier, manifest, manifest_path, STATUS_FAILED,
            lambda m: RunFailed(manifest=m, error=str(e), phase=PHASE_SCREEN),
        )
    except EmptyResultError as e:
        return _terminal(
            notifier, manifest, manifest_path, STATUS_DEGRADED,
            lambda m: RunDegraded(manifest=m, reason=str(e)),
        )
    except Exception as e:
        return _terminal(
            notifier, manifest, manifest_path, STATUS_FAILED,
            lambda m: RunFailed(manifest=m, error=f"{type(e).__name__}: {e}", phase=PHASE_RANK),
        )

    # Only here is a re-read required: rank() rewrote the manifest with the
    # ranking-phase counts (after_earnings_exclusion, after_dedupe,
    # final_ranked, finnhub_calls_failed), and the success embed reports
    # them.
    try:
        manifest = load_manifest(manifest_path)
    except (OSError, ValueError) as e:
        print(f"[pipeline] could not re-read manifest after ranking, "
              f"reporting screening-phase counts only: {type(e).__name__}: {e}",
              file=sys.stderr)

    return _terminal(
        notifier, manifest, manifest_path, STATUS_SUCCEEDED,
        lambda m: RunSucceeded(manifest=m, ranked=ranked),
    )


EXIT_CODE_HELP = f"""
exit codes (for the n8n branch):
  {EXIT_SUCCEEDED}  succeeded  ranked shortlist written to screen_ranked.json
  {EXIT_FAILED}  failed     no shortlist; something broke
  {EXIT_DEGRADED}  degraded   no shortlist; the run completed but the result was
                 implausible enough that ranking was refused

degraded is non-zero on purpose: it produces no shortlist, so treating it
as success would republish a stale one.
"""


def main():
    parser = argparse.ArgumentParser(
        description="Run the full put screener: screen, rank, and report the outcome to Discord.",
        epilog=EXIT_CODE_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--constituents", default=str(DEFAULT_CONSTITUENTS),
        help="Path to spy_constituents.json (default: fixtures/spy_constituents.json)",
    )
    parser.add_argument(
        "--out", default=str(DEFAULT_OUT_DIR),
        help="Output directory for all run artifacts (default: ../output)",
    )
    parser.add_argument(
        "--sleep", type=float, default=DEFAULT_SLEEP,
        help=f"Seconds to sleep between tickers during the screen (default: {DEFAULT_SLEEP})",
    )
    parser.add_argument(
        "--finnhub-sleep", type=float, default=DEFAULT_FINNHUB_SLEEP,
        help=f"Seconds to sleep between Finnhub calls during ranking (default: {DEFAULT_FINNHUB_SLEEP})",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only screen the first N constituents (for a quick dry run)",
    )
    parser.add_argument(
        "--save-raw", action="store_true",
        help="Also save each ticker's raw chain JSON to fixtures/",
    )
    parser.add_argument(
        "--no-notify", action="store_true",
        help="Send no Discord alert, and skip the DISCORD_WEBHOOK_URL preflight "
             "check. REQUIRED for the n8n path: in production the webhook lives "
             "in n8n's credential store, not in the environment, and n8n owns "
             "alerting by branching on the exit code. Without this flag a "
             "droplet run fails preflight on a variable that is deliberately "
             "absent. Omit it for local runs, where notify.py is still the way "
             "to see what a run did.",
    )
    args = parser.parse_args()

    save_raw_dir = (Path(__file__).resolve().parent.parent / "fixtures") if args.save_raw else None

    status, _manifest = run_pipeline(
        constituents_path=args.constituents,
        out_dir=args.out,
        sleep=args.sleep,
        limit=args.limit,
        save_raw_dir=save_raw_dir,
        finnhub_sleep=args.finnhub_sleep,
        notifier=NullNotifier() if args.no_notify else None,
    )
    sys.exit(EXIT_CODES[status])


if __name__ == "__main__":
    main()
