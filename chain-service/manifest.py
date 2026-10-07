"""
manifest.py — output/run_manifest.json: one record describing a
screen-then-rank run.

Written in two passes, matching the pipeline's two-script shape:

  1. run_screen.py creates it at the start of a run and checkpoints it
     after every ticker (same cadence as screen_results.json /
     screen_errors.json), filling in the screening-phase fields:
     tickers_total, tickers_screened, tickers_errored,
     qualifying_contracts, duration_seconds.

  2. rank_shortlist.py loads it — this is the sanity gate (see
     check_sanity_gate below) — and, once ranking succeeds, rewrites it
     with the ranking-phase fields: after_earnings_exclusion,
     after_quality_filters, after_dedupe, final_ranked,
     finnhub_calls_failed.

A manifest missing the ranking-phase fields (still None) means run_screen
completed but rank_shortlist has not been run against it yet — that is the
normal state between the two manual steps, not an error.

`status` is the exception to the two-pass shape: neither script sets it
past STATUS_RUNNING. Deciding whether a finished run counts as succeeded,
degraded, or failed is classification, and classification belongs to
run_pipeline.py — the orchestrator that owns the whole screen-then-rank
sequence and can see both phases. Running the two scripts by hand, as
CLAUDE.md's manual order does, therefore leaves the manifest at
STATUS_RUNNING even after a clean rank. That is accurate rather than
broken: nothing in a manual run ever judged the outcome.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from io_utils import atomic_write_json

DEFAULT_MANIFEST_PATH = Path(__file__).resolve().parent.parent / "output" / "run_manifest.json"

# Out of 503 S&P 500 constituents. The real baseline run screened 501; two
# tickers (VMRK, NVR) failed on no-expirations, which is normal churn, not
# a broken run. Below this, something interrupted the screen itself.
MIN_TICKERS_SCREENED = 490

# The manifest's `status` field. A manifest on disk reading STATUS_RUNNING
# with no live process behind it means the run died hard enough that
# nothing got to classify it — which is itself the diagnosis.
STATUS_RUNNING = "running"
STATUS_SUCCEEDED = "succeeded"
STATUS_DEGRADED = "degraded"
STATUS_FAILED = "failed"


class SanityGateError(RuntimeError):
    """Base for the two conditions check_sanity_gate refuses to rank on.

    Subclasses RuntimeError on purpose. Every caller written before these
    types existed catches RuntimeError, and must keep working untouched;
    the subclasses exist only so a caller that needs to tell the two
    conditions APART (run_pipeline.py, deciding between a FAILED and a
    DEGRADED alert) can, without every other caller having to care.
    """


class IncompleteScreenError(SanityGateError):
    """Fewer than MIN_TICKERS_SCREENED tickers made it through the screen.

    The gate only detects this at rank time, but the fault is upstream, in
    the SCREENING phase — run_screen.py was interrupted before it finished
    the index. Anything reporting this condition should name the screen as
    the place to look, not the ranker that noticed it.
    """


class EmptyResultError(SanityGateError):
    """A (near-)complete screen that nonetheless produced zero qualifying
    contracts.

    Distinct from IncompleteScreenError in kind, not just in message: the
    run did what it was told and finished. The result is merely
    implausible — across 490+ real S&P 500 names, a broken filter is a
    likelier explanation than a genuinely empty market — which makes this
    a signal to investigate, not a crash to fix.
    """


def new_manifest(run_id: str, git_commit: Optional[str], tickers_total: int) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "status": STATUS_RUNNING,
        "git_commit": git_commit,
        "tickers_total": tickers_total,
        "tickers_screened": 0,
        "tickers_errored": 0,
        "qualifying_contracts": 0,
        "after_earnings_exclusion": None,
        "after_quality_filters": None,
        "after_dedupe": None,
        "final_ranked": None,
        "finnhub_calls_failed": None,
        "duration_seconds": None,
    }


def load_manifest(path: Path = DEFAULT_MANIFEST_PATH) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def write_manifest(manifest: Dict[str, Any], path: Path = DEFAULT_MANIFEST_PATH) -> None:
    atomic_write_json(path, manifest)


def check_sanity_gate(manifest: Dict[str, Any]) -> None:
    """Raise if the screen this manifest describes should not be ranked.
    Two conditions, either one is fatal:

      - tickers_screened < MIN_TICKERS_SCREENED: the screen itself is
        incomplete (crash, interrupt, kill -9 mid-run). Ranking a partial
        index and presenting it as a full-market shortlist is the failure
        mode this whole manifest exists to prevent. Raises
        IncompleteScreenError.

      - qualifying_contracts == 0 despite a (near-)complete screen: across
        490+ real S&P 500 names, zero qualifying puts is far more likely to
        mean a broken filter (see put_filters.py) than a genuinely empty
        market. The real baseline run found 64. Raises EmptyResultError.

    Both are RuntimeError subclasses, so `except RuntimeError` still
    catches either one.
    """
    screened = manifest["tickers_screened"]
    total = manifest["tickers_total"]
    qualifying = manifest["qualifying_contracts"]

    if screened < MIN_TICKERS_SCREENED:
        raise IncompleteScreenError(
            f"Refusing to rank: only {screened}/{total} tickers were "
            f"screened (minimum {MIN_TICKERS_SCREENED}). This screen is "
            f"incomplete — run_screen.py was likely interrupted before it "
            f"finished. Re-run it to completion before ranking "
            f"(manifest: run_id={manifest.get('run_id')})."
        )

    if qualifying == 0:
        raise EmptyResultError(
            f"Refusing to rank: {screened}/{total} tickers were screened "
            f"successfully but zero qualifying contracts were found. This "
            f"is far more likely to indicate a broken filter than a "
            f"genuinely empty market — investigate put_filters.py before "
            f"ranking (manifest: run_id={manifest.get('run_id')})."
        )
