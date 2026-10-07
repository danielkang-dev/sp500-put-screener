#!/usr/bin/env python3
"""
api.py — FastAPI sidecar wrapping run_pipeline.py for n8n.

WHY A SIDECAR AT ALL: a full screen is ~10 minutes and 500+ Yahoo
requests. Nothing should hold an HTTP connection open that long — not
n8n's HTTP Request node, not a proxy, not a container healthcheck. So
this is a job API, not a request/response wrapper:

    POST /runs            starts a run, returns 202 + a run_id immediately
    GET  /runs/{run_id}   poll until state == "finished"
    then branch on exit_code

HTTP STATUS DOES NOT CARRY THE PIPELINE OUTCOME. This is the single most
important decision in this file, so it is written down rather than left
to be rediscovered:

    HTTP status answers  "did this API call work?"
    body.exit_code answers "how did the screen turn out?"

A degraded run returns HTTP 200 with exit_code 2 in the body — NOT a 5xx.
Returning 5xx would make n8n's HTTP Request node mark itself failed and
fire the Error Workflow, on top of the RunDegraded Discord message
run_pipeline.py already sent: one outcome, two conflicting alerts. That is
exactly the failure mode the Execute Command node has to be set to
continue-on-fail to avoid (see CLAUDE.md's Phase 9 wiring note), and it
would be perverse to reintroduce it here through the back door.

The payoff is that the n8n wiring has the same SHAPE either way. Whether
the workflow shells out to run_pipeline.py and switches on `exitCode`, or
calls this API and switches on `$json.exit_code`, it is the same Switch
node over the same three values:

    0  succeeded   shortlist in body.ranked
    1  failed      no shortlist; something broke
    2  degraded    no shortlist; run completed, result implausible

ONE RUN AT A TIME. Two concurrent screens would both write
output/screen_results.json and both stamp output/run_manifest.json,
interleaved and unusable. A second POST /runs while one is in flight gets
409, not a queue — n8n should not be starting a second screen while the
first is still going, and silently queueing would hide that it tried.

This depends on the registry below being process-local state, which is
why the container must run uvicorn with EXACTLY ONE worker. Two workers
means two registries, two locks, and no mutual exclusion at all. The
Dockerfile pins --workers 1 and says so.

PROGRESS comes from run_screen()'s progress_cb hook, forwarded by
run_pipeline(). That is the only progress mechanism in the pipeline —
nothing here tails stdout or polls files to infer how far along a run is.

ALERTING IS n8n's JOB, NOT THIS PROCESS'S. Every run here injects
NullNotifier, so the sidecar never posts to Discord: n8n holds the webhook
in its credential store and alerts by branching on exit_code. Two alerts
for one run would train whoever reads them to ignore both. Injecting a
notifier is also what switches off run_pipeline's DISCORD_WEBHOOK_URL
preflight check — necessary here, since that variable is deliberately
absent on the droplet. The only secret the sidecar needs is
FINNHUB_API_KEY.

SECRETS never cross this API. /preflight reports whether each variable is
set as a boolean and never its value, and every error string that reaches
a response goes through notify.redact() first — the same redaction that
guards the Discord payloads, applied for the same reason.

STATE IS IN MEMORY. A container restart loses the run history here;
output/run_manifest.json on the mounted volume remains the durable record
of what happened, and a run killed by a restart stays status="running"
there, which is accurate — nothing classified it.

Places no trades. It starts a screen and reports what the screen found.
"""

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Response, status as http_status
from pydantic import BaseModel, Field

from manifest import MIN_TICKERS_SCREENED, STATUS_FAILED
from notify import redact
from rank_shortlist import DEFAULT_SLEEP as DEFAULT_FINNHUB_SLEEP
from run_pipeline import EXIT_CODES, EXIT_FAILED, NullNotifier, preflight, run_pipeline
from run_screen import DEFAULT_CONSTITUENTS, DEFAULT_OUT_DIR, DEFAULT_SLEEP

STATE_RUNNING = "running"
STATE_FINISHED = "finished"

app = FastAPI(
    title="put-screener sidecar",
    description=(
        "Starts and reports on put-screener runs. Poll a run until "
        "state == 'finished', then branch on exit_code: 0 succeeded, "
        "1 failed, 2 degraded. HTTP status reports whether the API call "
        "worked, never how the screen turned out."
    ),
    version="8b",
)


class RunRequest(BaseModel):
    """Optional overrides for one run. All default to the same values
    run_pipeline.py's CLI uses."""

    limit: Optional[int] = Field(
        default=None,
        description=(
            "Screen only the first N constituents. For exercising the "
            f"plumbing, not for real runs: any limit below "
            f"{MIN_TICKERS_SCREENED} trips the sanity gate and the run ends "
            "failed by design, because a partial index is exactly what that "
            "gate exists to refuse."
        ),
    )
    sleep: float = Field(
        default=DEFAULT_SLEEP,
        description="Seconds between tickers during the screen.",
    )
    finnhub_sleep: float = Field(
        default=DEFAULT_FINNHUB_SLEEP,
        description="Seconds between Finnhub calls during ranking.",
    )


class RunRegistry:
    """Process-local record of runs, and the lock that keeps there being
    only one at a time.

    Every read returns a deep-ish copy: a caller serializing a run record
    must not see it mutate underneath them as progress_cb fires on the
    worker thread.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._runs: Dict[str, Dict[str, Any]] = {}
        self._current_id: Optional[str] = None

    def start(self, run_id: str) -> None:
        with self._lock:
            self._runs[run_id] = {
                "run_id": run_id,
                "state": STATE_RUNNING,
                "status": None,
                "exit_code": None,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "finished_at": None,
                "progress": {"done": 0, "total": None, "ticker": None, "percent": None},
                "manifest": None,
                "ranked": None,
                "error": None,
            }
            self._current_id = run_id

    def is_running(self) -> bool:
        with self._lock:
            if self._current_id is None:
                return False
            return self._runs[self._current_id]["state"] == STATE_RUNNING

    def update_progress(self, run_id: str, done: int, total: int, ticker: str) -> None:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return
            run["progress"] = {
                "done": done,
                "total": total,
                "ticker": ticker,
                "percent": round(done / total * 100, 1) if total else None,
            }

    def finish(self, run_id: str, status_value: str, manifest: Optional[Dict[str, Any]],
               ranked: Optional[List[Dict[str, Any]]], error: Optional[str] = None) -> None:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return
            run["state"] = STATE_FINISHED
            run["status"] = status_value
            # EXIT_CODES is the same mapping run_pipeline.main() exits with,
            # imported rather than restated so the HTTP contract and the CLI
            # contract cannot drift apart.
            run["exit_code"] = EXIT_CODES.get(status_value, EXIT_FAILED)
            run["finished_at"] = datetime.now(timezone.utc).isoformat()
            run["manifest"] = manifest
            run["ranked"] = ranked
            run["error"] = redact(error) if error else None

    def get(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            run = self._runs.get(run_id)
            return _copy_run(run) if run else None

    def current(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            if self._current_id is None:
                return None
            return _copy_run(self._runs[self._current_id])


def _copy_run(run: Dict[str, Any]) -> Dict[str, Any]:
    """Snapshot a run record. `progress` is copied explicitly because it is
    the one field the worker thread rewrites while a caller may be reading
    it; manifest/ranked are only ever set once, at finish."""
    snapshot = dict(run)
    snapshot["progress"] = dict(run["progress"])
    return snapshot


registry = RunRegistry()


def _execute(run_id: str, request: RunRequest, out_dir: Path, constituents_path: Path) -> None:
    """The worker thread body: run the pipeline, record how it ended.

    run_pipeline() never raises by contract, so the except branch here is
    not expected to fire. It exists because a worker thread that dies
    silently leaves the run stuck at state="running" forever, and n8n
    polling it would hang until its own timeout with no explanation. A
    contract this depends on is worth a backstop.
    """
    try:
        status_value, manifest = run_pipeline(
            constituents_path=constituents_path,
            out_dir=out_dir,
            sleep=request.sleep,
            limit=request.limit,
            finnhub_sleep=request.finnhub_sleep,
            # The sidecar never posts to Discord. It is the n8n path by
            # definition, and n8n owns alerting there — it holds the webhook
            # in its credential store and branches on exit_code. A pipeline
            # that also alerted would double-report every run.
            #
            # This is not merely cosmetic: injecting a notifier is also what
            # switches off run_pipeline's DISCORD_WEBHOOK_URL preflight check.
            # Without it, every run on the droplet would fail preflight on a
            # variable that is deliberately absent there.
            notifier=NullNotifier(),
            progress_cb=lambda done, total, ticker: registry.update_progress(
                run_id, done, total, ticker
            ),
        )
    except BaseException as e:  # noqa: BLE001 - see docstring
        registry.finish(
            run_id, STATUS_FAILED, manifest=None, ranked=None,
            error=f"{type(e).__name__}: {e}",
        )
        return

    ranked = _load_ranked(out_dir) if EXIT_CODES.get(status_value) == 0 else None
    registry.finish(run_id, status_value, manifest=manifest, ranked=ranked)


def _load_ranked(out_dir: Path) -> Optional[List[Dict[str, Any]]]:
    """The shortlist, read back from the file rank_shortlist.py just wrote.

    Read from disk rather than plumbed back through run_pipeline's return
    value on purpose: the file is the deliverable, and serving anything
    else would let the API report a shortlist that does not match what is
    actually on the volume n8n reads.
    """
    try:
        with open(Path(out_dir) / "screen_ranked.json") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


@app.get("/health")
def health() -> Dict[str, Any]:
    """Liveness for the container healthcheck. Deliberately says nothing
    about secrets or run state — a healthcheck that fails because
    screener.env is missing would restart-loop the container instead of
    letting /preflight report the problem."""
    return {"status": "ok", "service": "put-screener-sidecar"}


@app.get("/preflight")
def preflight_check() -> Dict[str, Any]:
    """Whether the environment is configured, as booleans only.

    Never returns a secret's value — only whether it is set. This is the
    endpoint to hit after wiring screener.env in Phase 9, so a missing
    variable is found in one curl rather than at the end of a ten-minute
    screen.

    check_webhook=False, matching what the runs themselves do: the sidecar
    injects NullNotifier and never posts to Discord, because n8n owns
    alerting and holds the webhook in its own credential store. Readiness
    here therefore turns on FINNHUB_API_KEY alone. Reporting ready: false
    over a webhook that is deliberately absent would flag a correctly
    configured droplet as broken.

    discord_webhook_url_set is still reported, as information rather than
    a verdict — it is worth being able to see whether a local shell has
    one — but it does not gate `ready`.
    """
    _api_key, problems = preflight(check_webhook=False)
    return {
        "ready": not problems,
        "finnhub_api_key_set": bool(os.environ.get("FINNHUB_API_KEY")),
        "discord_webhook_url_set": bool(os.environ.get("DISCORD_WEBHOOK_URL")),
        "problems": [redact(p) for p in problems],
    }


@app.post("/runs", status_code=http_status.HTTP_202_ACCEPTED)
def start_run(request: Optional[RunRequest] = None) -> Dict[str, Any]:
    """Start a run and return immediately with its run_id.

    202, not 200: the run has been accepted and is not finished. Poll
    GET /runs/{run_id} until state == "finished".

    409 if a run is already in flight. Not a queue — n8n should not be
    starting a second screen while the first is still going, and quietly
    queueing it would hide that it tried.
    """
    request = request or RunRequest()

    if registry.is_running():
        current = registry.current()
        raise HTTPException(
            status_code=http_status.HTTP_409_CONFLICT,
            detail={
                "error": "a run is already in progress",
                "run_id": current["run_id"] if current else None,
                "progress": current["progress"] if current else None,
            },
        )

    run_id = uuid.uuid4().hex
    registry.start(run_id)

    thread = threading.Thread(
        target=_execute,
        args=(run_id, request, Path(DEFAULT_OUT_DIR), Path(DEFAULT_CONSTITUENTS)),
        name=f"screen-{run_id[:8]}",
        daemon=True,
    )
    thread.start()

    return registry.get(run_id)


@app.get("/runs/current")
def get_current_run(response: Response) -> Dict[str, Any]:
    """The most recent run in THIS process.

    404 when nothing has run since the container started — including after
    a restart that interrupted a run. output/run_manifest.json is the
    durable record in that case; a run killed by a restart stays
    status="running" there, which is accurate, since nothing classified it.
    """
    run = registry.current()
    if run is None:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail="no run has been started in this process; see output/run_manifest.json",
        )
    return run


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> Dict[str, Any]:
    """One run by id.

    Always 200 for a run that exists, whatever its outcome. exit_code in
    the body carries the outcome — see this module's docstring for why
    that is not an HTTP status.
    """
    run = registry.get(run_id)
    if run is None:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=f"no run with id {run_id} in this process",
        )
    return run
