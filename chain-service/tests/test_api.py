"""The sidecar's job is to expose run_pipeline's three-way outcome over
HTTP without distorting it. So these tests are mostly about the seam:
which HTTP status, which body field, what happens when two runs collide,
and whether progress reaches a poller through run_screen's progress_cb
rather than some second mechanism.

run_pipeline itself is patched out throughout -- it has 233 tests of its
own, and what matters here is what the API does with what it returns.
"""

import threading
import time

import pytest
from fastapi.testclient import TestClient

import api as api_module
from api import STATE_FINISHED, STATE_RUNNING, RunRegistry, app
from manifest import STATUS_DEGRADED, STATUS_FAILED, STATUS_SUCCEEDED, new_manifest


@pytest.fixture(autouse=True)
def fresh_registry(monkeypatch):
    """Each test gets its own registry -- runs are process-local state, so
    leaking one test's run into the next would make ordering matter."""
    monkeypatch.setattr(api_module, "registry", RunRegistry())


@pytest.fixture
def client():
    return TestClient(app)


def healthy_manifest(**overrides):
    m = new_manifest(run_id="2026-08-20T00:00:00+00:00", git_commit="abc1234", tickers_total=503)
    m.update(tickers_screened=501, tickers_errored=2, qualifying_contracts=64)
    m.update(overrides)
    return m


RANKED = [
    {"underlyingSymbol": "AAPL", "strike": 190.0, "delta": -0.08,
     "return_on_capital": 0.012, "bid": 2.30},
]


def fake_pipeline(status_value, manifest=None, on_call=None):
    """A run_pipeline stand-in returning a chosen outcome."""
    def _pipeline(**kwargs):
        if on_call is not None:
            on_call(kwargs)
        return status_value, manifest if manifest is not None else healthy_manifest()
    return _pipeline


def run_to_completion(client, monkeypatch, status_value, ranked=RANKED, manifest=None, on_call=None):
    """POST a run, wait for the worker thread, return the finished record."""
    monkeypatch.setattr(api_module, "run_pipeline",
                        fake_pipeline(status_value, manifest=manifest, on_call=on_call))
    monkeypatch.setattr(api_module, "_load_ranked", lambda out_dir: ranked)

    started = client.post("/runs")
    assert started.status_code == 202
    run_id = started.json()["run_id"]

    for _ in range(200):
        record = client.get(f"/runs/{run_id}").json()
        if record["state"] == STATE_FINISHED:
            return record
        time.sleep(0.01)
    pytest.fail("run did not finish within the timeout")


class TestOutcomeIsInTheBodyNotTheHttpStatus:
    """The core contract. HTTP status says whether the API call worked;
    exit_code says how the screen turned out. Conflating them is what
    makes n8n fire its Error Workflow on top of the Discord alert
    run_pipeline already sent."""

    @pytest.mark.parametrize(
        "status_value, expected_exit",
        [(STATUS_SUCCEEDED, 0), (STATUS_FAILED, 1), (STATUS_DEGRADED, 2)],
    )
    def test_every_outcome_returns_http_200(self, client, monkeypatch, status_value, expected_exit):
        record = run_to_completion(client, monkeypatch, status_value)
        polled = client.get(f"/runs/{record['run_id']}")

        assert polled.status_code == 200, "a failed screen is not a failed API call"
        assert polled.json()["status"] == status_value
        assert polled.json()["exit_code"] == expected_exit

    def test_degraded_does_not_return_a_5xx(self, client, monkeypatch):
        """Pinned on its own because degraded is the tempting one to map
        onto an error status -- and the one where doing so would produce
        two conflicting alerts for a single outcome."""
        record = run_to_completion(client, monkeypatch, STATUS_DEGRADED)
        polled = client.get(f"/runs/{record['run_id']}")

        assert polled.status_code < 400
        assert polled.json()["exit_code"] == 2

    def test_exit_codes_match_the_cli_contract(self, client, monkeypatch):
        """The HTTP contract and the CLI contract are the same three
        numbers, imported from one place so they cannot drift."""
        from run_pipeline import EXIT_CODES

        for status_value in (STATUS_SUCCEEDED, STATUS_FAILED, STATUS_DEGRADED):
            record = run_to_completion(client, monkeypatch, status_value)
            assert record["exit_code"] == EXIT_CODES[status_value]


class TestRunLifecycle:
    def test_post_returns_202_and_a_run_id(self, client, monkeypatch):
        monkeypatch.setattr(api_module, "run_pipeline", fake_pipeline(STATUS_SUCCEEDED))
        monkeypatch.setattr(api_module, "_load_ranked", lambda out_dir: RANKED)

        response = client.post("/runs")

        assert response.status_code == 202, "the run is accepted, not complete"
        body = response.json()
        assert body["run_id"]
        assert body["state"] in (STATE_RUNNING, STATE_FINISHED)
        assert body["exit_code"] is None or body["state"] == STATE_FINISHED

    def test_finished_run_carries_the_shortlist(self, client, monkeypatch):
        record = run_to_completion(client, monkeypatch, STATUS_SUCCEEDED)
        assert record["ranked"] == RANKED

    def test_a_failed_run_carries_no_shortlist(self, client, monkeypatch):
        """exit 1 and exit 2 both mean no shortlist was written. Serving a
        stale one would be worse than serving none."""
        record = run_to_completion(client, monkeypatch, STATUS_FAILED)
        assert record["ranked"] is None

    def test_a_degraded_run_carries_no_shortlist(self, client, monkeypatch):
        record = run_to_completion(client, monkeypatch, STATUS_DEGRADED)
        assert record["ranked"] is None

    def test_finished_run_carries_the_manifest(self, client, monkeypatch):
        record = run_to_completion(client, monkeypatch, STATUS_SUCCEEDED)
        assert record["manifest"]["tickers_screened"] == 501

    def test_timestamps_are_recorded(self, client, monkeypatch):
        record = run_to_completion(client, monkeypatch, STATUS_SUCCEEDED)
        assert record["started_at"] is not None
        assert record["finished_at"] is not None

    def test_current_returns_the_latest_run(self, client, monkeypatch):
        record = run_to_completion(client, monkeypatch, STATUS_SUCCEEDED)
        current = client.get("/runs/current")
        assert current.status_code == 200
        assert current.json()["run_id"] == record["run_id"]

    def test_current_is_404_before_any_run(self, client):
        response = client.get("/runs/current")
        assert response.status_code == 404
        assert "run_manifest.json" in response.json()["detail"]

    def test_unknown_run_id_is_404(self, client):
        assert client.get("/runs/does-not-exist").status_code == 404

    def test_current_is_not_shadowed_by_the_run_id_route(self, client, monkeypatch):
        """/runs/current must not be swallowed as a run_id of 'current'."""
        run_to_completion(client, monkeypatch, STATUS_SUCCEEDED)
        assert client.get("/runs/current").status_code == 200


class TestOneRunAtATime:
    """Two concurrent screens would interleave writes to
    screen_results.json and run_manifest.json. The lock is the only thing
    preventing that, and it only works because the container runs a single
    uvicorn worker."""

    def test_second_run_while_one_is_in_flight_is_409(self, client, monkeypatch):
        release = threading.Event()

        def blocking_pipeline(**kwargs):
            release.wait(timeout=5)
            return STATUS_SUCCEEDED, healthy_manifest()

        monkeypatch.setattr(api_module, "run_pipeline", blocking_pipeline)
        monkeypatch.setattr(api_module, "_load_ranked", lambda out_dir: RANKED)

        first = client.post("/runs")
        assert first.status_code == 202
        try:
            second = client.post("/runs")
            assert second.status_code == 409
            assert second.json()["detail"]["run_id"] == first.json()["run_id"]
        finally:
            release.set()

    def test_409_is_not_a_queue(self, client, monkeypatch):
        """The rejected request must not start a second pipeline later."""
        release = threading.Event()
        calls = []

        def blocking_pipeline(**kwargs):
            calls.append(kwargs)
            release.wait(timeout=5)
            return STATUS_SUCCEEDED, healthy_manifest()

        monkeypatch.setattr(api_module, "run_pipeline", blocking_pipeline)
        monkeypatch.setattr(api_module, "_load_ranked", lambda out_dir: RANKED)

        client.post("/runs")
        client.post("/runs")
        release.set()
        time.sleep(0.2)

        assert len(calls) == 1

    def test_a_new_run_is_allowed_once_the_previous_finished(self, client, monkeypatch):
        run_to_completion(client, monkeypatch, STATUS_SUCCEEDED)
        assert client.post("/runs").status_code == 202


class TestProgressComesFromTheHook:
    """run_screen's progress_cb, forwarded by run_pipeline, is the only
    progress mechanism. Nothing here tails stdout or stats files."""

    def test_progress_cb_is_passed_to_run_pipeline(self, client, monkeypatch):
        seen = {}
        run_to_completion(client, monkeypatch, STATUS_SUCCEEDED, on_call=seen.update)
        assert callable(seen["progress_cb"])

    def test_progress_is_visible_to_a_poller_mid_run(self, client, monkeypatch):
        """The point of the hook: a caller polling /runs/{id} during a
        ten-minute screen can see how far along it is."""
        reported = threading.Event()
        release = threading.Event()

        def pipeline_with_progress(**kwargs):
            kwargs["progress_cb"](137, 503, "MSFT")
            reported.set()
            release.wait(timeout=5)
            return STATUS_SUCCEEDED, healthy_manifest()

        monkeypatch.setattr(api_module, "run_pipeline", pipeline_with_progress)
        monkeypatch.setattr(api_module, "_load_ranked", lambda out_dir: RANKED)

        run_id = client.post("/runs").json()["run_id"]
        assert reported.wait(timeout=5)
        try:
            progress = client.get(f"/runs/{run_id}").json()["progress"]
            assert progress["done"] == 137
            assert progress["total"] == 503
            assert progress["ticker"] == "MSFT"
            assert progress["percent"] == 27.2
        finally:
            release.set()

    def test_percent_is_none_when_total_is_zero(self):
        """An empty constituents file must not divide by zero on the
        worker thread, where the traceback would go nowhere useful."""
        registry = RunRegistry()
        registry.start("r1")
        registry.update_progress("r1", 0, 0, "?")
        assert registry.get("r1")["progress"]["percent"] is None

    def test_progress_snapshot_does_not_mutate_under_a_reader(self):
        registry = RunRegistry()
        registry.start("r1")
        registry.update_progress("r1", 1, 503, "AAPL")
        snapshot = registry.get("r1")
        registry.update_progress("r1", 2, 503, "MSFT")
        assert snapshot["progress"]["done"] == 1


class TestRunParameters:
    def test_defaults_match_the_cli(self, client, monkeypatch):
        from rank_shortlist import DEFAULT_SLEEP as FINNHUB_SLEEP
        from run_screen import DEFAULT_SLEEP as SCREEN_SLEEP

        seen = {}
        run_to_completion(client, monkeypatch, STATUS_SUCCEEDED, on_call=seen.update)

        assert seen["sleep"] == SCREEN_SLEEP
        assert seen["finnhub_sleep"] == FINNHUB_SLEEP
        assert seen["limit"] is None

    def test_overrides_are_forwarded(self, client, monkeypatch):
        seen = {}
        monkeypatch.setattr(api_module, "run_pipeline",
                            fake_pipeline(STATUS_SUCCEEDED, on_call=seen.update))
        monkeypatch.setattr(api_module, "_load_ranked", lambda out_dir: RANKED)

        client.post("/runs", json={"limit": 5, "sleep": 0, "finnhub_sleep": 0.5})
        for _ in range(200):
            if seen:
                break
            time.sleep(0.01)

        assert seen["limit"] == 5
        assert seen["sleep"] == 0
        assert seen["finnhub_sleep"] == 0.5

    def test_no_body_is_accepted(self, client, monkeypatch):
        monkeypatch.setattr(api_module, "run_pipeline", fake_pipeline(STATUS_SUCCEEDED))
        monkeypatch.setattr(api_module, "_load_ranked", lambda out_dir: RANKED)
        assert client.post("/runs").status_code == 202


class TestWorkerThreadBackstop:
    """run_pipeline never raises by contract. If that contract is ever
    broken, a run stuck at state='running' would hang an n8n poll until
    its own timeout with no explanation -- so the thread has a backstop."""

    def test_an_exception_finishes_the_run_as_failed(self, client, monkeypatch):
        def exploding_pipeline(**kwargs):
            raise RuntimeError("contract violated")

        monkeypatch.setattr(api_module, "run_pipeline", exploding_pipeline)

        run_id = client.post("/runs").json()["run_id"]
        for _ in range(200):
            record = client.get(f"/runs/{run_id}").json()
            if record["state"] == STATE_FINISHED:
                break
            time.sleep(0.01)

        assert record["state"] == STATE_FINISHED
        assert record["status"] == STATUS_FAILED
        assert record["exit_code"] == 1
        assert "RuntimeError" in record["error"]

    def test_the_lock_is_released_after_a_crash(self, client, monkeypatch):
        """A crashed run must not wedge the sidecar into permanent 409s."""
        def exploding_pipeline(**kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(api_module, "run_pipeline", exploding_pipeline)
        run_id = client.post("/runs").json()["run_id"]
        for _ in range(200):
            if client.get(f"/runs/{run_id}").json()["state"] == STATE_FINISHED:
                break
            time.sleep(0.01)

        monkeypatch.setattr(api_module, "run_pipeline", fake_pipeline(STATUS_SUCCEEDED))
        monkeypatch.setattr(api_module, "_load_ranked", lambda out_dir: RANKED)
        assert client.post("/runs").status_code == 202


class TestSecretsNeverCrossTheApi:
    def test_preflight_reports_booleans_not_values(self, client, monkeypatch):
        monkeypatch.setenv("FINNHUB_API_KEY", "super-secret-key-value")
        monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/1/tok")

        body = client.get("/preflight").json()
        serialized = str(body)

        assert body["ready"] is True
        assert body["finnhub_api_key_set"] is True
        assert body["discord_webhook_url_set"] is True
        assert "super-secret-key-value" not in serialized
        assert "tok" not in serialized

    def test_preflight_reports_what_is_missing(self, client, monkeypatch):
        monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
        monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)

        body = client.get("/preflight").json()

        assert body["ready"] is False
        assert body["finnhub_api_key_set"] is False
        assert body["discord_webhook_url_set"] is False
        # One problem, not two: the missing webhook is not a problem for
        # the sidecar. See TestAlertingBelongsToN8n below.
        assert len(body["problems"]) == 1
        assert "FINNHUB_API_KEY" in body["problems"][0]

    def test_run_errors_are_redacted(self, client, monkeypatch):
        """Errors reaching a response go through notify.redact(), the same
        pass that guards the Discord payloads."""
        monkeypatch.setenv("FINNHUB_API_KEY", "leaky-key-abc123")

        def exploding_pipeline(**kwargs):
            raise RuntimeError("failed with key leaky-key-abc123 at /Users/daniel/secret/path.py")

        monkeypatch.setattr(api_module, "run_pipeline", exploding_pipeline)
        run_id = client.post("/runs").json()["run_id"]
        for _ in range(200):
            record = client.get(f"/runs/{run_id}").json()
            if record["state"] == STATE_FINISHED:
                break
            time.sleep(0.01)

        assert "leaky-key-abc123" not in record["error"]
        assert "/Users/daniel" not in record["error"]


class TestHealth:
    def test_health_is_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_health_does_not_depend_on_secrets(self, client, monkeypatch):
        """A healthcheck that failed on a missing screener.env would
        restart-loop the container instead of letting /preflight say
        what is wrong."""
        monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
        monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
        assert client.get("/health").status_code == 200

    def test_health_leaks_no_secrets(self, client, monkeypatch):
        monkeypatch.setenv("FINNHUB_API_KEY", "secret-value-xyz")
        assert "secret-value-xyz" not in str(client.get("/health").json())


class TestAlertingBelongsToN8n:
    """In production the Discord webhook lives in n8n's credential store,
    not in the environment: n8n alerts by branching on exit_code, and the
    pipeline stays quiet. The sidecar IS the n8n path, so it never posts
    to Discord -- and because injecting a notifier is also what switches
    off run_pipeline's DISCORD_WEBHOOK_URL preflight check, that injection
    is what lets a run work at all on a droplet where the variable is
    deliberately absent."""

    def test_the_sidecar_injects_a_null_notifier(self, client, monkeypatch):
        from run_pipeline import NullNotifier

        seen = {}
        run_to_completion(client, monkeypatch, STATUS_SUCCEEDED, on_call=seen.update)

        assert isinstance(seen["notifier"], NullNotifier), (
            "the sidecar must not post to Discord -- n8n owns alerting"
        )

    def test_a_run_works_with_no_webhook_in_the_environment(self, client, monkeypatch):
        """The droplet's actual configuration: FINNHUB_API_KEY only."""
        monkeypatch.setenv("FINNHUB_API_KEY", "key-from-screener-env")
        monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)

        record = run_to_completion(client, monkeypatch, STATUS_SUCCEEDED)

        assert record["state"] == STATE_FINISHED
        assert record["exit_code"] == 0

    def test_preflight_is_ready_with_the_api_key_alone(self, client, monkeypatch):
        """The regression this guards: reporting ready: false over a webhook
        that is deliberately absent would flag a correctly configured
        droplet as broken."""
        monkeypatch.setenv("FINNHUB_API_KEY", "key-from-screener-env")
        monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)

        body = client.get("/preflight").json()

        assert body["ready"] is True
        assert body["finnhub_api_key_set"] is True
        assert body["discord_webhook_url_set"] is False
        assert body["problems"] == []

    def test_the_webhook_is_reported_but_does_not_gate_readiness(self, client, monkeypatch):
        """discord_webhook_url_set stays in the response as information --
        useful for a local shell -- but it is not a verdict."""
        monkeypatch.setenv("FINNHUB_API_KEY", "key")
        monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
        without = client.get("/preflight").json()

        monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/1/tok")
        with_hook = client.get("/preflight").json()

        assert without["ready"] == with_hook["ready"] is True
        assert without["discord_webhook_url_set"] is False
        assert with_hook["discord_webhook_url_set"] is True

    def test_no_discord_problem_ever_reaches_the_api(self, client, monkeypatch):
        monkeypatch.setenv("FINNHUB_API_KEY", "key")
        monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)

        problems = client.get("/preflight").json()["problems"]

        assert not any("DISCORD_WEBHOOK_URL" in p for p in problems)
