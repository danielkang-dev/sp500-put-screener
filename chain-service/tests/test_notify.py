import json

import pytest

from notify import (
    COLOR_AMBER,
    COLOR_GREEN,
    COLOR_RED,
    MAX_CANDIDATES_SHOWN,
    DiscordNotifier,
    RunDegraded,
    RunFailed,
    RunSucceeded,
    notify,
    redact,
)


def healthy_manifest(**overrides):
    m = {
        "run_id": "2026-08-20T00:00:00+00:00",
        "git_commit": "abc1234",
        "tickers_total": 503,
        "tickers_screened": 501,
        "tickers_errored": 2,
        "qualifying_contracts": 64,
        "after_earnings_exclusion": 55,
        "after_dedupe": 27,
        "final_ranked": 18,
        "finnhub_calls_failed": 0,
        "duration_seconds": 512.3,
    }
    m.update(overrides)
    return m


def a_candidate(symbol="AAPL", strike=150.0, delta=-0.12, roc=0.018, bid=1.5):
    return {
        "underlyingSymbol": symbol, "strike": strike, "delta": delta,
        "return_on_capital": roc, "bid": bid, "contractSymbol": f"{symbol}T",
    }


class TestRedactSecrets:
    def test_redacts_finnhub_key_from_env(self, monkeypatch):
        monkeypatch.setenv("FINNHUB_API_KEY", "sekret-finnhub-value")
        assert "sekret-finnhub-value" not in redact("error calling with sekret-finnhub-value")

    def test_redacts_discord_webhook_from_env(self, monkeypatch):
        url = "https://discord.com/api/webhooks/123456789/abcDEF-token_here"
        monkeypatch.setenv("DISCORD_WEBHOOK_URL", url)
        assert url not in redact(f"posting to {url} failed")

    def test_redacts_discord_webhook_by_shape_even_without_env(self, monkeypatch):
        """Defense in depth: matched by pattern even if it doesn't equal
        whatever's currently in DISCORD_WEBHOOK_URL (e.g. a stale or
        different webhook string embedded in an error message)."""
        monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
        url = "https://discord.com/api/webhooks/999/someOtherToken"
        assert url not in redact(f"tried {url}")
        assert "[REDACTED_WEBHOOK]" in redact(f"tried {url}")

    @pytest.mark.parametrize("pattern", ["token=abc123", "api_key=abc123", "apikey=abc123", "key=abc123"])
    def test_redacts_key_value_secret_patterns(self, pattern):
        result = redact(f"GET /calendar?{pattern}&symbol=AAPL")
        assert "abc123" not in result
        assert "[REDACTED]" in result

    def test_key_value_redaction_is_case_insensitive(self):
        assert "SECRET999" not in redact("TOKEN=SECRET999")

    def test_does_not_redact_unrelated_query_params(self):
        result = redact("GET /calendar?symbol=AAPL&from=2026-08-20")
        assert "symbol=AAPL" in result
        assert "from=2026-08-20" in result


class TestRedactPaths:
    @pytest.mark.parametrize("prefix", ["/Users/daniel", "/home/daniel", "/root", "/tmp/x", "/app", "/private/tmp/x"])
    def test_collapses_local_paths_to_basename(self, prefix):
        path = f"{prefix}/put-screener/output/screen_results.json"
        result = redact(f"wrote to {path}")
        assert path not in result
        assert "screen_results.json" in result

    def test_does_not_touch_relative_or_non_system_paths(self):
        result = redact("see output/screen_results.json for details")
        assert result == "see output/screen_results.json for details"

    def test_leaves_plain_text_completely_unchanged(self):
        text = "501/503 tickers screened, 64 qualifying contracts found"
        assert redact(text) == text


class TestRedactEdgeCases:
    def test_handles_none(self):
        assert redact(None) is None

    def test_handles_empty_string(self):
        assert redact("") == ""

    def test_is_idempotent(self, monkeypatch):
        monkeypatch.setenv("FINNHUB_API_KEY", "sekret")
        text = "error with sekret at /Users/daniel/x/screen_results.json"
        once = redact(text)
        twice = redact(once)
        assert once == twice


class _FakeResponse:
    def __init__(self, status_code=204):
        self.status_code = status_code


class _CapturingPost:
    """Stand-in for requests.post that records what it was called with,
    instead of a MagicMock, so assertions read as plain data access."""

    def __init__(self, status_code=204, raise_exc=None):
        self.calls = []
        self._status_code = status_code
        self._raise_exc = raise_exc

    def __call__(self, url, json=None, timeout=None):
        if self._raise_exc:
            raise self._raise_exc
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        return _FakeResponse(self._status_code)


class TestDiscordNotifierNeverRaises:
    def test_missing_webhook_url_does_not_raise_or_call_post(self, monkeypatch):
        monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
        post = _CapturingPost()
        DiscordNotifier(post_fn=post).send(RunSucceeded(manifest=healthy_manifest()))
        assert post.calls == []

    def test_network_error_does_not_raise(self):
        post = _CapturingPost(raise_exc=ConnectionError("connection refused"))
        notifier = DiscordNotifier(webhook_url="https://discord.com/api/webhooks/1/x", post_fn=post)
        notifier.send(RunSucceeded(manifest=healthy_manifest()))  # must not raise

    def test_timeout_does_not_raise(self):
        import requests as _rq
        post = _CapturingPost(raise_exc=_rq.exceptions.Timeout("timed out"))
        notifier = DiscordNotifier(webhook_url="https://discord.com/api/webhooks/1/x", post_fn=post)
        notifier.send(RunFailed(manifest=healthy_manifest(), error="boom", phase="screen"))

    def test_non_2xx_response_does_not_raise(self):
        post = _CapturingPost(status_code=429)
        notifier = DiscordNotifier(webhook_url="https://discord.com/api/webhooks/1/x", post_fn=post)
        notifier.send(RunSucceeded(manifest=healthy_manifest()))  # must not raise

    def test_unrecognized_event_type_does_not_raise(self):
        post = _CapturingPost()
        notifier = DiscordNotifier(webhook_url="https://discord.com/api/webhooks/1/x", post_fn=post)
        notifier.send(object())  # not one of the three event types
        assert post.calls == []  # never got far enough to send anything

    def test_malformed_manifest_does_not_raise(self):
        """A manifest missing expected keys (e.g. read from a corrupt or
        partial run_manifest.json) must degrade gracefully, not crash the
        notifier -- the whole point of alerting is to survive things going
        wrong elsewhere in the pipeline."""
        post = _CapturingPost()
        notifier = DiscordNotifier(webhook_url="https://discord.com/api/webhooks/1/x", post_fn=post)
        notifier.send(RunSucceeded(manifest={}))  # empty manifest
        assert len(post.calls) == 1  # still sends, just with "?" placeholders

    def test_convenience_function_never_raises_without_webhook(self, monkeypatch):
        monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
        notify(RunSucceeded(manifest=healthy_manifest()))  # must not raise


class TestPayloadRedaction:
    """The property that actually matters: a secret or local path present
    anywhere in event data must not survive into what gets POSTed."""

    def test_secret_in_error_message_is_redacted_before_sending(self, monkeypatch):
        monkeypatch.setenv("FINNHUB_API_KEY", "top-secret-key-value")
        post = _CapturingPost()
        notifier = DiscordNotifier(webhook_url="https://discord.com/api/webhooks/1/x", post_fn=post)

        notifier.send(RunFailed(
            manifest=healthy_manifest(),
            error="Finnhub call failed with token=top-secret-key-value in the URL",
            phase="rank",
        ))

        sent = json.dumps(post.calls[0]["json"])
        assert "top-secret-key-value" not in sent

    def test_local_path_in_error_message_is_redacted_before_sending(self):
        post = _CapturingPost()
        notifier = DiscordNotifier(webhook_url="https://discord.com/api/webhooks/1/x", post_fn=post)

        notifier.send(RunFailed(
            manifest=healthy_manifest(),
            error="FileNotFoundError: /Users/daniel/Claude/Projects/put-screener/output/screen_results.json",
            phase="rank",
        ))

        sent = json.dumps(post.calls[0]["json"])
        assert "/Users/daniel" not in sent
        assert "screen_results.json" in sent  # filename itself is fine to show

    def test_configured_webhook_url_never_appears_in_its_own_payload(self, monkeypatch):
        """Defense in depth: even if the webhook URL somehow ends up inside
        event data (e.g. echoed back in an error message), it must not be
        visible in what gets sent back out through that same webhook."""
        webhook = "https://discord.com/api/webhooks/555/superSecretPath"
        monkeypatch.setenv("DISCORD_WEBHOOK_URL", webhook)
        post = _CapturingPost()
        notifier = DiscordNotifier(webhook_url=webhook, post_fn=post)

        notifier.send(RunFailed(
            manifest=healthy_manifest(),
            error=f"could not reach {webhook}",
            phase="screen",
        ))

        sent = json.dumps(post.calls[0]["json"])
        assert webhook not in sent

    def test_send_failure_log_message_is_also_redacted(self, monkeypatch, capsys):
        """Even the stderr line printed when sending itself fails must not
        leak the webhook URL from the exception text requests can raise."""
        webhook = "https://discord.com/api/webhooks/1/secretpath"
        post = _CapturingPost(raise_exc=ConnectionError(f"Failed to reach {webhook}"))
        notifier = DiscordNotifier(webhook_url=webhook, post_fn=post)

        notifier.send(RunSucceeded(manifest=healthy_manifest()))

        stderr = capsys.readouterr().err
        assert webhook not in stderr


class TestEmbedContent:
    def test_failed_embed_is_red_and_has_the_right_shape(self):
        post = _CapturingPost()
        DiscordNotifier(webhook_url="https://discord.com/api/webhooks/1/x", post_fn=post).send(
            RunFailed(manifest=healthy_manifest(), error="boom", phase="screen")
        )
        embed = post.calls[0]["json"]["embeds"][0]
        assert embed["color"] == COLOR_RED
        assert "FAILED" in embed["title"]
        assert embed["description"] == "boom"
        assert any(f["name"] == "Phase" and f["value"] == "screen" for f in embed["fields"])

    def test_degraded_embed_is_amber(self):
        post = _CapturingPost()
        DiscordNotifier(webhook_url="https://discord.com/api/webhooks/1/x", post_fn=post).send(
            RunDegraded(manifest=healthy_manifest(), reason="zero qualifying contracts")
        )
        embed = post.calls[0]["json"]["embeds"][0]
        assert embed["color"] == COLOR_AMBER
        assert "DEGRADED" in embed["title"]
        assert embed["description"] == "zero qualifying contracts"

    def test_succeeded_embed_is_green_with_summary_counts(self):
        post = _CapturingPost()
        DiscordNotifier(webhook_url="https://discord.com/api/webhooks/1/x", post_fn=post).send(
            RunSucceeded(manifest=healthy_manifest(), ranked=[a_candidate()])
        )
        embed = post.calls[0]["json"]["embeds"][0]
        assert embed["color"] == COLOR_GREEN
        field_values = {f["name"]: f["value"] for f in embed["fields"]}
        assert field_values["Qualifying contracts"] == "64"
        assert field_values["Final ranked"] == "18"

    def test_footer_identifies_the_emitter(self):
        post = _CapturingPost()
        DiscordNotifier(webhook_url="https://discord.com/api/webhooks/1/x", post_fn=post).send(
            RunSucceeded(manifest=healthy_manifest())
        )
        embed = post.calls[0]["json"]["embeds"][0]
        assert "screener" in embed["footer"]["text"].lower()

    def test_timestamp_comes_from_manifest_run_id(self):
        post = _CapturingPost()
        DiscordNotifier(webhook_url="https://discord.com/api/webhooks/1/x", post_fn=post).send(
            RunSucceeded(manifest=healthy_manifest(run_id="2026-08-20T12:34:56+00:00"))
        )
        embed = post.calls[0]["json"]["embeds"][0]
        assert embed["timestamp"] == "2026-08-20T12:34:56+00:00"


class TestCandidateFormatting:
    def test_shows_up_to_five_candidates(self):
        post = _CapturingPost()
        candidates = [a_candidate(symbol=f"T{i}") for i in range(3)]
        DiscordNotifier(webhook_url="https://discord.com/api/webhooks/1/x", post_fn=post).send(
            RunSucceeded(manifest=healthy_manifest(), ranked=candidates)
        )
        description = post.calls[0]["json"]["embeds"][0]["description"]
        for c in candidates:
            assert c["underlyingSymbol"] in description
        assert "more in" not in description

    def test_caps_at_five_with_an_overflow_note(self):
        post = _CapturingPost()
        candidates = [a_candidate(symbol=f"T{i}") for i in range(18)]  # matches real baseline shape
        DiscordNotifier(webhook_url="https://discord.com/api/webhooks/1/x", post_fn=post).send(
            RunSucceeded(manifest=healthy_manifest(), ranked=candidates)
        )
        description = post.calls[0]["json"]["embeds"][0]["description"]
        shown = [c["underlyingSymbol"] for c in candidates[:MAX_CANDIDATES_SHOWN]]
        hidden = [c["underlyingSymbol"] for c in candidates[MAX_CANDIDATES_SHOWN:]]
        for symbol in shown:
            assert symbol in description
        for symbol in hidden:
            assert symbol not in description
        assert "...and 13 more in screen_ranked.json" in description

    def test_empty_ranked_list_has_a_clear_message_not_a_blank_embed(self):
        post = _CapturingPost()
        DiscordNotifier(webhook_url="https://discord.com/api/webhooks/1/x", post_fn=post).send(
            RunSucceeded(manifest=healthy_manifest(final_ranked=0), ranked=[])
        )
        description = post.calls[0]["json"]["embeds"][0]["description"]
        assert "no qualifying" in description.lower()

    def test_candidate_line_includes_strike_delta_roc_and_bid(self):
        post = _CapturingPost()
        DiscordNotifier(webhook_url="https://discord.com/api/webhooks/1/x", post_fn=post).send(
            RunSucceeded(manifest=healthy_manifest(), ranked=[a_candidate("AAPL", 150.0, -0.12, 0.018, 1.5)])
        )
        description = post.calls[0]["json"]["embeds"][0]["description"]
        assert "AAPL" in description
        assert "150.00" in description
        assert "-0.12" in description
        assert "1.8%" in description
        assert "1.50" in description


class TestDiscordNotifierRequestShape:
    def test_posts_to_the_configured_webhook_url(self):
        post = _CapturingPost()
        url = "https://discord.com/api/webhooks/42/abc"
        DiscordNotifier(webhook_url=url, post_fn=post).send(RunSucceeded(manifest=healthy_manifest()))
        assert post.calls[0]["url"] == url

    def test_reads_webhook_url_from_environment_when_not_passed_explicitly(self, monkeypatch):
        url = "https://discord.com/api/webhooks/99/fromenv"
        monkeypatch.setenv("DISCORD_WEBHOOK_URL", url)
        post = _CapturingPost()
        DiscordNotifier(post_fn=post).send(RunSucceeded(manifest=healthy_manifest()))
        assert post.calls[0]["url"] == url

    def test_sends_a_reasonable_timeout(self):
        post = _CapturingPost()
        DiscordNotifier(webhook_url="https://discord.com/api/webhooks/1/x", post_fn=post).send(
            RunSucceeded(manifest=healthy_manifest())
        )
        assert 0 < post.calls[0]["timeout"] <= 30
