"""
notify.py — Discord alerting for the put-screener pipeline.

Two rules, non-negotiable:

  1. This module must never raise. An alert exists to report a failure
     that has already been contained; a notifier that itself throws turns
     that contained failure into an unhandled crash of whatever called it.
     Every path through DiscordNotifier.send() — a missing webhook, a
     network error, a non-2xx response from Discord, an unrecognized
     event — is caught and, at most, printed to stderr.

  2. Nothing leaves this process before passing through redact(). That
     covers FINNHUB_API_KEY and DISCORD_WEBHOOK_URL by exact value (read
     fresh from the environment on every call, never cached), a
     token=/api_key=/key= fallback pattern for anything not caught by the
     exact match, and any local filesystem path (/Users/..., /home/...,
     /tmp/..., /app/..., /private/...) collapsed down to just its
     filename — enough to say "which file", nothing about whose machine
     or what directory structure it lives in.

     redact() is applied both at the point each dynamic string is built
     AND once more, as a final pass over the fully-serialized JSON payload
     right before it is sent. The second pass is the actual guarantee: it
     catches anything a future embed field forgets to redact individually.
     redact() is idempotent, so doing both costs nothing.

Single entry point: send(event), where event is one of RunFailed /
RunDegraded / RunSucceeded below, built from a run manifest (see
manifest.py) and, for RunSucceeded, the ranked shortlist. Classifying
*which* event applies to a given failure is the caller's job, not this
module's — manifest.check_sanity_gate() raises RuntimeError uniformly for
both the "incomplete screen" and "zero qualifying" conditions, and the
caller decides whether that reads as a hard failure or a softer degraded
signal.

DISCORD_WEBHOOK_URL is read from the environment only, the same rule
CLAUDE.md applies to FINNHUB_API_KEY: no flag, nothing here ever accepts
or persists a literal URL.
"""

import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol

import requests

COLOR_RED = 0xE74C3C
COLOR_AMBER = 0xF1C40F
COLOR_GREEN = 0x2ECC71

MAX_CANDIDATES_SHOWN = 5
EMBED_DESCRIPTION_LIMIT = 4096  # Discord's cap; a truncated embed still sends.

_SENSITIVE_ENV_VARS = ("FINNHUB_API_KEY", "DISCORD_WEBHOOK_URL")

_KV_SECRET_RE = re.compile(r'(?i)\b(token|api[_-]?key|key)=([^\s&"\']+)')
_DISCORD_WEBHOOK_RE = re.compile(r'https://(?:ptb\.|canary\.)?discord(?:app)?\.com/api/webhooks/\S+')
_LOCAL_PATH_PREFIXES = ("/Users/", "/home/", "/root/", "/private/", "/var/", "/tmp/", "/app/")
_LOCAL_PATH_RE = re.compile(
    "(?:" + "|".join(re.escape(p) for p in _LOCAL_PATH_PREFIXES) + r')[^\s"\')]*'
)


def redact(text: Optional[str]) -> Optional[str]:
    """Strip anything from `text` that could leak a secret or reveal local
    filesystem/system detail. Order is deliberate: exact known secrets
    first (works even if a pattern below has a gap), then the Discord
    webhook shape and key=value fallback, then local-path collapsing last
    — it must run after the URL-shaped patterns above it, not interfere
    with them.
    """
    if not text:
        return text

    for var in _SENSITIVE_ENV_VARS:
        value = os.environ.get(var)
        if value:
            text = text.replace(value, "[REDACTED]")

    text = _DISCORD_WEBHOOK_RE.sub("[REDACTED_WEBHOOK]", text)
    text = _KV_SECRET_RE.sub(lambda m: f"{m.group(1)}=[REDACTED]", text)
    text = _LOCAL_PATH_RE.sub(lambda m: Path(m.group(0)).name, text)

    return text


@dataclass
class RunFailed:
    """A run did not complete. `phase` is "screen" or "rank"; `error` is
    typically str(exception) and may contain a file path or, in principle,
    a stray secret — hence redaction, not trust, at the point it's used."""
    manifest: Dict[str, Any]
    error: str
    phase: str


@dataclass
class RunDegraded:
    """A run completed but looks wrong enough not to trust silently —
    e.g. the sanity gate's conditions (incomplete screen, zero qualifying
    contracts) surfaced as a soft signal rather than a hard failure."""
    manifest: Dict[str, Any]
    reason: str


@dataclass
class RunSucceeded:
    """A run completed and passed the sanity gate. `ranked` is the final
    shortlist (already sorted by bucketed abs(delta) ascending, annualized
    return descending within each bucket, per rank_top_n) — only the first
    MAX_CANDIDATES_SHOWN are rendered."""
    manifest: Dict[str, Any]
    ranked: List[Dict[str, Any]] = field(default_factory=list)


class Notifier(Protocol):
    def send(self, event: Any) -> None: ...


def _footer() -> Dict[str, str]:
    return {"text": "put-screener"}


def _base_embed(manifest: Dict[str, Any], color: int, title: str) -> Dict[str, Any]:
    embed: Dict[str, Any] = {"title": title, "color": color, "footer": _footer(), "fields": []}
    run_id = manifest.get("run_id")
    if run_id:
        embed["timestamp"] = run_id
    return embed


def _summary_fields(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    screened = manifest.get("tickers_screened", "?")
    total = manifest.get("tickers_total", "?")
    errored = manifest.get("tickers_errored", "?")
    duration = manifest.get("duration_seconds")
    return [
        {
            "name": "Tickers",
            "value": f"{screened}/{total} screened, {errored} errored",
            "inline": False,
        },
        {
            "name": "Qualifying contracts",
            "value": str(manifest.get("qualifying_contracts", "?")),
            "inline": True,
        },
        {
            "name": "Duration",
            "value": f"{duration}s" if duration is not None else "?",
            "inline": True,
        },
    ]


def _fmt_money(x: Any) -> str:
    return f"${x:.2f}" if isinstance(x, (int, float)) else "?"


def _fmt_pct(x: Any) -> str:
    return f"{x:.1%}" if isinstance(x, (int, float)) else "?"


def _fmt_signed(x: Any) -> str:
    return f"{x:+.2f}" if isinstance(x, (int, float)) else "?"


def _format_candidate(hit: Dict[str, Any]) -> str:
    symbol = hit.get("underlyingSymbol", "?")
    strike = _fmt_money(hit.get("strike"))
    delta = _fmt_signed(hit.get("delta"))
    roc = _fmt_pct(hit.get("return_on_capital"))
    bid = _fmt_money(hit.get("bid"))
    return f"**{symbol}** {strike} put — Δ{delta}, ROC {roc}, bid {bid}"


def _failed_embed(event: RunFailed) -> Dict[str, Any]:
    embed = _base_embed(event.manifest, COLOR_RED, "\N{LARGE RED CIRCLE} put-screener run FAILED")
    embed["description"] = (redact(event.error) or "")[:EMBED_DESCRIPTION_LIMIT]
    embed["fields"] = [
        {"name": "Phase", "value": redact(event.phase) or "?", "inline": True},
        *_summary_fields(event.manifest),
    ]
    return embed


def _degraded_embed(event: RunDegraded) -> Dict[str, Any]:
    embed = _base_embed(event.manifest, COLOR_AMBER, "\N{LARGE YELLOW CIRCLE} put-screener run DEGRADED")
    embed["description"] = (redact(event.reason) or "")[:EMBED_DESCRIPTION_LIMIT]
    embed["fields"] = _summary_fields(event.manifest)
    return embed


def _succeeded_embed(event: RunSucceeded) -> Dict[str, Any]:
    manifest = event.manifest
    embed = _base_embed(manifest, COLOR_GREEN, "\N{LARGE GREEN CIRCLE} put-screener run complete")

    ranked = event.ranked
    shown = ranked[:MAX_CANDIDATES_SHOWN]
    lines = [_format_candidate(h) for h in shown]
    if len(ranked) > MAX_CANDIDATES_SHOWN:
        lines.append(f"...and {len(ranked) - MAX_CANDIDATES_SHOWN} more in screen_ranked.json")
    description = "\n".join(lines) if lines else "No qualifying candidates this run."
    embed["description"] = description[:EMBED_DESCRIPTION_LIMIT]

    embed["fields"] = [
        *_summary_fields(manifest),
        {
            "name": "After earnings exclusion",
            "value": str(manifest.get("after_earnings_exclusion", "?")),
            "inline": True,
        },
        {"name": "Final ranked", "value": str(manifest.get("final_ranked", "?")), "inline": True},
    ]
    return embed


def _build_payload(event: Any) -> Dict[str, Any]:
    if isinstance(event, RunSucceeded):
        embed = _succeeded_embed(event)
    elif isinstance(event, RunDegraded):
        embed = _degraded_embed(event)
    elif isinstance(event, RunFailed):
        embed = _failed_embed(event)
    else:
        # A programmer error (unknown event type), not a runtime failure —
        # deliberately raised here rather than silently ignored. send()
        # below still catches it, so this never escapes to the caller.
        raise TypeError(f"Unknown notification event type: {type(event).__name__}")
    return {"embeds": [embed]}


class DiscordNotifier:
    """Posts NotificationEvents to an incoming Discord webhook.

    `webhook_url` is read from DISCORD_WEBHOOK_URL at send time (not at
    construction), so a webhook configured after this object exists — or
    in a different environment than it was created in, as in a test —
    still works, and nothing here caches a value that could go stale.
    """

    def __init__(self, webhook_url: Optional[str] = None, timeout: float = 10.0,
                 post_fn: Optional[Callable[..., Any]] = None):
        self._webhook_url_override = webhook_url
        self._timeout = timeout
        self._post_fn = post_fn or requests.post

    def send(self, event: Any) -> None:
        try:
            self._send(event)
        except Exception as e:
            # The only acceptable outcome of a notifier failure: a stderr
            # line, not a propagated exception. Message text still goes
            # through redact() — an exception from a network layer can
            # legitimately embed the URL it tried to reach.
            print(f"[notify] failed to send {type(event).__name__}: "
                  f"{redact(f'{type(e).__name__}: {e}')}", file=sys.stderr)

    def _send(self, event: Any) -> None:
        webhook_url = self._webhook_url_override or os.environ.get("DISCORD_WEBHOOK_URL")
        if not webhook_url:
            print(f"[notify] DISCORD_WEBHOOK_URL not set — skipping "
                  f"{type(event).__name__} notification", file=sys.stderr)
            return

        payload = _build_payload(event)
        # Defense in depth: even if some future embed field is assembled
        # without an explicit redact() call, nothing leaves this process
        # without this final pass over the fully serialized payload.
        payload = json.loads(redact(json.dumps(payload)))

        response = self._post_fn(webhook_url, json=payload, timeout=self._timeout)
        if response.status_code >= 300:
            print(f"[notify] Discord returned {response.status_code} for "
                  f"{type(event).__name__}", file=sys.stderr)


def notify(event: Any) -> None:
    """Convenience: build a DiscordNotifier reading DISCORD_WEBHOOK_URL
    from the environment and send `event`. Never raises."""
    DiscordNotifier().send(event)
