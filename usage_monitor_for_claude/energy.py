"""
Energy Estimate
================

Rough estimate of the electricity used by Claude Code sessions this week
and this month, derived from local token counts.

Anthropic does not publish a per-token energy figure for Claude, so this
is *not* a measurement - it multiplies token counts already logged by
Claude Code by configurable Wh/1K-token rates (see ``settings.py`` and
``docs/energy-estimate.md`` for the reasoning behind the defaults).

Token counts come from the transcript files Claude Code writes to
``~/.claude/projects/**/*.jsonl`` (one line per turn, including a
``message.usage`` object with input/output/cache token counts) - a
different, local-only data source from the OAuth usage API in ``api.py``,
which only reports quota utilization percentages, not token counts.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from .api import CLAUDE_CONFIG_DIR
from .settings import (
    ENERGY_WH_PER_1K_CACHE_READ_TOKENS, ENERGY_WH_PER_1K_INPUT_TOKENS, ENERGY_WH_PER_1K_OUTPUT_TOKENS,
)

__all__ = ['PROJECTS_DIR', 'TokenTotals', 'energy_summary']

PROJECTS_DIR = CLAUDE_CONFIG_DIR / 'projects'

_CACHE_TTL = 60.0


@dataclass(frozen=True)
class TokenTotals:
    """Summed token counts for a period, broken down by billing category."""

    input_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: TokenTotals) -> TokenTotals:
        return TokenTotals(
            self.input_tokens + other.input_tokens,
            self.cache_creation_tokens + other.cache_creation_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
            self.output_tokens + other.output_tokens,
        )

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.cache_creation_tokens + self.cache_read_tokens + self.output_tokens


def _parse_record(line: str) -> tuple[str | None, str | None, datetime, TokenTotals] | None:
    """Parse one transcript line into (message_id, request_id, timestamp, tokens), or None."""
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None

    message = obj.get('message')
    if not isinstance(message, dict):
        return None
    usage = message.get('usage')
    if not isinstance(usage, dict):
        return None

    ts_raw = obj.get('timestamp')
    if not ts_raw:
        return None
    try:
        timestamp = datetime.fromisoformat(ts_raw.replace('Z', '+00:00'))
    except ValueError:
        return None

    tokens = TokenTotals(
        input_tokens=usage.get('input_tokens', 0) or 0,
        cache_creation_tokens=usage.get('cache_creation_input_tokens', 0) or 0,
        cache_read_tokens=usage.get('cache_read_input_tokens', 0) or 0,
        output_tokens=usage.get('output_tokens', 0) or 0,
    )
    return message.get('id'), obj.get('requestId'), timestamp, tokens


def _iter_usage_records(cutoff: datetime) -> list[tuple[datetime, TokenTotals]]:
    """Return (timestamp, tokens) for every assistant turn on/after *cutoff*.

    Deduplicates by (message id, request id): the same assistant turn
    sometimes appears more than once across transcript files (e.g. resumed
    or compacted sessions), and re-counting it would inflate the estimate.
    """
    if not PROJECTS_DIR.is_dir():
        return []

    records: list[tuple[datetime, TokenTotals]] = []
    seen: set[tuple[str | None, str | None]] = set()

    for path in PROJECTS_DIR.rglob('*.jsonl'):
        try:
            if datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc) < cutoff:
                continue
        except OSError:
            continue

        try:
            with path.open(encoding='utf-8') as f:
                for line in f:
                    if '"usage"' not in line:
                        continue
                    parsed = _parse_record(line)
                    if parsed is None:
                        continue
                    message_id, request_id, timestamp, tokens = parsed
                    if timestamp < cutoff:
                        continue
                    key = (message_id, request_id)
                    if key in seen:
                        continue
                    seen.add(key)
                    records.append((timestamp, tokens))
        except OSError:
            continue

    return records


def _period_starts(now: datetime) -> tuple[datetime, datetime]:
    """Return (week_start, month_start) as UTC-aware datetimes.

    Both boundaries are anchored to local time (week starts Monday
    00:00 local, month starts on the 1st 00:00 local) since that matches
    how a person thinks about "this week" / "this month", then converted
    to UTC to compare against transcript timestamps.
    """
    local_now = now.astimezone()
    local_midnight = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = local_midnight - timedelta(days=local_midnight.weekday())
    month_start = local_midnight.replace(day=1)
    return week_start.astimezone(timezone.utc), month_start.astimezone(timezone.utc)


def _estimate_wh(totals: TokenTotals) -> float:
    """Convert token totals into an estimated Wh figure.

    Uses three separate rates because inference cost differs a lot by
    token type: fresh input/cache-write tokens are processed in a
    parallelizable prefill pass, cache-read tokens reuse an existing KV
    cache for very little extra compute, and output tokens are generated
    one at a time (the most expensive per token). All three rates are
    rough public estimates, not Anthropic-published figures.
    """
    return (
        (totals.input_tokens + totals.cache_creation_tokens) / 1000 * ENERGY_WH_PER_1K_INPUT_TOKENS
        + totals.cache_read_tokens / 1000 * ENERGY_WH_PER_1K_CACHE_READ_TOKENS
        + totals.output_tokens / 1000 * ENERGY_WH_PER_1K_OUTPUT_TOKENS
    )


_cache_lock = threading.Lock()
_cached_summary: dict[str, Any] | None = None
_cached_at = 0.0


def energy_summary(*, force: bool = False) -> dict[str, Any]:
    """Return estimated energy use for the current week and current month.

    Cached for ``_CACHE_TTL`` seconds - scanning transcripts is fast
    (well under a second for typical usage), but there is no need to
    redo it on every 2-second popup refresh tick.

    Returns
    -------
    dict
        ``{'week_wh': float, 'week_tokens': int, 'month_wh': float, 'month_tokens': int}``
    """
    global _cached_summary, _cached_at

    with _cache_lock:
        if not force and _cached_summary is not None and time.time() - _cached_at < _CACHE_TTL:
            return _cached_summary

        now = datetime.now(timezone.utc)
        week_start, month_start = _period_starts(now)
        cutoff = min(week_start, month_start)

        week_totals = TokenTotals()
        month_totals = TokenTotals()
        for timestamp, tokens in _iter_usage_records(cutoff):
            if timestamp >= week_start:
                week_totals += tokens
            if timestamp >= month_start:
                month_totals += tokens

        summary = {
            'week_wh': _estimate_wh(week_totals),
            'week_tokens': week_totals.total_tokens,
            'month_wh': _estimate_wh(month_totals),
            'month_tokens': month_totals.total_tokens,
        }
        _cached_summary = summary
        _cached_at = time.time()
        return summary
