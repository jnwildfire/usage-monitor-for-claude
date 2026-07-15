"""
Energy Tests
=============

Unit tests for transcript scanning/deduplication, week/month period
boundaries, the Wh estimate formula, and energy_summary() caching.
"""
from __future__ import annotations

import json
import os
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from usage_monitor_for_claude import energy as energy_mod
from usage_monitor_for_claude.energy import TokenTotals, _estimate_wh, _iter_usage_records, _period_starts, energy_summary


def _write_record(
    path: Path, *, timestamp: str, message_id: str = 'msg_1', request_id: str = 'req_1',
    input_tokens: int = 100, output_tokens: int = 50, cache_creation_tokens: int = 0, cache_read_tokens: int = 0,
    append: bool = True,
) -> None:
    """Append one transcript-style JSONL line to *path*."""
    line = json.dumps({
        'timestamp': timestamp,
        'requestId': request_id,
        'message': {
            'id': message_id,
            'usage': {
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'cache_creation_input_tokens': cache_creation_tokens,
                'cache_read_input_tokens': cache_read_tokens,
            },
        },
    })
    mode = 'a' if append else 'w'
    with path.open(mode, encoding='utf-8') as f:
        f.write(line + '\n')


# ---------------------------------------------------------------------------
# _iter_usage_records
# ---------------------------------------------------------------------------

class TestIterUsageRecords(unittest.TestCase):
    """Tests for _iter_usage_records()."""

    def test_missing_projects_dir_returns_empty(self):
        with TemporaryDirectory() as tmp:
            missing = Path(tmp) / 'does-not-exist'
            with patch.object(energy_mod, 'PROJECTS_DIR', missing):
                self.assertEqual(_iter_usage_records(datetime.min.replace(tzinfo=timezone.utc)), [])

    def test_parses_a_record(self):
        with TemporaryDirectory() as tmp:
            projects = Path(tmp)
            _write_record(projects / 'a.jsonl', timestamp='2026-07-14T12:00:00Z', input_tokens=1000, output_tokens=200)
            with patch.object(energy_mod, 'PROJECTS_DIR', projects):
                records = _iter_usage_records(datetime(2026, 1, 1, tzinfo=timezone.utc))
            self.assertEqual(len(records), 1)
            _, tokens = records[0]
            self.assertEqual(tokens, TokenTotals(input_tokens=1000, output_tokens=200))

    def test_dedupes_by_message_and_request_id(self):
        """The same (message id, request id) logged twice counts once."""
        with TemporaryDirectory() as tmp:
            projects = Path(tmp)
            path = projects / 'a.jsonl'
            _write_record(path, timestamp='2026-07-14T12:00:00Z', message_id='msg_1', request_id='req_1', input_tokens=1000)
            _write_record(path, timestamp='2026-07-14T12:00:01Z', message_id='msg_1', request_id='req_1', input_tokens=1000)
            with patch.object(energy_mod, 'PROJECTS_DIR', projects):
                records = _iter_usage_records(datetime(2026, 1, 1, tzinfo=timezone.utc))
            self.assertEqual(len(records), 1)

    def test_distinct_request_ids_not_deduped(self):
        with TemporaryDirectory() as tmp:
            projects = Path(tmp)
            path = projects / 'a.jsonl'
            _write_record(path, timestamp='2026-07-14T12:00:00Z', message_id='msg_1', request_id='req_1')
            _write_record(path, timestamp='2026-07-14T12:00:01Z', message_id='msg_2', request_id='req_2')
            with patch.object(energy_mod, 'PROJECTS_DIR', projects):
                records = _iter_usage_records(datetime(2026, 1, 1, tzinfo=timezone.utc))
            self.assertEqual(len(records), 2)

    def test_records_before_cutoff_excluded(self):
        with TemporaryDirectory() as tmp:
            projects = Path(tmp)
            path = projects / 'a.jsonl'
            _write_record(path, timestamp='2026-07-01T00:00:00Z', message_id='old')
            _write_record(path, timestamp='2026-07-14T00:00:00Z', message_id='new')
            with patch.object(energy_mod, 'PROJECTS_DIR', projects):
                records = _iter_usage_records(datetime(2026, 7, 10, tzinfo=timezone.utc))
            self.assertEqual(len(records), 1)

    def test_file_older_than_cutoff_skipped_entirely(self):
        """A file whose mtime predates the cutoff is never opened - it cannot contain newer entries."""
        with TemporaryDirectory() as tmp:
            projects = Path(tmp)
            path = projects / 'a.jsonl'
            _write_record(path, timestamp='2026-07-14T00:00:00Z')  # would match if the file were scanned
            old_time = datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()
            os.utime(path, (old_time, old_time))
            with patch.object(energy_mod, 'PROJECTS_DIR', projects):
                records = _iter_usage_records(datetime(2026, 7, 1, tzinfo=timezone.utc))
            self.assertEqual(records, [])

    def test_lines_without_usage_are_skipped(self):
        with TemporaryDirectory() as tmp:
            projects = Path(tmp)
            path = projects / 'a.jsonl'
            with path.open('w', encoding='utf-8') as f:
                f.write(json.dumps({'timestamp': '2026-07-14T00:00:00Z', 'message': {'role': 'user'}}) + '\n')
                f.write('not even json\n')
            with patch.object(energy_mod, 'PROJECTS_DIR', projects):
                records = _iter_usage_records(datetime(2026, 1, 1, tzinfo=timezone.utc))
            self.assertEqual(records, [])


# ---------------------------------------------------------------------------
# _period_starts
# ---------------------------------------------------------------------------

class TestPeriodStarts(unittest.TestCase):
    """Tests for _period_starts()."""

    def test_week_starts_monday_month_starts_first(self):
        # 2026-07-15 is a Wednesday.
        now = datetime(2026, 7, 15, 18, 30, tzinfo=timezone.utc)
        week_start, month_start = _period_starts(now)
        self.assertEqual(week_start.astimezone().date(), datetime(2026, 7, 13).date())
        self.assertEqual(month_start.astimezone().date(), datetime(2026, 7, 1).date())

    def test_week_can_precede_month_start(self):
        # 2026-08-01 is a Saturday - the Monday of that week falls in July.
        now = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
        week_start, month_start = _period_starts(now)
        self.assertLess(week_start, month_start)


# ---------------------------------------------------------------------------
# _estimate_wh
# ---------------------------------------------------------------------------

class TestEstimateWh(unittest.TestCase):
    """Tests for _estimate_wh()."""

    @patch.object(energy_mod, 'ENERGY_WH_PER_1K_INPUT_TOKENS', 0.1)
    @patch.object(energy_mod, 'ENERGY_WH_PER_1K_OUTPUT_TOKENS', 1.0)
    @patch.object(energy_mod, 'ENERGY_WH_PER_1K_CACHE_READ_TOKENS', 0.01)
    def test_weights_each_token_category(self):
        totals = TokenTotals(input_tokens=1000, cache_creation_tokens=1000, cache_read_tokens=1000, output_tokens=1000)
        # (1000 + 1000)/1000 * 0.1 + 1000/1000 * 0.01 + 1000/1000 * 1.0
        self.assertAlmostEqual(_estimate_wh(totals), 0.2 + 0.01 + 1.0)

    @patch.object(energy_mod, 'ENERGY_WH_PER_1K_INPUT_TOKENS', 0.1)
    @patch.object(energy_mod, 'ENERGY_WH_PER_1K_OUTPUT_TOKENS', 1.0)
    @patch.object(energy_mod, 'ENERGY_WH_PER_1K_CACHE_READ_TOKENS', 0.01)
    def test_zero_tokens_is_zero(self):
        self.assertEqual(_estimate_wh(TokenTotals()), 0.0)


# ---------------------------------------------------------------------------
# energy_summary
# ---------------------------------------------------------------------------

class TestEnergySummary(unittest.TestCase):
    """Tests for energy_summary(), including caching."""

    def setUp(self):
        # Each test starts with a clean module-level cache.
        energy_mod._cached_summary = None
        energy_mod._cached_at = 0.0

    @patch.object(energy_mod, 'ENERGY_WH_PER_1K_INPUT_TOKENS', 0.1)
    @patch.object(energy_mod, 'ENERGY_WH_PER_1K_OUTPUT_TOKENS', 1.0)
    @patch.object(energy_mod, 'ENERGY_WH_PER_1K_CACHE_READ_TOKENS', 0.01)
    def test_aggregates_current_week_and_month(self):
        with TemporaryDirectory() as tmp:
            projects = Path(tmp)
            path = projects / 'a.jsonl'
            now = datetime.now(timezone.utc)
            _write_record(
                path, timestamp=now.isoformat().replace('+00:00', 'Z'),
                message_id='recent', request_id='r1', input_tokens=1000, output_tokens=0,
            )
            with patch.object(energy_mod, 'PROJECTS_DIR', projects):
                summary = energy_summary()
            self.assertEqual(summary['week_tokens'], 1000)
            self.assertEqual(summary['month_tokens'], 1000)
            self.assertAlmostEqual(summary['week_wh'], 0.1)

    def test_result_is_cached_until_forced(self):
        with TemporaryDirectory() as tmp:
            projects = Path(tmp)
            with patch.object(energy_mod, 'PROJECTS_DIR', projects):
                first = energy_summary()
                # Add a new file after the first call - a cached call must not see it.
                _write_record(projects / 'a.jsonl', timestamp=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'))
                cached = energy_summary()
                self.assertEqual(cached, first)
                forced = energy_summary(force=True)
                self.assertNotEqual(forced['week_tokens'], first['week_tokens'])

    @patch.object(energy_mod, '_CACHE_TTL', 0.0)
    def test_ttl_expiry_triggers_rescan(self):
        with TemporaryDirectory() as tmp:
            projects = Path(tmp)
            with patch.object(energy_mod, 'PROJECTS_DIR', projects):
                first = energy_summary()
                _write_record(projects / 'a.jsonl', timestamp=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'))
                second = energy_summary()
                self.assertNotEqual(second['week_tokens'], first['week_tokens'])


if __name__ == '__main__':
    unittest.main()
