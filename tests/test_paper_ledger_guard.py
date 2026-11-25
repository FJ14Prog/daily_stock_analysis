# -*- coding: utf-8 -*-
"""Tests for the scheduled-run guard (primary + backup cron)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

import scripts.paper_ledger_guard as guard
from src.services.paper_ledger import ledger_has_session

pytest.importorskip("exchange_calendars")


def _ledger(tmp_path: Path, *sessions: str) -> Path:
    path = tmp_path / "signals.jsonl"
    lines = [json.dumps({"session_date": s, "ticker": "AAPL"}) for s in sessions]
    path.write_text("\n".join(lines + ["", "not json"]) + "\n", encoding="utf-8")
    return path


def test_ledger_has_session(tmp_path: Path) -> None:
    path = _ledger(tmp_path, "2026-11-23", "2026-11-24")
    assert ledger_has_session(path, "2026-11-24") is True
    assert ledger_has_session(path, "2026-11-25") is False
    assert ledger_has_session(tmp_path / "missing.jsonl", "2026-11-24") is False


@pytest.mark.parametrize(
    "now, expected",
    [
        # Primary cron after the close (17:07 EST) -> same-day session.
        (datetime(2026, 11, 24, 22, 7, tzinfo=timezone.utc), "2026-11-24"),
        # Backup cron delayed past midnight New York time -> previous session.
        (datetime(2026, 11, 25, 9, 30, tzinfo=timezone.utc), "2026-11-24"),
        # Saturday backup -> Friday session.
        (datetime(2026, 11, 21, 2, 43, tzinfo=timezone.utc), "2026-11-20"),
        # Thanksgiving (2026-11-26) is a holiday -> last session is Wednesday.
        (datetime(2026, 11, 27, 2, 43, tzinfo=timezone.utc), "2026-11-25"),
    ],
)
def test_decide_session(tmp_path: Path, now: datetime, expected: str) -> None:
    skip, session = guard.decide(_ledger(tmp_path), now=now)
    assert session == expected
    assert skip is False


def test_decide_skips_when_session_recorded(tmp_path: Path) -> None:
    now = datetime(2026, 11, 25, 7, 12, tzinfo=timezone.utc)
    skip, session = guard.decide(_ledger(tmp_path, "2026-11-24"), now=now)
    assert (skip, session) == (True, "2026-11-24")


def test_main_writes_github_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "out.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setattr(guard, "decide", lambda ledger, market="us", now=None: (True, "2026-11-24"))
    assert guard.main(["--ledger", str(tmp_path / "x.jsonl")]) == 0
    assert output.read_text(encoding="utf-8") == "skip=true\nsession=2026-11-24\n"


def test_workflow_has_backup_cron_and_guard() -> None:
    workflow = yaml.safe_load(Path(".github/workflows/00-daily-analysis.yml").read_text(encoding="utf-8"))
    triggers = workflow.get("on") or workflow.get(True)
    crons = [item["cron"] for item in triggers["schedule"]]
    assert crons == ["7 22 * * 1-5", "43 2 * * 2-6"]
    guard_job = workflow["jobs"]["guard"]
    assert guard_job["outputs"]["skip"] == "${{ steps.check.outputs.skip }}"
    check = {s.get("name"): s for s in guard_job["steps"]}["Check paper ledger for this session"]
    assert check["if"] == "github.event_name == 'schedule'"
    assert check["continue-on-error"] is True
    analyze = workflow["jobs"]["analyze"]
    assert analyze["needs"] == "guard"
    assert analyze["if"] == "always() && needs.guard.outputs.skip != 'true'"
