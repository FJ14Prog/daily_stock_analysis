# -*- coding: utf-8 -*-
"""Paper-trading ledger: one JSON line per stock, opt-in, fail-open."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.analyzer import AnalysisResult
from src.services.paper_ledger import append_ledger_records, build_ledger_record


def _result(**overrides) -> AnalysisResult:
    values = dict(
        code="AAPL",
        name="Apple Inc.",
        sentiment_score=59,
        trend_prediction="Bullish",
        operation_advice="Watch",
        decision_type="hold",
        confidence_level="Medium",
        report_language="en",
        action="watch",
        current_price=352.45,
        change_pct=0.15,
        model_used="gemini/gemini-3.6-flash",
        dashboard={
            "battle_plan": {
                "sniper_points": {
                    "ideal_buy": "Ideal buy point: $353.66 (near MA5)",
                    "secondary_buy": "Secondary buy point: $350.83 (near MA10)",
                    "stop_loss": "Stop-loss: $347.49 (break below MA20)",
                    "take_profit": "$356.42 (previous high / resistance level)",
                }
            },
            "action_plan_validation": {
                "violations": ["risk_reward_below_minimum"],
                "downgraded": False,
            },
        },
    )
    values.update(overrides)
    return AnalysisResult(**values)


def test_record_contains_plan_and_derived_metrics() -> None:
    record = build_ledger_record(
        _result(), recorded_at=datetime(2026, 11, 19, 2, 50, tzinfo=timezone.utc), run_id="42"
    )

    assert record["schema"] == 1
    assert record["ticker"] == "AAPL"
    assert record["recorded_at"] == "2026-11-19T02:50:00Z"
    assert record["run_id"] == "42"
    assert (record["decision"], record["action"], record["advice"]) == ("hold", "watch", "Watch")
    assert (record["ideal_buy"], record["secondary_buy"]) == (353.66, 350.83)
    assert (record["stop_loss"], record["take_profit"]) == (347.49, 356.42)
    # (356.42 - 353.66) / (353.66 - 347.49) = 0.4473
    assert record["risk_reward"] == 0.4473
    assert record["entry_distance_pct"] == 0.3433
    assert record["plan_violations"] == ["risk_reward_below_minimum"]
    assert record["model"] == "gemini/gemini-3.6-flash"
    assert record["session_date"] is not None


def test_incoherent_plan_has_no_risk_reward() -> None:
    result = _result()
    result.dashboard["battle_plan"]["sniper_points"]["stop_loss"] = "Stop-loss: 360.00"
    assert build_ledger_record(result)["risk_reward"] is None


def test_failed_results_are_skipped() -> None:
    assert build_ledger_record(_result(success=False)) is None
    assert build_ledger_record(None) is None


def test_append_writes_one_line_per_stock(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger" / "signals.jsonl"
    written = append_ledger_records([_result(), _result(code="MSFT", name="Microsoft")], path=ledger)
    written += append_ledger_records([_result(code="NVDA", name="NVIDIA")], path=ledger)

    lines = ledger.read_text(encoding="utf-8").splitlines()
    assert written == 3
    assert [json.loads(line)["ticker"] for line in lines] == ["AAPL", "MSFT", "NVDA"]


def test_disabled_without_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("PAPER_LEDGER_FILE", raising=False)
    assert append_ledger_records([_result()]) == 0


def test_env_var_enables_ledger(monkeypatch, tmp_path: Path) -> None:
    ledger = tmp_path / "signals.jsonl"
    monkeypatch.setenv("PAPER_LEDGER_FILE", str(ledger))
    assert append_ledger_records([_result()]) == 1
    assert ledger.exists()


def test_write_errors_never_raise(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x")
    assert append_ledger_records([_result()], path=blocker / "signals.jsonl") == 0


def test_workflow_wires_ledger_branch() -> None:
    workflow = yaml.safe_load(Path(".github/workflows/00-daily-analysis.yml").read_text(encoding="utf-8"))
    job = workflow["jobs"]["analyze"]
    assert job["permissions"] == {"contents": "write"}
    steps = {step.get("name"): step for step in job["steps"]}
    checkout = steps["Checkout paper ledger branch (optional)"]
    assert checkout["with"] == {"ref": "paper-ledger", "path": "paper-ledger"}
    assert checkout["continue-on-error"] is True
    push = steps["Push paper ledger"]
    assert push["continue-on-error"] is True
    assert "steps.paper_ledger.outcome == 'success'" in push["if"]
    analysis_script = steps["Run stock analysis"]["run"]
    assert 'export PAPER_LEDGER_FILE="paper-ledger/signals.jsonl"' in analysis_script
