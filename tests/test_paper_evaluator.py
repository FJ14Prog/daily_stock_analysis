# -*- coding: utf-8 -*-
"""Paper-trading evaluator: fills, exits, gaps, costs, no look-ahead, summary."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from src.services.paper_evaluator import (
    Bar,
    EvalConfig,
    dedupe_ledger,
    evaluate_signal,
    render_summary_markdown,
    rows_to_csv,
    summarize,
)

SESSION = date(2026, 11, 2)  # Monday
ZERO_COST = EvalConfig(cost_bps=0.0, entry_valid_sessions=3, max_hold_sessions=5, direction_sessions=3)


def _days(n: int, start: date = SESSION):
    out, day = [], start
    while len(out) < n:
        day += timedelta(days=1)
        if day.weekday() < 5:
            out.append(day)
    return out


def _bars(ohlc):
    return [Bar(d, *values) for d, values in zip(_days(len(ohlc)), ohlc)]


def _signal(decision="buy", ideal=100.0, stop=95.0, target=110.0, **extra):
    record = {
        "session_date": SESSION.isoformat(), "ticker": "TEST", "decision": decision, "advice": decision.title(),
        "score": 70, "ideal_buy": ideal, "stop_loss": stop, "take_profit": target,
    }
    record.update(extra)
    return record


def test_session_day_bar_is_never_used() -> None:
    same_day = Bar(SESSION, 99, 120, 90, 100)  # would fill and hit target
    later = _bars([(105, 106, 104, 105)] * 5)
    row = evaluate_signal(_signal(), [same_day, *later], config=ZERO_COST)
    assert row["status"] == "not_filled"


def test_limit_fill_then_target() -> None:
    bars = _bars([(103, 104, 99.5, 102), (102, 111, 101, 110), (110, 110, 109, 109)])
    row = evaluate_signal(_signal(), bars, config=ZERO_COST)
    assert row["status"] == "closed"
    assert row["entry_price"] == 100.0  # intraday at the limit
    assert row["exit_reason"] == "target" and row["exit_price"] == 110.0
    assert row["r_multiple"] == 2.0 and row["return_pct"] == 10.0
    assert row["hold_sessions"] == 2
    assert row["planned_rr"] == 2.0


def test_gap_below_limit_fills_at_open() -> None:
    bars = _bars([(98, 99, 97, 98)] * 5)
    row = evaluate_signal(_signal(), bars, config=ZERO_COST)
    assert row["entry_price"] == 98.0


def test_stop_gap_exits_at_open_not_stop() -> None:
    bars = _bars([(100, 101, 99, 100), (90, 91, 89, 90)])
    row = evaluate_signal(_signal(), bars, config=ZERO_COST)
    assert row["exit_reason"] == "stop_gap" and row["exit_price"] == 90.0
    assert row["r_multiple"] == -2.0  # worse than the planned -1R


def test_stop_and_target_same_bar_assumes_stop() -> None:
    bars = _bars([(100, 101, 99.5, 100), (100, 112, 94, 100)])
    row = evaluate_signal(_signal(), bars, config=ZERO_COST)
    assert row["exit_reason"] == "ambiguous_stop" and row["r_multiple"] == -1.0


def test_intraday_fill_does_not_count_same_bar_target() -> None:
    # Opens above the limit, dips to fill and also prints the target: order unknown.
    bars = _bars([(103, 111, 99, 104), (104, 105, 103, 104), (104, 105, 103, 104), (104, 105, 103, 104), (104, 105, 103, 104)])
    row = evaluate_signal(_signal(), bars, config=ZERO_COST)
    assert row["exit_reason"] == "timeout"
    assert row["exit_price"] == 104.0


def test_timeout_and_open_trade() -> None:
    flat = [(100, 101, 99, 100)] + [(101, 102, 100.5, 101)] * 4
    assert evaluate_signal(_signal(), _bars(flat), config=ZERO_COST)["exit_reason"] == "timeout"
    assert evaluate_signal(_signal(), _bars(flat[:3]), config=ZERO_COST)["status"] == "open"


def test_pending_entry_when_window_not_finished() -> None:
    assert evaluate_signal(_signal(), _bars([(105, 106, 104, 105)]), config=ZERO_COST)["status"] == "pending_entry"


def test_costs_reduce_result() -> None:
    bars = _bars([(103, 104, 99.5, 102), (102, 111, 101, 110)])
    row = evaluate_signal(_signal(), bars, config=EvalConfig(cost_bps=10, entry_valid_sessions=3, max_hold_sessions=5))
    # entry 100.10, exit 109.89
    assert row["return_pct"] == pytest.approx(9.7802, abs=1e-3)
    assert row["r_multiple"] == pytest.approx(1.958, abs=1e-3)


def test_split_inside_window_excludes_signal() -> None:
    bars = _bars([(100, 101, 99, 100), (100, 111, 100, 110)])
    row = evaluate_signal(_signal(), bars, config=ZERO_COST, split_days=[bars[1].day])
    assert row["status"] == "excluded_split"


def test_watch_with_valid_plan_is_shadow_and_without_plan_is_skipped() -> None:
    bars = _bars([(103, 104, 99.5, 102), (102, 111, 101, 110)])
    assert evaluate_signal(_signal("hold"), bars, config=ZERO_COST)["group"] == "shadow"
    incoherent = _signal("hold", ideal=100, stop=101, target=110)
    assert evaluate_signal(incoherent, bars, config=ZERO_COST)["status"] == "no_plan"


def test_invalid_buy_plan() -> None:
    row = evaluate_signal(_signal(stop=100.0), _bars([(100, 101, 99, 100)] * 5), config=ZERO_COST)
    assert row["status"] == "invalid_plan"


def test_sell_direction_check() -> None:
    bars = _bars([(100, 101, 97, 98), (98, 99, 95, 96), (96, 97, 93, 94)])
    row = evaluate_signal(_signal("sell"), bars, config=ZERO_COST)
    assert row["group"] == "direction" and row["direction_correct"] is True
    assert row["benchmark_return_pct"] == -6.0


def test_benchmark_returns() -> None:
    bars = _bars([(103, 104, 99.5, 102), (102, 111, 101, 110)])
    spy = _bars([(500, 501, 499, 500), (500, 506, 499, 505)])
    row = evaluate_signal(_signal(), bars, config=ZERO_COST, benchmark_bars=spy)
    assert row["benchmark_return_pct"] == pytest.approx(6.7961, abs=1e-3)  # 103 -> 110
    assert row["spy_return_pct"] == 1.0


def test_dedupe_keeps_last_record_per_session_and_ticker() -> None:
    records = [_signal(score=50), _signal(score=60), _signal(ticker="MSFT")]
    out = dedupe_ledger(records)
    assert len(out) == 2
    assert [r["score"] for r in out if r["ticker"] == "TEST"] == [60]


def test_summary_and_rendering() -> None:
    win = evaluate_signal(_signal(), _bars([(103, 104, 99.5, 102), (102, 111, 101, 110)]), config=ZERO_COST)
    loss = evaluate_signal(_signal(ticker="B"), _bars([(100, 101, 99, 100), (90, 91, 89, 90)]), config=ZERO_COST)
    summary = summarize([win, loss])
    buy = summary["buy"]
    assert buy["closed"] == 2 and buy["win_rate"] == 0.5
    assert buy["expectancy_r"] == 0.0  # (+2 - 2) / 2
    assert buy["max_drawdown_r"] == -2.0
    assert buy["small_sample"] is True
    text = render_summary_markdown(summary, as_of="2026-11-28", config=ZERO_COST)
    assert "# 📊 Paper Trading Review - 2026-11-28" in text
    assert "| Buy | 2 | 2 | 0 | 100.00% | 50.00% | 0.00 |" in text
    assert "not statistically meaningful" in text
    csv_text = rows_to_csv([win, loss])
    assert csv_text.splitlines()[0].startswith("session_date,ticker,decision")
    assert len(csv_text.splitlines()) == 3


def test_weekly_workflow_wiring() -> None:
    from pathlib import Path

    import yaml

    workflow = yaml.safe_load(Path(".github/workflows/paper-evaluation.yml").read_text(encoding="utf-8"))
    job = workflow["jobs"]["evaluate"]
    assert job["permissions"] == {"contents": "write"}
    steps = {step["name"]: step for step in job["steps"]}
    assert steps["Checkout paper ledger branch"]["with"] == {"ref": "paper-ledger", "path": "paper-ledger"}
    assert "scripts/evaluate_paper_ledger.py" in steps["Evaluate ledger"]["run"]
    assert "evaluations.csv summary.md" in steps["Push evaluation"]["run"]
    triggers = workflow.get("on") or workflow.get(True)
    assert triggers["schedule"][0]["cron"].endswith("* * 6")


def test_script_writes_outputs_without_network(tmp_path, monkeypatch) -> None:
    import json

    import scripts.evaluate_paper_ledger as script

    ledger = tmp_path / "signals.jsonl"
    ledger.write_text(json.dumps(_signal()) + "\n", encoding="utf-8")
    bars = _bars([(103, 104, 99.5, 102), (102, 111, 101, 110)])
    monkeypatch.setattr(script, "fetch_bars", lambda symbol, start: (bars, []))
    assert script.main(["--ledger", str(ledger), "--cost-bps", "0"]) == 0
    assert (tmp_path / "evaluations.csv").read_text(encoding="utf-8").count("\n") == 2
    assert "Paper Trading Review" in (tmp_path / "summary.md").read_text(encoding="utf-8")


def test_script_without_ledger_is_noop(tmp_path) -> None:
    import scripts.evaluate_paper_ledger as script

    assert script.main(["--ledger", str(tmp_path / "missing.jsonl")]) == 0
