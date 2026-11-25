# -*- coding: utf-8 -*-
"""Earnings gap-risk warning: released after the session / upcoming within the window."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Optional
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from data_provider.yfinance_fundamental_adapter import YfinanceFundamentalAdapter
from src.analyzer import AnalysisResult
from src.earnings_event_guardrail import apply_earnings_event_notice, detect_earnings_event
from src.services.paper_ledger import build_ledger_record

pytest.importorskip("exchange_calendars")


def _ctx(latest: Optional[str] = None, upcoming: Optional[str] = None) -> dict:
    report = {"report_date": "2026-09-30", "latest_earnings_date": latest, "next_earnings_date": upcoming}
    return {"earnings": {"data": {"financial_report": report}}}


def _result(code: str = "AAPL", decision: str = "hold", confidence: str = "High") -> AnalysisResult:
    return AnalysisResult(
        code=code,
        name="Test",
        sentiment_score=60,
        trend_prediction="Sideways",
        operation_advice="Buy" if decision == "buy" else "Watch",
        decision_type=decision,
        confidence_level=confidence,
        report_language="en",
        dashboard={"intelligence": {"risk_alerts": ["Existing risk."]}},
    )


def _utc(*args) -> datetime:
    return datetime(*args, tzinfo=timezone.utc)


def test_upcoming_after_close_within_window() -> None:
    # Session Tue 2027-01-26; AAPL reports Thu 2027-01-28 after the close -> reacts Fri 01-29.
    result = _result()
    adjustments = apply_earnings_event_notice(
        result, _ctx(upcoming="2027-01-28T16:00:00-05:00"), report_language="en", now=_utc(2027, 1, 26, 23, 0)
    )
    assert adjustments == ["earnings_event_flagged"]
    alerts = result.dashboard["intelligence"]["risk_alerts"]
    assert alerts[0] == (
        "⚠️ Earnings on 2027-01-28 after the close (in 3 session(s)): overnight gap risk; "
        "entry, stop and target levels may be skipped by a gap."
    )
    assert alerts[1] == "Existing risk."
    assert result.dashboard["earnings_event"]["reaction_session"] == "2027-01-29"
    assert result.confidence_level == "High"  # hold/watch: confidence untouched


def test_upcoming_outside_window_is_ignored() -> None:
    result = _result()
    assert apply_earnings_event_notice(
        result, _ctx(upcoming="2027-01-28T16:00:00-05:00"), now=_utc(2027, 1, 12, 23, 0)
    ) == []
    assert "earnings_event" not in result.dashboard


def test_released_after_the_analysed_session() -> None:
    # NVDA reported 2026-11-17 16:00 ET; the night run analyses the 11-17 session.
    result = _result(code="NVDA")
    apply_earnings_event_notice(
        result, _ctx(latest="2026-11-17T16:00:00-05:00"), report_language="en", now=_utc(2026, 11, 18, 3, 0)
    )
    event = result.dashboard["earnings_event"]
    assert (event["kind"], event["session"], event["reaction_session"]) == ("released", "2026-11-17", "2026-11-18")
    assert result.dashboard["intelligence"]["risk_alerts"][0].startswith(
        "⚠️ Earnings released 2026-11-17 after the close, after the analysed session 2026-11-17"
    )


def test_released_before_the_open_of_the_session_is_already_priced() -> None:
    report = {"latest_earnings_date": "2027-01-14T08:00:00-05:00"}
    assert detect_earnings_event(report, date(2027, 1, 14)) is None


def test_holiday_is_not_counted_as_a_session() -> None:
    # Session Tue 2026-11-24; results Wed 11-25 after close; Thu 11-26 is Thanksgiving.
    event = detect_earnings_event({"next_earnings_date": "2026-11-25T16:00:00-05:00"}, date(2026, 11, 24))
    assert (event["reaction_session"], event["sessions"]) == ("2026-11-27", 2)


def test_unknown_time_assumes_after_close() -> None:
    event = detect_earnings_event({"next_earnings_date": "2027-01-28T00:00:00-05:00"}, date(2027, 1, 26))
    assert (event["timing"], event["reaction_session"]) == ("unknown", "2027-01-29")


def test_buy_high_confidence_is_capped_at_medium() -> None:
    result = _result(decision="buy")
    adjustments = apply_earnings_event_notice(
        result, _ctx(upcoming="2027-01-28T16:00:00-05:00"), report_language="en", now=_utc(2027, 1, 26, 23, 0)
    )
    assert adjustments == ["earnings_event_flagged", "earnings_event_confidence_capped"]
    assert result.confidence_level == "Medium"
    assert result.decision_type == "buy"


def test_non_us_and_missing_data_are_ignored() -> None:
    assert apply_earnings_event_notice(_result(code="600519"), _ctx(upcoming="2027-01-28T16:00:00-05:00")) == []
    assert apply_earnings_event_notice(_result(), None) == []
    assert apply_earnings_event_notice(_result(), _ctx()) == []


def test_chinese_note() -> None:
    result = _result()
    apply_earnings_event_notice(
        result, _ctx(upcoming="2027-01-28T16:00:00-05:00"), report_language="zh", now=_utc(2027, 1, 26, 23, 0)
    )
    assert result.dashboard["intelligence"]["risk_alerts"][0].startswith("⚠️ 财报将于 2027-01-28盘后发布（3 个交易日内）")


def test_ledger_records_event() -> None:
    result = _result()
    apply_earnings_event_notice(
        result, _ctx(upcoming="2027-01-28T16:00:00-05:00"), now=_utc(2027, 1, 26, 23, 0)
    )
    record = build_ledger_record(result)
    assert record["earnings_event"] == {"kind": "upcoming", "date": "2027-01-28", "timing": "after_close", "sessions": 3}
    assert build_ledger_record(_result())["earnings_event"] is None


def test_adapter_exposes_next_earnings_date() -> None:
    future = (pd.Timestamp.now(tz="America/New_York") + pd.Timedelta(days=40)).normalize() + pd.Timedelta(hours=16)
    past = pd.Timestamp("2026-10-29 16:00", tz="America/New_York")
    frame = pd.DataFrame({"EPS Estimate": [1.5, 1.2], "Reported EPS": [None, 1.31]}, index=pd.DatetimeIndex([future, past]))
    ticker = MagicMock()
    info = {"financialCurrency": "USD", "currency": "USD", "totalRevenue": 1e9, "returnOnEquity": 0.2}
    ticker.get_info.return_value = info
    ticker.info = info
    ticker.quarterly_income_stmt = pd.DataFrame()
    ticker.quarterly_cashflow = pd.DataFrame()
    ticker.dividends = pd.Series(dtype="float64")
    ticker.get_earnings_dates.return_value = frame
    with patch("yfinance.Ticker", return_value=ticker):
        report = YfinanceFundamentalAdapter().get_fundamental_bundle("AAPL")["earnings"]["financial_report"]
    assert report["next_earnings_date"] == future.to_pydatetime().isoformat()
    assert report["latest_earnings_date"].startswith("2026-10-29T16:00")
