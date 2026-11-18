# -*- coding: utf-8 -*-
"""Report titles must carry the market session date, not the host wall-clock date."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

from src.core.trading_calendar import format_report_timestamp, resolve_report_date
from src.market_phase_summary import report_date_from_market_phase_summary
from src.notification import NotificationService

UTC = timezone.utc
NY = ZoneInfo("America/New_York")


@pytest.mark.parametrize(
    "moment,expected",
    [
        # Tue 2026-11-17 21:30 UTC = 16:30 ET, after close -> same session
        (datetime(2026, 11, 17, 21, 30, tzinfo=UTC), "2026-11-17"),
        # Wed 2026-11-18 03:06 UTC = 22:06 ET Tue -> still Tuesday's session
        (datetime(2026, 11, 18, 3, 6, tzinfo=UTC), "2026-11-17"),
        # Wed 2026-11-18 13:00 UTC = 08:00 ET, pre-market -> previous session
        (datetime(2026, 11, 18, 13, 0, tzinfo=UTC), "2026-11-17"),
        # Wed 2026-11-18 15:00 UTC = 10:00 ET, intraday -> today's partial session
        (datetime(2026, 11, 18, 15, 0, tzinfo=UTC), "2026-11-18"),
        # Sat 2026-11-21 10:00 UTC -> Friday's session
        (datetime(2026, 11, 21, 10, 0, tzinfo=UTC), "2026-11-20"),
        # Thanksgiving Thu 2026-11-26 (NYSE closed) -> Wednesday's session
        (datetime(2026, 11, 26, 18, 0, tzinfo=UTC), "2026-11-25"),
    ],
)
def test_resolve_report_date_us(moment, expected):
    assert resolve_report_date("us", current_time=moment).isoformat() == expected


def test_resolve_report_date_unknown_market_keeps_legacy_local_date():
    moment = datetime(2026, 11, 18, 3, 6)
    assert resolve_report_date(None, current_time=moment).isoformat() == "2026-11-18"
    assert resolve_report_date("xx", current_time=moment).isoformat() == "2026-11-18"


def test_format_report_timestamp_has_explicit_offset():
    moment = datetime(2026, 11, 18, 3, 6, 35, tzinfo=ZoneInfo("Europe/Madrid"))
    with patch("src.core.trading_calendar.datetime") as fake_dt:
        fake_dt.now.return_value = moment
        assert format_report_timestamp() == "2026-11-18 03:06:35 (UTC+01:00)"
    assert format_report_timestamp("%H:%M", current_time=moment) == "03:06 (UTC+01:00)"
    utc_moment = datetime(2026, 11, 17, 21, 30, tzinfo=UTC)
    assert format_report_timestamp(current_time=utc_moment) == "2026-11-17 21:30:00 (UTC+00:00)"


@pytest.mark.parametrize(
    "summary,expected",
    [
        ({"phase": "postmarket", "session_date": "2026-11-17", "effective_daily_bar_date": "2026-11-17"}, "2026-11-17"),
        ({"phase": "premarket", "session_date": "2026-11-18", "effective_daily_bar_date": "2026-11-17"}, "2026-11-17"),
        ({"phase": "intraday", "session_date": "2026-11-18", "effective_daily_bar_date": "2026-11-17"}, "2026-11-18"),
        ({"phase": "non_trading", "session_date": "2026-11-21", "effective_daily_bar_date": "2026-11-20"}, "2026-11-20"),
        ({"phase": "postmarket", "effective_daily_bar_date": "not-a-date"}, None),
        ({}, None),
        (None, None),
    ],
)
def test_report_date_from_market_phase_summary(summary, expected):
    assert report_date_from_market_phase_summary(summary) == expected


def _result(code, summary=None):
    return SimpleNamespace(code=code, market_phase_summary=summary)


def test_notification_report_date_prefers_phase_summary():
    summary = {"phase": "postmarket", "session_date": "2026-11-17", "effective_daily_bar_date": "2026-11-17"}
    results = [_result("AAPL", summary), _result("MSFT", summary), _result("NVDA", summary)]
    assert NotificationService._resolve_report_date(results) == "2026-11-17"


def test_notification_report_date_uses_calendar_without_summary():
    with patch("src.notification.resolve_report_date") as resolver:
        resolver.return_value = datetime(2026, 11, 17).date()
        assert NotificationService._resolve_report_date([_result("AAPL"), _result("NVDA")]) == "2026-11-17"
        resolver.assert_called_with("us")


def test_notification_report_date_falls_back_on_mixed_session_dates():
    us = {"phase": "postmarket", "effective_daily_bar_date": "2026-11-17"}
    cn = {"phase": "premarket", "effective_daily_bar_date": "2026-11-18"}
    with patch("src.notification.datetime") as fake_dt:
        fake_dt.now.return_value = datetime(2026, 11, 18, 9, 0)
        assert NotificationService._resolve_report_date([_result("AAPL", us), _result("600519", cn)]) == "2026-11-18"


def test_notification_report_date_falls_back_on_unknown_code_and_empty():
    with patch("src.notification.datetime") as fake_dt:
        fake_dt.now.return_value = datetime(2026, 11, 18, 9, 0)
        assert NotificationService._resolve_report_date([_result("???")]) == "2026-11-18"
        assert NotificationService._resolve_report_date([]) == "2026-11-18"


def test_us_market_overview_uses_session_date():
    from src.config import Config
    from src.market_analyzer import MarketAnalyzer

    with patch("src.market_analyzer.DataFetcherManager") as manager, patch(
        "src.market_analyzer.resolve_report_date",
        return_value=datetime(2026, 11, 17).date(),
    ) as resolver:
        manager.return_value.get_main_indices.return_value = []
        overview = MarketAnalyzer(region="us", config=Config()).get_market_overview()
    resolver.assert_called_once_with("us")
    assert overview.date == "2026-11-17"
