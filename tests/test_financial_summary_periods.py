# -*- coding: utf-8 -*-
"""Financial summary: one consistent period per row, TTM labels, stale warnings.

Regression for the 2026-11-18/20 reports: NVDA showed Q2 figures after Q3 was
published, and later a row labelled 2026-10-31 with TTM revenue/net income and
the Q2 operating cash flow.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd

from data_provider.yfinance_fundamental_adapter import YfinanceFundamentalAdapter
from src.notification import NotificationService


def _epoch(day: str) -> int:
    return int(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp())


def _ticker(info, income=None, cashflow=None, earnings_dates=None):
    ticker = MagicMock()
    ticker.get_info.return_value = info
    ticker.info = info
    ticker.quarterly_income_stmt = income if income is not None else pd.DataFrame()
    ticker.quarterly_cashflow = cashflow if cashflow is not None else pd.DataFrame()
    ticker.dividends = pd.Series(dtype="float64")
    ticker.get_earnings_dates.return_value = earnings_dates
    return ticker


def _bundle(ticker):
    with patch("yfinance.Ticker", return_value=ticker):
        return YfinanceFundamentalAdapter().get_fundamental_bundle("NVDA")


_NVDA_INFO = {
    "financialCurrency": "USD",
    "currency": "USD",
    "totalRevenue": 361.81e9,  # TTM
    "operatingCashflow": 150e9,  # TTM
    "profitMargins": 0.6254,
    "returnOnEquity": 1.1736,
    "grossMargins": 0.7492,
    "mostRecentQuarter": _epoch("2026-10-25"),
}

_EARNINGS_DATES = pd.DataFrame(
    {"EPS Estimate": [1.5, 1.2, 1.1], "Reported EPS": [None, 1.31, 1.05]},
    index=pd.DatetimeIndex(
        ["2027-02-24 16:20", "2026-11-17 16:20", "2026-08-26 16:20"], tz="America/New_York"
    ),
)


def _nvda_income(newest_empty: bool = True) -> pd.DataFrame:
    columns = {
        pd.Timestamp("2026-07-31"): {"Total Revenue": 96.22e9, "Net Income": 59.69e9, "Gross Profit": 72.0e9},
        pd.Timestamp("2026-04-30"): {"Total Revenue": 80.0e9, "Net Income": 45.0e9, "Gross Profit": 60.0e9},
    }
    if newest_empty:
        columns[pd.Timestamp("2026-10-31")] = {
            "Total Revenue": float("nan"),
            "Net Income": float("nan"),
            "Gross Profit": float("nan"),
        }
    return pd.DataFrame(columns)


def test_empty_new_quarter_column_does_not_mix_periods() -> None:
    cashflow = pd.DataFrame({pd.Timestamp("2026-07-31"): {"Operating Cash Flow": 24.08e9}})
    bundle = _bundle(_ticker(_NVDA_INFO, _nvda_income(), cashflow, _EARNINGS_DATES))

    report = bundle["earnings"]["financial_report"]
    assert report["report_date"] == "2026-07-31"
    assert report["period_basis"] == "quarter"
    assert report["revenue"] == 96.22e9  # same quarter, not TTM 361.81B
    assert report["net_profit_parent"] == 59.69e9
    assert report["operating_cash_flow"] == 24.08e9
    assert report["stale"] is True
    assert report["latest_quarter_end"] == "2026-10-25"
    assert report["latest_earnings_date"].startswith("2026-11-17")
    # Gross margin from the same quarter: 72.0 / 96.22
    assert round(bundle["growth"]["gross_margin"], 2) == 74.83
    assert report["gross_margin_basis"] == "quarter"
    assert report["roe_basis"] == "ttm"


def test_announced_results_flag_stale_even_without_new_column() -> None:
    info = dict(_NVDA_INFO)
    info.pop("mostRecentQuarter")
    bundle = _bundle(_ticker(info, _nvda_income(newest_empty=False), None, _EARNINGS_DATES))
    report = bundle["earnings"]["financial_report"]
    assert report["report_date"] == "2026-07-31"
    assert report["stale"] is True  # announced 2026-11-17, 109 days after quarter end
    assert report["latest_quarter_end"] is None


def test_cash_flow_from_another_quarter_is_not_used() -> None:
    cashflow = pd.DataFrame({pd.Timestamp("2026-04-30"): {"Operating Cash Flow": 20e9}})
    bundle = _bundle(_ticker(_NVDA_INFO, _nvda_income(newest_empty=False), cashflow, None))
    report = bundle["earnings"]["financial_report"]
    assert report["operating_cash_flow"] is None  # neither Q1 nor TTM


def test_fresh_statements_are_not_stale() -> None:
    info = dict(_NVDA_INFO, mostRecentQuarter=_epoch("2026-07-26"))
    dates = pd.DataFrame(
        {"Reported EPS": [1.05]},
        index=pd.DatetimeIndex(["2026-08-26 16:20"], tz="America/New_York"),
    )
    bundle = _bundle(_ticker(info, _nvda_income(newest_empty=False), None, dates))
    report = bundle["earnings"]["financial_report"]
    assert report["stale"] is False


def test_no_statements_falls_back_to_full_ttm_row() -> None:
    bundle = _bundle(_ticker(_NVDA_INFO))
    report = bundle["earnings"]["financial_report"]
    assert report["period_basis"] == "ttm"
    assert report["report_date"] == "2026-10-25"
    assert report["revenue"] == 361.81e9
    assert report["operating_cash_flow"] == 150e9
    assert report["stale"] is False


def test_bank_zero_gross_margin_is_unavailable() -> None:
    # Yahoo reports grossMargins = 0.0 for banks (JPM 2026-11-24 showed "0.00% (TTM)").
    bank = dict(_NVDA_INFO, grossMargins=0.0, industry="Banks - Diversified")
    growth = _bundle(_ticker(bank))["growth"]
    assert growth["gross_margin"] is None
    assert growth["roe"] is not None
    # Banks are skipped even if Yahoo ever returns a non-zero value.
    assert _bundle(_ticker(dict(bank, grossMargins=0.41)))["growth"]["gross_margin"] is None
    # Exact zero is unavailable for any company; normal values are kept.
    assert _bundle(_ticker(dict(_NVDA_INFO, grossMargins=0.0)))["growth"]["gross_margin"] is None
    assert _bundle(_ticker(_NVDA_INFO))["growth"]["gross_margin"] == 74.92


# --------------------------------------------------------------------------- rendering


def _render(report, growth, language="en"):
    service = NotificationService.__new__(NotificationService)
    labels = __import__("src.report_language", fromlist=["get_report_labels"]).get_report_labels(language)
    lines: list[str] = []
    service._append_financial_summary(
        lines, {"financial_report": report, "growth": growth}, labels, report_language=language
    )
    return "\n".join(lines)


def test_render_labels_ttm_ratios_and_stale_warning() -> None:
    report = {
        "report_date": "2026-07-31",
        "revenue": 96.22e9,
        "net_profit_parent": 59.69e9,
        "operating_cash_flow": 24.08e9,
        "roe": 117.36,
        "currency": "USD",
        "period_basis": "quarter",
        "roe_basis": "ttm",
        "gross_margin_basis": "ttm",
        "stale": True,
        "latest_quarter_end": "2026-10-25",
        "latest_earnings_date": "2026-11-17T16:20:00-05:00",
    }
    text = _render(report, {"gross_margin": 74.92})
    assert "117.36% (TTM)" in text
    assert "74.92% (TTM)" in text
    assert (
        "⚠️ Newer results are not yet reflected in these figures (quarter ending 2026-10-25, "
        "published 2026-11-17); figures above are for the quarter ending 2026-07-31."
    ) in text


def test_render_ttm_row() -> None:
    report = {"report_date": "2026-10-25", "revenue": 361.81e9, "currency": "USD", "period_basis": "ttm", "roe": 117.36, "roe_basis": "ttm"}
    text = _render(report, {})
    assert "| TTM (2026-10-25) |" in text
    assert "117.36% (TTM)" not in text  # whole row already TTM


def test_render_without_basis_fields_is_unchanged() -> None:
    report = {"report_date": "2026-09-30", "revenue": 1.0e9, "roe": 12.5, "currency": "CNY"}
    text = _render(report, {"gross_margin": 30.0}, language="zh")
    assert "| 2026-09-30 |" in text
    assert "(TTM)" not in text
    assert "⚠️" not in text


# --------------------------------------------------------------------------- LLM prompt


def test_prompt_marks_stale_statements_and_basis() -> None:
    from types import SimpleNamespace

    try:
        import litellm  # noqa: F401
    except ModuleNotFoundError:  # pragma: no cover
        from tests.litellm_stub import ensure_litellm_stub

        ensure_litellm_stub()
    from src.analyzer import GeminiAnalyzer

    with patch.object(GeminiAnalyzer, "_init_litellm", return_value=None):
        analyzer = GeminiAnalyzer()
    context = {
        "code": "NVDA",
        "stock_name": "NVIDIA",
        "date": "2026-11-18",
        "today": {},
        "fundamental_context": {
            "earnings": {
                "data": {
                    "financial_report": {
                        "report_date": "2026-07-31",
                        "revenue": 96.22e9,
                        "roe": 117.36,
                        "period_basis": "quarter",
                        "roe_basis": "ttm",
                        "stale": True,
                        "latest_quarter_end": "2026-10-25",
                        "latest_earnings_date": "2026-11-17T16:20:00-05:00",
                    }
                }
            }
        },
    }
    fake_cfg = SimpleNamespace(news_max_age_days=30, news_strategy_profile="medium")
    with patch("src.analyzer.get_config", return_value=fake_cfg):
        prompt = analyzer._format_prompt(context, "NVIDIA", news_context="news", report_language="en")

    assert "| Latest Report Period | 2026-07-31 | Single quarter |" in prompt
    assert "| ROE | 117.36 | TTM |" in prompt
    assert "These statements are NOT the latest" in prompt
    assert "quarter ending 2026-10-25, announced 2026-11-17" in prompt
