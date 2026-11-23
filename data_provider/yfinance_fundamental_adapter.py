# -*- coding: utf-8 -*-
"""
Yfinance fundamental adapter for HK/US markets (fail-open).

Mirrors the bundle shape of `AkshareFundamentalAdapter.get_fundamental_bundle`
so it can be plugged into `data_provider.base.get_fundamental_context()`
without changing downstream consumers. Adds HK/US-specific fields:

- ``earnings.financial_report.currency`` — financial statement currency
  (``USD`` / ``HKD`` / ``CNY``) from ``info.financialCurrency``. For HK ADRs
  yfinance commonly reports ``financialCurrency=CNY`` while trades settle in
  HKD, so this differs from the dividend currency below.
- ``earnings.dividend.currency`` — trading / dividend currency from
  ``info.currency`` (e.g. HKD for 0700.HK). Used to suffix 港元/美元/元 for
  per-share cash dividends and to scope the TTM yield denominator.
- ``earnings.dividend.ttm_dividend_yield_pct`` — computed as
  ``ttm_cash_dividend_per_share / latest_price * 100``, both sides in the
  trading currency (info.currentPrice/regularMarketPrice/previousClose).
  ``info.dividendYield`` is only used as a last-resort fallback and is
  passed through as-is (current yfinance reports it in percent units).
- ``belong_boards`` — derived from ``info.sector`` + ``info.industry``; the CN
  pipeline derives it from AkShare 板块名单, this is the HK/US analogue.

This adapter intentionally treats every yfinance call as best-effort and never
raises to caller. Partial data is allowed; downstream `_infer_block_status` will
mark the block as ``partial`` when only some fields are populated.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


_INCOME_REVENUE_KEYS = ("Total Revenue", "TotalRevenue", "Revenue")
_INCOME_NET_PROFIT_KEYS = (
    "Net Income Common Stockholders",
    "Net Income From Continuing Operation Net Minority Interest",
    "Net Income",
    "NetIncome",
)
_INCOME_GROSS_PROFIT_KEYS = ("Gross Profit", "GrossProfit")
# A reported quarter is announced within ~60 days of its end (10-Q deadline is
# 40-45 days). An announcement further than this from the statement quarter
# means a newer quarter has been published.
_EARNINGS_ANNOUNCEMENT_MAX_LAG_DAYS = 75
# Yahoo labels some fiscal quarters with the calendar month end (NVDA
# 2026-07-26 -> 2026-07-31); allow that slack when matching quarters.
_QUARTER_MATCH_TOLERANCE_DAYS = 10
_CASHFLOW_OP_KEYS = (
    "Operating Cash Flow",
    "Cash Flow From Continuing Operating Activities",
    "Total Cash From Operating Activities",
)


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result:  # NaN guard
        return None
    return result


def _ratio_to_pct(value: Any) -> Optional[float]:
    """yfinance returns ratios as decimal (0.166 = 16.6%); convert to percent."""
    raw = _safe_float(value)
    if raw is None:
        return None
    return round(raw * 100.0, 4)


def _pick_row(df: pd.DataFrame, keys) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    for key in keys:
        if key in df.index:
            try:
                return df.loc[key]
            except KeyError:
                continue
    return None


def _latest_value(row: Optional[pd.Series]) -> Optional[float]:
    if row is None or row.empty:
        return None
    try:
        return _safe_float(row.iloc[0])
    except IndexError:
        return None


def _date_indexed(row: Optional[pd.Series]) -> Optional[pd.Series]:
    """Return the row indexed by Timestamp, newest first; None when unusable."""
    if row is None or row.empty:
        return None
    try:
        dates = pd.to_datetime(row.index)
    except (TypeError, ValueError):
        return None
    return pd.Series(row.values, index=dates).sort_index(ascending=False)


def _newest_column_date(df: pd.DataFrame) -> Optional[pd.Timestamp]:
    try:
        dates = pd.to_datetime(list(df.columns), errors="coerce")
    except (TypeError, ValueError):
        return None
    dates = [d for d in dates if pd.notna(d)]
    return max(dates) if dates else None


def _latest_populated_date(row: Optional[pd.Series]) -> Optional[pd.Timestamp]:
    if row is None:
        return None
    for date, value in row.items():
        if _safe_float(value) is not None:
            return date
    return None


def _value_at(row: Optional[pd.Series], date: pd.Timestamp) -> Optional[float]:
    if row is None or date not in row.index:
        return None
    value = row[date]
    if isinstance(value, pd.Series):
        value = value.iloc[0]
    return _safe_float(value)


def _value_near(row: Optional[pd.Series], date: pd.Timestamp) -> Optional[float]:
    """Value for the same quarter (date within tolerance); None otherwise."""
    if row is None:
        return None
    for candidate, value in row.items():
        if abs((candidate - date).days) <= _QUARTER_MATCH_TOLERANCE_DAYS:
            parsed = _safe_float(value)
            if parsed is not None:
                return parsed
    return None


def _latest_earnings_announcement(ticker: Any, errors: list) -> Optional[datetime]:
    """Most recent past earnings announcement (with reported EPS), best effort."""
    getter = getattr(ticker, "get_earnings_dates", None)
    if getter is None:
        return None
    try:
        frame = getter(limit=8)
    except Exception as exc:
        errors.append(f"earnings_dates:{type(exc).__name__}")
        return None
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return None
    now = pd.Timestamp.now(tz="UTC")
    latest = None
    for raw_date, row in frame.iterrows():
        try:
            ts = pd.Timestamp(raw_date)
        except (TypeError, ValueError):
            continue
        ts_utc = ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")
        if ts_utc > now:
            continue
        reported = row.get("Reported EPS") if hasattr(row, "get") else None
        if _safe_float(reported) is None:
            continue
        if latest is None or ts_utc > latest[0]:
            latest = (ts_utc, ts)
    return latest[1].to_pydatetime() if latest else None


def _statement_freshness(
    ticker: Any,
    info: Dict[str, Any],
    *,
    report_date: Optional[str],
    newest_statement_date: Optional[pd.Timestamp],
    errors: list,
) -> Dict[str, Any]:
    """Flag quarterly statements that lag behind already-published results."""
    announcement = _latest_earnings_announcement(ticker, errors)
    freshness: Dict[str, Any] = {
        "latest_earnings_date": announcement.isoformat() if announcement else None,
        "stale": False,
        "latest_quarter_end": None,
    }
    if not report_date:
        return freshness
    period = pd.Timestamp(report_date)
    tolerance = pd.Timedelta(days=_QUARTER_MATCH_TOLERANCE_DAYS)

    newer_quarter_ends = []
    if newest_statement_date is not None and newest_statement_date > period + tolerance:
        newer_quarter_ends.append(newest_statement_date)
    most_recent_quarter = _epoch_to_date(info.get("mostRecentQuarter"))
    if most_recent_quarter and pd.Timestamp(most_recent_quarter) > period + tolerance:
        newer_quarter_ends.append(pd.Timestamp(most_recent_quarter))

    announced_newer = False
    if announcement is not None:
        announced_day = pd.Timestamp(announcement.date())
        announced_newer = (announced_day - period).days > _EARNINGS_ANNOUNCEMENT_MAX_LAG_DAYS

    if newer_quarter_ends or announced_newer:
        freshness["stale"] = True
        if newer_quarter_ends:
            freshness["latest_quarter_end"] = min(newer_quarter_ends).date().isoformat()
    return freshness


def _yoy_from_row(row: Optional[pd.Series]) -> Optional[float]:
    """Statement-derived YoY: requires the same quarter one year earlier.

    The prior-year quarter is matched by date (within 45 days), not by
    position, so a quarter missing from the statement cannot shift the base.
    yfinance ``quarterly_*_stmt`` returns 4 quarters by default, so this
    typically returns None and callers fall back to ``info.revenueGrowth`` /
    ``info.earningsGrowth`` (already TTM YoY ratios). Doing QoQ via ``iloc[1]``
    is wrong for seasonal businesses — explicitly refuse it.
    """
    if row is None or row.empty:
        return None
    try:
        dates = pd.to_datetime(row.index)
    except (TypeError, ValueError):
        return None
    series = pd.Series(row.values, index=dates).sort_index(ascending=False)
    # Skip quarters whose value is not yet populated (new empty columns).
    series = series[[_safe_float(v) is not None for v in series.values]]
    if series.empty:
        return None
    latest_date = series.index[0]
    latest = _safe_float(series.iloc[0])
    target = latest_date - pd.DateOffset(years=1)
    candidates = [(abs((d - target).days), d) for d in series.index[1:] if abs((d - target).days) <= 45]
    if not candidates:
        return None
    prev_year = _safe_float(series[min(candidates)[1]])
    if latest is None or prev_year in (None, 0):
        return None
    return round((latest - prev_year) / abs(prev_year) * 100.0, 4)


def _epoch_to_date(value: Any) -> Optional[str]:
    raw = _safe_float(value)
    if raw is None:
        return None
    try:
        return datetime.fromtimestamp(raw, tz=timezone.utc).date().isoformat()
    except (OverflowError, OSError, ValueError):
        return None


def _convert_to_yf_symbol(stock_code: str) -> str:
    """Convert internal code to yfinance ticker. Lightweight inline reproduction
    of YFinanceFetcher._convert_stock_code to avoid pulling the full fetcher
    into the fundamental path.
    """
    code = (stock_code or "").strip().upper()
    if not code:
        return code
    if code.startswith("HK"):
        digits = code[2:].lstrip("0") or "0"
        return f"{digits.zfill(4)}.HK"
    if "." in code:
        return code
    # Assume US ticker by default for non-HK / non-CN callers
    return code


class YfinanceFundamentalAdapter:
    """HK/US fundamental adapter backed by yfinance.

    Returns the same bundle keys as :class:`AkshareFundamentalAdapter` so the
    aggregation in :func:`data_provider.base.get_fundamental_context` can stay
    market-agnostic.
    """

    def get_fundamental_bundle(self, stock_code: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "status": "not_supported",
            "growth": {},
            "earnings": {},
            "institution": {},
            "boards": {},
            "belong_boards": [],
            "source_chain": [],
            "errors": [],
        }

        try:
            import yfinance as yf
        except Exception as exc:
            result["errors"].append(f"import_yfinance:{type(exc).__name__}")
            return result

        symbol = _convert_to_yf_symbol(stock_code)
        if not symbol:
            result["errors"].append("empty_symbol")
            return result

        ticker = yf.Ticker(symbol)
        info: Dict[str, Any] = {}
        try:
            info = ticker.get_info() if hasattr(ticker, "get_info") else (ticker.info or {})
            if not isinstance(info, dict):
                info = {}
        except Exception as exc:
            result["errors"].append(f"info:{type(exc).__name__}:{exc}")
            info = {}

        # Financial statements (income/cashflow) are reported in `financialCurrency`;
        # for HK ADRs that is often CNY even when the stock trades in HKD. Dividends
        # and live price are paid/quoted in `currency` — keep them separate so the
        # renderer can suffix per-block currency tags correctly.
        financial_currency = str(info.get("financialCurrency") or info.get("currency") or "").upper() or None
        dividend_currency = str(info.get("currency") or info.get("financialCurrency") or "").upper() or None

        # ---------------- growth block ----------------
        growth_payload: Dict[str, Any] = {
            "revenue_yoy": _ratio_to_pct(info.get("revenueGrowth")),
            "net_profit_yoy": _ratio_to_pct(info.get("earningsGrowth")),
            "roe": _ratio_to_pct(info.get("returnOnEquity")),
            "gross_margin": _ratio_to_pct(info.get("grossMargins")),
        }
        if any(v is not None for v in growth_payload.values()):
            result["growth"] = growth_payload
            result["source_chain"].append("growth:yfinance.info")

        # ---------------- financial_report ----------------
        # All statement figures come from ONE quarter column: the most recent
        # quarter whose revenue is populated. Yahoo often adds a new quarter
        # column before its values are filled; using that column's date with
        # TTM fallbacks produced rows mixing three periods. Cash flow is only
        # taken when the cash-flow statement has the same quarter.
        report_date: Optional[str] = None
        statement_basis: Optional[str] = None
        revenue_latest: Optional[float] = None
        net_profit_latest: Optional[float] = None
        operating_cash_flow_latest: Optional[float] = None
        quarter_gross_margin: Optional[float] = None
        revenue_row = None
        net_profit_row = None
        newest_statement_date: Optional[pd.Timestamp] = None

        try:
            income_df = ticker.quarterly_income_stmt
        except Exception as exc:
            result["errors"].append(f"quarterly_income_stmt:{type(exc).__name__}")
            income_df = None
        if isinstance(income_df, pd.DataFrame) and not income_df.empty:
            revenue_row = _date_indexed(_pick_row(income_df, _INCOME_REVENUE_KEYS))
            net_profit_row = _date_indexed(_pick_row(income_df, _INCOME_NET_PROFIT_KEYS))
            gross_profit_row = _date_indexed(_pick_row(income_df, _INCOME_GROSS_PROFIT_KEYS))
            newest_statement_date = _newest_column_date(income_df)
            period = _latest_populated_date(revenue_row)
            if period is not None:
                report_date = period.date().isoformat()
                statement_basis = "quarter"
                revenue_latest = _value_at(revenue_row, period)
                net_profit_latest = _value_at(net_profit_row, period)
                gross_profit = _value_at(gross_profit_row, period)
                if gross_profit is not None and revenue_latest:
                    quarter_gross_margin = round(gross_profit / revenue_latest * 100.0, 4)

                try:
                    cashflow_df = ticker.quarterly_cashflow
                except Exception as exc:
                    result["errors"].append(f"quarterly_cashflow:{type(exc).__name__}")
                    cashflow_df = None
                if isinstance(cashflow_df, pd.DataFrame) and not cashflow_df.empty:
                    cashflow_row = _date_indexed(_pick_row(cashflow_df, _CASHFLOW_OP_KEYS))
                    operating_cash_flow_latest = _value_near(cashflow_row, period)

        # Fallback to TTM aggregates from .info only when no quarterly statement
        # is usable; the whole row is then TTM, never mixed with a quarter.
        if statement_basis is None:
            revenue_latest = _safe_float(info.get("totalRevenue"))
            operating_cash_flow_latest = _safe_float(info.get("operatingCashflow"))
            if revenue_latest is not None:
                margin = _safe_float(info.get("profitMargins"))
                if margin is not None:
                    net_profit_latest = revenue_latest * margin
            if revenue_latest is not None or operating_cash_flow_latest is not None:
                statement_basis = "ttm"
                report_date = _epoch_to_date(info.get("mostRecentQuarter"))

        # Statement-derived YoY (requires the same quarter a year earlier) is preferred
        # over .info ratios; otherwise keep the TTM growth values already set
        # from info.revenueGrowth / info.earningsGrowth above. Refuse QoQ
        # fallback — it produces misleading numbers for seasonal businesses.
        statement_revenue_yoy = _yoy_from_row(revenue_row)
        statement_net_profit_yoy = _yoy_from_row(net_profit_row)
        if statement_revenue_yoy is not None:
            growth_payload["revenue_yoy"] = statement_revenue_yoy
        if statement_net_profit_yoy is not None:
            growth_payload["net_profit_yoy"] = statement_net_profit_yoy
        gross_margin_basis = None
        if quarter_gross_margin is not None:
            growth_payload["gross_margin"] = quarter_gross_margin
            gross_margin_basis = "quarter"
        elif growth_payload.get("gross_margin") is not None:
            gross_margin_basis = "ttm"
        if any(v is not None for v in growth_payload.values()):
            result["growth"] = growth_payload

        freshness = _statement_freshness(
            ticker,
            info,
            report_date=report_date if statement_basis == "quarter" else None,
            newest_statement_date=newest_statement_date,
            errors=result["errors"],
        )

        financial_report = {
            "report_date": report_date,
            "revenue": revenue_latest,
            "net_profit_parent": net_profit_latest,
            "operating_cash_flow": operating_cash_flow_latest,
            "roe": growth_payload.get("roe"),
            "currency": financial_currency,
        }
        if any(v is not None and v != "" for v in financial_report.values()):
            financial_report["period_basis"] = statement_basis
            financial_report["roe_basis"] = "ttm" if growth_payload.get("roe") is not None else None
            financial_report["gross_margin_basis"] = gross_margin_basis
            financial_report.update(freshness)
            result.setdefault("earnings", {})["financial_report"] = financial_report
            result["source_chain"].append("earnings.financial_report:yfinance")

        # ---------------- dividend block ----------------
        events: List[Dict[str, Any]] = []
        as_of_date = datetime.now(timezone.utc).date()
        try:
            div_series = ticker.dividends
        except Exception as exc:
            result["errors"].append(f"dividends:{type(exc).__name__}")
            div_series = None
        if div_series is not None and not div_series.empty:
            # yfinance 1.2.x returns Ticker.dividends as a single-column DataFrame, not a
            # Series; coerce so `.items()` yields (timestamp, value) rather than
            # (column_name, Series). Otherwise every event is dropped (`_safe_float(Series)`
            # -> None) and TTM silently falls back to the annual-rate estimate.
            if hasattr(div_series, "columns"):
                div_series = div_series.iloc[:, 0]
            try:
                cutoff = pd.Timestamp(as_of_date).tz_localize(div_series.index.tz) - pd.Timedelta(days=365)
                for ts, value in div_series.items():
                    per_share = _safe_float(value)
                    if per_share is None or per_share <= 0:
                        continue
                    try:
                        event_date = pd.Timestamp(ts).date().isoformat()
                    except Exception:
                        continue
                    events.append({
                        "event_date": event_date,
                        "ex_dividend_date": event_date,
                        "record_date": None,
                        "announcement_date": None,
                        "cash_dividend_per_share": per_share,
                        "is_pre_tax": True,
                    })
                ttm_events = []
                for item in events:
                    try:
                        event_ts = pd.Timestamp(item["event_date"]).tz_localize(div_series.index.tz)
                    except Exception:
                        continue
                    if event_ts >= cutoff:
                        ttm_events.append(item)
            except Exception as exc:
                result["errors"].append(f"dividend_window:{type(exc).__name__}")
                ttm_events = []
        else:
            ttm_events = []

        ttm_cash = sum(item["cash_dividend_per_share"] for item in ttm_events) if ttm_events else None
        if ttm_cash is None:
            ttm_cash = _safe_float(info.get("trailingAnnualDividendRate"))

        if events or ttm_cash is not None:
            events.sort(key=lambda item: item.get("event_date") or "", reverse=True)
            dividend_payload: Dict[str, Any] = {
                "events": events[:5],
                "ttm_event_count": len(ttm_events),
                "ttm_cash_dividend_per_share": round(ttm_cash, 6) if ttm_cash is not None else None,
                "coverage": "cash_dividend_pre_tax",
                "currency": dividend_currency,
                "as_of": as_of_date.isoformat(),
            }

            # Yield: prefer recomputing from TTM cash / latest price so the
            # numerator and denominator are consistent (and both in the trading
            # currency). yfinance's `info.dividendYield` is now reported in
            # percent units, but past versions returned a ratio and some ADR
            # payloads still drift — keep it as a last-resort passthrough only.
            latest_price = (
                _safe_float(info.get("currentPrice"))
                or _safe_float(info.get("regularMarketPrice"))
                or _safe_float(info.get("previousClose"))
            )
            yield_pct: Optional[float] = None
            if ttm_cash is not None and latest_price not in (None, 0):
                yield_pct = round(float(ttm_cash) / float(latest_price) * 100.0, 4)
            elif _safe_float(info.get("trailingAnnualDividendYield")) is not None:
                yield_pct = _ratio_to_pct(info.get("trailingAnnualDividendYield"))
            else:
                raw_yield = _safe_float(info.get("dividendYield"))
                if raw_yield is not None:
                    # Pass through as-is; current yfinance already returns percent.
                    yield_pct = round(raw_yield, 4)
            if yield_pct is not None:
                dividend_payload["ttm_dividend_yield_pct"] = yield_pct
            result.setdefault("earnings", {})["dividend"] = dividend_payload
            result["source_chain"].append("earnings.dividend:yfinance")

        # ---------------- belong_boards (sector + industry) ----------------
        belong_boards: List[Dict[str, Any]] = []
        sector_name = str(info.get("sector") or info.get("sectorDisp") or "").strip()
        if sector_name:
            belong_boards.append({"name": sector_name, "type": "行业"})
        industry_name = str(info.get("industry") or info.get("industryDisp") or "").strip()
        if industry_name and industry_name != sector_name:
            belong_boards.append({"name": industry_name, "type": "概念"})
        if belong_boards:
            result["belong_boards"] = belong_boards
            result["source_chain"].append("belong_boards:yfinance.info")

        has_content = bool(
            result.get("growth")
            or result.get("earnings")
            or result.get("belong_boards")
        )
        result["status"] = "partial" if has_content else "not_supported"
        return result
