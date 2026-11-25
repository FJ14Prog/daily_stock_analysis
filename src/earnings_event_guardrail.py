# -*- coding: utf-8 -*-
"""Earnings-event warning (US): gap risk around results announcements.

Two cases, both based on the yfinance earnings calendar
(``financial_report.latest_earnings_date`` / ``next_earnings_date``):

- ``released``: results were published after the analysed session closed
  (typically after hours), so prices and the battle plan do not reflect the
  market reaction yet.
- ``upcoming``: the next results fall within ``EARNINGS_WARNING_SESSIONS``
  sessions (the paper-trading entry window), so entry/stop/target levels may be
  skipped by an overnight gap.

Effects: a risk alert at the top of ``dashboard.intelligence.risk_alerts``,
``dashboard.earnings_event`` metadata, and a High confidence capped at Medium
on buy decisions. The decision itself is never changed.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, time, timedelta
from typing import Any, Dict, List, Mapping, Optional
from zoneinfo import ZoneInfo

from src.report_language import localize_confidence_level, normalize_report_language

logger = logging.getLogger(__name__)

EARNINGS_WARNING_SESSIONS = 5
_NY = ZoneInfo("America/New_York")
_OPEN = time(9, 30)
_CLOSE = time(16, 0)

_TIMING_LABELS: Dict[str, Dict[str, str]] = {
    "en": {
        "before_open": "before the open",
        "after_close": "after the close",
        "intraday": "during the session",
        "unknown": "time not confirmed",
    },
    "zh": {
        "before_open": "盘前",
        "after_close": "盘后",
        "intraday": "盘中",
        "unknown": "时间未确认",
    },
    "ko": {
        "before_open": "장 시작 전",
        "after_close": "장 마감 후",
        "intraday": "장중",
        "unknown": "시간 미확정",
    },
}

_TEMPLATES: Dict[str, Dict[str, str]] = {
    "en": {
        "released": (
            "⚠️ Earnings released {day} {timing}, after the analysed session {session}: prices and the "
            "plan do not reflect the market reaction yet; expect a gap at the next open."
        ),
        "upcoming": (
            "⚠️ Earnings on {day} {timing} (in {sessions} session(s)): overnight gap risk; entry, "
            "stop and target levels may be skipped by a gap."
        ),
    },
    "zh": {
        "released": "⚠️ 财报已于 {day}{timing}发布，晚于本次分析的交易日 {session}：价格与作战计划尚未反映市场反应，下一交易日开盘可能跳空。",
        "upcoming": "⚠️ 财报将于 {day}{timing}发布（{sessions} 个交易日内）：存在隔夜跳空风险，买入、止损与目标价位可能被跳空越过。",
    },
    "ko": {
        "released": "⚠️ 실적이 {day} {timing}에 발표되어 분석 거래일 {session} 이후입니다. 가격과 계획에 시장 반응이 아직 반영되지 않았으며 다음 개장 시 갭이 예상됩니다.",
        "upcoming": "⚠️ 실적 발표 {day} {timing} ({sessions}거래일 이내): 오버나이트 갭 위험이 있어 매수·손절·목표 가격이 갭으로 건너뛰어질 수 있습니다.",
    },
}


def apply_earnings_event_notice(
    result: Any,
    fundamental_context: Optional[Mapping[str, Any]],
    *,
    report_language: Optional[str] = None,
    now: Optional[datetime] = None,
    window_sessions: int = EARNINGS_WARNING_SESSIONS,
) -> List[str]:
    """Attach an earnings gap-risk warning to ``result``; returns adjustment codes."""
    try:
        from src.core.trading_calendar import get_market_for_stock, resolve_report_date

        code = str(getattr(result, "code", "") or "")
        if get_market_for_stock(code) != "us":
            return []
        report = _financial_report(fundamental_context)
        if not report:
            return []
        session = resolve_report_date("us", current_time=now)
        event = detect_earnings_event(report, session, window_sessions=window_sessions)
    except Exception as exc:  # pragma: no cover - never block a report
        logger.warning("[earnings_event] skipped: %s", exc)
        return []
    if event is None:
        return []

    language = normalize_report_language(report_language or getattr(result, "report_language", "en"))
    note = render_earnings_note(event, language)
    dashboard = result.dashboard if isinstance(getattr(result, "dashboard", None), dict) else {}
    result.dashboard = dashboard
    intelligence = dashboard.get("intelligence")
    if not isinstance(intelligence, dict):
        intelligence = {}
        dashboard["intelligence"] = intelligence
    alerts = intelligence.get("risk_alerts")
    if not isinstance(alerts, list):
        alerts = [alerts] if alerts else []
    if note not in alerts:
        alerts.insert(0, note)
    intelligence["risk_alerts"] = alerts
    dashboard["earnings_event"] = dict(event)

    adjustments = ["earnings_event_flagged"]
    decision = str(getattr(result, "decision_type", "") or "").lower()
    confidence = localize_confidence_level(getattr(result, "confidence_level", ""), "en")
    if decision == "buy" and str(confidence).strip().lower() == "high":
        result.confidence_level = localize_confidence_level("medium", language)
        adjustments.append("earnings_event_confidence_capped")
    logger.info("[earnings_event] %s %s", getattr(result, "code", ""), event)
    return adjustments


def detect_earnings_event(
    report: Mapping[str, Any],
    session: date,
    *,
    window_sessions: int = EARNINGS_WARNING_SESSIONS,
) -> Optional[Dict[str, Any]]:
    """Return the most relevant earnings event for ``session`` or None."""
    released = _parse_ts(report.get("latest_earnings_date"))
    if released is not None:
        reaction = _reaction_session(released)
        if reaction is not None and reaction > session:
            return _event("released", released, session, reaction, _sessions_between(session, reaction))

    upcoming = _parse_ts(report.get("next_earnings_date"))
    if upcoming is not None:
        reaction = _reaction_session(upcoming)
        if reaction is not None and reaction > session:
            sessions = _sessions_between(session, reaction)
            if sessions is not None and 1 <= sessions <= window_sessions:
                return _event("upcoming", upcoming, session, reaction, sessions)
    return None


def render_earnings_note(event: Mapping[str, Any], language: str) -> str:
    templates = _TEMPLATES.get(language, _TEMPLATES["en"])
    labels = _TIMING_LABELS.get(language, _TIMING_LABELS["en"])
    return templates[event["kind"]].format(
        day=event["date"],
        timing=labels.get(event["timing"], labels["unknown"]),
        session=event["session"],
        sessions=event["sessions"],
    )


# --------------------------------------------------------------------------- helpers


def _financial_report(context: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not isinstance(context, Mapping):
        return {}
    earnings = context.get("earnings")
    if not isinstance(earnings, Mapping):
        return {}
    data = earnings.get("data") if isinstance(earnings.get("data"), Mapping) else earnings
    report = data.get("financial_report") if isinstance(data, Mapping) else None
    return dict(report) if isinstance(report, Mapping) else {}


def _parse_ts(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_NY)
    return parsed.astimezone(_NY)


def _timing(moment: datetime) -> str:
    clock = moment.time()
    if clock == time(0, 0):
        return "unknown"
    if clock < _OPEN:
        return "before_open"
    if clock >= _CLOSE:
        return "after_close"
    return "intraday"


def _reaction_session(moment: datetime) -> Optional[date]:
    """First session whose prices can reflect the announcement."""
    day = moment.date()
    if _timing(moment) in ("after_close", "unknown"):
        # Unknown time: assume the worst case for a plan made on that day.
        return _next_session(day)
    return day if _is_session(day) else _next_session(day)


def _sessions_between(start: date, end: date) -> Optional[int]:
    """Number of sessions in (start, end]."""
    from src.core.trading_calendar import get_trading_dates

    dates = get_trading_dates("us", start + timedelta(days=1), end)
    if dates is not None:
        return len(dates)
    count, day = 0, start
    while day < end:
        day += timedelta(days=1)
        if day.weekday() < 5:
            count += 1
    return count


def _is_session(day: date) -> bool:
    from src.core.trading_calendar import get_trading_dates

    dates = get_trading_dates("us", day, day)
    return bool(len(dates)) if dates is not None else day.weekday() < 5


def _next_session(day: date) -> Optional[date]:
    from src.core.trading_calendar import get_next_trading_date

    following = get_next_trading_date("us", day)
    if following is not None:
        return following
    day += timedelta(days=1)
    while day.weekday() >= 5:
        day += timedelta(days=1)
    return day


def _event(kind: str, moment: datetime, session: date, reaction: date, sessions: Optional[int]) -> Dict[str, Any]:
    return {
        "kind": kind,
        "date": moment.date().isoformat(),
        "timing": _timing(moment),
        "session": session.isoformat(),
        "reaction_session": reaction.isoformat(),
        "sessions": sessions,
    }
