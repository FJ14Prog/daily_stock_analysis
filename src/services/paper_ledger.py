# -*- coding: utf-8 -*-
"""Append-only paper-trading ledger of daily signals.

Each analysis run appends one JSON line per stock to ``PAPER_LEDGER_FILE`` so
signals can later be evaluated against the bars that followed (no look-ahead:
only data known when the report was generated is recorded).

Disabled unless ``PAPER_LEDGER_FILE`` is set. Failures are logged and never
break the analysis run.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.utils.sniper_points import extract_sniper_points

logger = logging.getLogger(__name__)

LEDGER_SCHEMA_VERSION = 1
LEDGER_ENV_VAR = "PAPER_LEDGER_FILE"


def ledger_path_from_env() -> Optional[Path]:
    raw = str(os.getenv(LEDGER_ENV_VAR) or "").strip()
    return Path(raw) if raw else None


def build_ledger_record(
    result: Any,
    *,
    recorded_at: Optional[datetime] = None,
    run_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Build one ledger line from a finished AnalysisResult; None if unusable."""

    if result is None or not getattr(result, "success", True):
        return None
    code = str(getattr(result, "code", "") or "").strip()
    if not code:
        return None

    points = extract_sniper_points(result)
    ideal = points.get("ideal_buy")
    stop = points.get("stop_loss")
    target = points.get("take_profit")
    price = _number(getattr(result, "current_price", None))

    risk_reward = None
    if ideal is not None and stop is not None and target is not None and stop < ideal < target:
        risk_reward = round((target - ideal) / (ideal - stop), 4)
    entry_distance_pct = None
    if ideal is not None and price:
        entry_distance_pct = round((ideal - price) / price * 100, 4)

    dashboard = getattr(result, "dashboard", None)
    dashboard = dashboard if isinstance(dashboard, dict) else {}
    validation = dashboard.get("action_plan_validation")
    validation = validation if isinstance(validation, dict) else {}

    return {
        "schema": LEDGER_SCHEMA_VERSION,
        "session_date": _session_date(code),
        "recorded_at": (recorded_at or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_id": run_id,
        "ticker": code,
        "name": getattr(result, "name", None),
        "decision": getattr(result, "decision_type", None),
        "action": getattr(result, "action", None),
        "advice": getattr(result, "operation_advice", None),
        "score": getattr(result, "sentiment_score", None),
        "confidence": getattr(result, "confidence_level", None),
        "price": price,
        "change_pct": _number(getattr(result, "change_pct", None)),
        "ideal_buy": ideal,
        "secondary_buy": points.get("secondary_buy"),
        "stop_loss": stop,
        "take_profit": target,
        "risk_reward": risk_reward,
        "entry_distance_pct": entry_distance_pct,
        "plan_violations": list(validation.get("violations") or []),
        "downgraded": bool(validation.get("downgraded", False)),
        "earnings_event": _earnings_event(dashboard),
        "model": getattr(result, "model_used", None),
        "llm_fallback": bool(getattr(result, "llm_fallback", None)),
        "language": getattr(result, "report_language", None),
    }


def append_ledger_records(results: Iterable[Any], path: Optional[Path] = None) -> int:
    """Append one line per result to the ledger. Returns lines written."""

    target = path if path is not None else ledger_path_from_env()
    if target is None:
        return 0
    try:
        run_id = os.getenv("GITHUB_RUN_ID") or None
        recorded_at = datetime.now(timezone.utc)
        records: List[Dict[str, Any]] = []
        for result in results or []:
            record = build_ledger_record(result, recorded_at=recorded_at, run_id=run_id)
            if record is not None:
                records.append(record)
        if not records:
            return 0
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        logger.info("[paper_ledger] appended %d record(s) to %s", len(records), target)
        return len(records)
    except Exception as exc:
        logger.warning("[paper_ledger] write skipped: %s", exc)
        return 0


def _session_date(code: str) -> Optional[str]:
    try:
        from src.core.trading_calendar import get_market_for_stock, resolve_report_date

        return resolve_report_date(get_market_for_stock(code)).isoformat()
    except Exception:
        return None


def _earnings_event(dashboard: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    event = dashboard.get("earnings_event")
    if not isinstance(event, dict):
        return None
    return {key: event.get(key) for key in ("kind", "date", "timing", "sessions")}


def _number(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def ledger_has_session(path: Path, session_date: str) -> bool:
    """Return True when the ledger already holds a signal for ``session_date``.

    Used by the scheduled-run guard so a backup cron does not analyse the same
    session twice. Unreadable or malformed lines are ignored; a missing file
    means "not recorded yet".
    """
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if isinstance(record, dict) and record.get("session_date") == session_date:
                    return True
    except OSError:
        return False
    return False
