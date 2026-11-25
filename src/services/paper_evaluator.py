# -*- coding: utf-8 -*-
"""Evaluate the paper-trading ledger against the daily bars that followed.

Pure, deterministic logic (no I/O): the caller supplies ledger records and
per-ticker daily bars. Rules (see docs/paper-ledger.md):

- Only bars strictly after ``session_date`` are used (no look-ahead).
- buy: limit entry at ``ideal_buy`` valid ``entry_valid_sessions``; fill at the
  open when it gaps below the limit. Exits: stop (open if gapped through),
  target (open if gapped above), stop first when both touch the same bar,
  otherwise close of the last bar after ``max_hold_sessions``.
- hold/watch with a valid plan: same simulation, reported as "shadow".
- sell: direction check over ``direction_sessions`` (correct if price fell).
- Costs: ``cost_bps`` per side on entry and exit.
- A split inside the evaluated window excludes the signal (plan prices are
  pre-split, Yahoo history is split-adjusted).
- Incoherent plans are excluded from trade statistics (status
  ``excluded_plan``): structural action-plan violations recorded by the
  guardrail, a stop not below the secondary entry, or a risk (ideal - stop)
  below ``min_risk_pct`` of the entry, which would inflate R multiples.
"""

from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

GROUP_BUY = "buy"
GROUP_SHADOW = "shadow"
GROUP_DIRECTION = "direction"
SMALL_SAMPLE = 30


@dataclass(frozen=True)
class Bar:
    day: date
    open: float
    high: float
    low: float
    close: float


@dataclass(frozen=True)
class EvalConfig:
    entry_valid_sessions: int = 5
    max_hold_sessions: int = 20
    direction_sessions: int = 10
    cost_bps: float = 10.0
    min_risk_pct: float = 0.5


# Action-plan guardrail codes that make the plan structurally incoherent, so a
# simulated trade (and its R multiple) would not be meaningful.
SEVERE_PLAN_VIOLATIONS = frozenset({
    "entry_range_invalid",
    "stop_loss_not_below_entry_high",
    "stop_loss_not_below_entry_low",
    "stop_loss_not_below_target_price",
    "stop_loss_too_close_to_entry",
    "target_price_not_above_entry_high",
    "target_price_not_above_entry_low",
    "hold_stop_loss_not_below_target_price",
})


EVALUATION_FIELDS = (
    "session_date", "ticker", "decision", "advice", "score", "group", "status",
    "ideal_buy", "stop_loss", "take_profit", "planned_rr",
    "entry_date", "entry_price", "exit_date", "exit_price", "exit_reason", "hold_sessions",
    "return_pct", "r_multiple", "benchmark_return_pct", "spy_return_pct", "direction_correct",
    "plan_issues",
)


# --------------------------------------------------------------------------- ledger


def dedupe_ledger(records: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Keep the last record per (session_date, ticker); stable output order."""
    latest: Dict[tuple, Dict[str, Any]] = {}
    for record in records:
        key = (record.get("session_date"), record.get("ticker"))
        if not all(key):
            continue
        latest[key] = dict(record)
    return [latest[key] for key in sorted(latest)]


def classify(record: Mapping[str, Any]) -> Optional[str]:
    decision = str(record.get("decision") or "").lower()
    if decision == "buy":
        return GROUP_BUY
    if decision == "sell":
        return GROUP_DIRECTION
    if decision == "hold" and _valid_plan(record):
        return GROUP_SHADOW
    return None


def plan_quality_issues(record: Mapping[str, Any], config: EvalConfig = EvalConfig()) -> List[str]:
    """Return the reasons a trade plan is too incoherent to simulate."""
    issues = sorted(
        code for code in (record.get("plan_violations") or []) if code in SEVERE_PLAN_VIOLATIONS
    )
    try:
        ideal = float(record.get("ideal_buy"))
        stop = float(record.get("stop_loss"))
    except (TypeError, ValueError):
        return issues
    secondary = _optional_float(record.get("secondary_buy"))
    if secondary is not None and secondary < ideal and stop >= secondary:
        issues.append("stop_not_below_secondary_entry")
    if ideal > 0 and (ideal - stop) / ideal * 100 < config.min_risk_pct:
        issues.append("risk_below_min_pct")
    return sorted(set(issues))


# --------------------------------------------------------------------------- evaluation


def evaluate_signal(
    record: Mapping[str, Any],
    bars: Sequence[Bar],
    *,
    config: EvalConfig = EvalConfig(),
    split_days: Sequence[date] = (),
    benchmark_bars: Sequence[Bar] = (),
) -> Dict[str, Any]:
    session = _parse_date(record.get("session_date"))
    row: Dict[str, Any] = {key: None for key in EVALUATION_FIELDS}
    for key in ("session_date", "ticker", "decision", "advice", "score", "ideal_buy", "stop_loss", "take_profit"):
        row[key] = record.get(key)
    group = classify(record)
    row["group"] = group
    if group is None:
        row["status"] = "no_plan"
        return row
    if session is None:
        row["status"] = "invalid_date"
        return row

    forward = [bar for bar in sorted(bars, key=lambda b: b.day) if bar.day > session]

    if group == GROUP_DIRECTION:
        return _evaluate_direction(row, forward, config, split_days, session)

    if not _valid_plan(record):
        row["status"] = "invalid_plan"
        return row
    ideal = float(record["ideal_buy"])
    stop = float(record["stop_loss"])
    target = float(record["take_profit"])
    row["planned_rr"] = round((target - ideal) / (ideal - stop), 4)
    issues = plan_quality_issues(record, config)
    if issues:
        row["status"] = "excluded_plan"
        row["plan_issues"] = issues
        return row
    return _evaluate_trade(row, forward, ideal, stop, target, config, split_days, session, benchmark_bars)


def _evaluate_trade(row, forward, ideal, stop, target, config, split_days, session, benchmark_bars):
    entry_index = None
    entry_price = None
    for index, bar in enumerate(forward[: config.entry_valid_sessions]):
        if bar.open <= ideal:
            entry_index, entry_price = index, bar.open
            break
        if bar.low <= ideal:
            entry_index, entry_price = index, ideal
            break
    if entry_index is None:
        row["status"] = "not_filled" if len(forward) >= config.entry_valid_sessions else "pending_entry"
        return row

    entry_bar = forward[entry_index]
    row["entry_date"] = entry_bar.day.isoformat()
    row["entry_price"] = round(entry_price, 4)

    exit_price = exit_reason = exit_index = None
    window = forward[entry_index: entry_index + config.max_hold_sessions]
    for offset, bar in enumerate(window):
        is_entry_bar = offset == 0
        if not is_entry_bar and bar.open <= stop:
            exit_price, exit_reason = bar.open, "stop_gap"
        elif not is_entry_bar and bar.open >= target:
            exit_price, exit_reason = bar.open, "target_gap"
        elif bar.low <= stop and bar.high >= target:
            exit_price, exit_reason = min(stop, entry_price), "ambiguous_stop"
        elif bar.low <= stop:
            exit_price, exit_reason = min(stop, entry_price), "stop"
        elif bar.high >= target and not (is_entry_bar and entry_price == ideal and bar.open > ideal):
            # Filled intraday at the limit: the high may have printed before the
            # fill, so a same-bar target only counts when filled at the open.
            exit_price, exit_reason = target, "target"
        if exit_reason:
            exit_index = entry_index + offset
            break
    if exit_reason is None:
        if len(window) < config.max_hold_sessions:
            row["status"] = "open"
            return row
        exit_index = entry_index + config.max_hold_sessions - 1
        exit_price, exit_reason = forward[exit_index].close, "timeout"

    exit_bar = forward[exit_index]
    if _split_between(split_days, session, exit_bar.day):
        row["status"] = "excluded_split"
        return row

    cost = config.cost_bps / 10_000.0
    entry_eff = entry_price * (1 + cost)
    exit_eff = exit_price * (1 - cost)
    row.update(
        status="closed",
        exit_date=exit_bar.day.isoformat(),
        exit_price=round(exit_price, 4),
        exit_reason=exit_reason,
        hold_sessions=exit_index - entry_index + 1,
        return_pct=round((exit_eff / entry_eff - 1) * 100, 4),
        r_multiple=round((exit_eff - entry_eff) / (ideal - stop), 4),
        benchmark_return_pct=_hold_return(forward, entry_index, exit_index),
        spy_return_pct=_window_return(benchmark_bars, entry_bar.day, exit_bar.day),
    )
    return row


def _evaluate_direction(row, forward, config, split_days, session):
    if len(forward) < config.direction_sessions:
        row["status"] = "open"
        return row
    last = forward[config.direction_sessions - 1]
    if _split_between(split_days, session, last.day):
        row["status"] = "excluded_split"
        return row
    change = (last.close / forward[0].open - 1) * 100
    row.update(
        status="closed",
        entry_date=forward[0].day.isoformat(),
        entry_price=round(forward[0].open, 4),
        exit_date=last.day.isoformat(),
        exit_price=round(last.close, 4),
        exit_reason="direction_window",
        hold_sessions=config.direction_sessions,
        benchmark_return_pct=round(change, 4),
        direction_correct=change < 0,
    )
    return row


# --------------------------------------------------------------------------- summary


def summarize(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}
    for group in (GROUP_BUY, GROUP_SHADOW, GROUP_DIRECTION):
        members = [row for row in rows if row.get("group") == group]
        closed = [row for row in members if row.get("status") == "closed"]
        stats: Dict[str, Any] = {
            "signals": len(members),
            "closed": len(closed),
            "open": sum(1 for row in members if row.get("status") in ("open", "pending_entry")),
            "excluded_plan": sum(1 for row in members if row.get("status") == "excluded_plan"),
            "small_sample": len(closed) < SMALL_SAMPLE,
        }
        if group == GROUP_DIRECTION:
            stats["direction_hit_rate"] = _rate([row.get("direction_correct") for row in closed])
            stats["avg_change_pct"] = _mean(row.get("benchmark_return_pct") for row in closed)
        else:
            attempted = [row for row in members if row.get("status") in ("closed", "open", "not_filled", "excluded_split")]
            filled = [row for row in members if row.get("entry_date")]
            r_values = [float(row["r_multiple"]) for row in closed]
            stats.update(
                fill_rate=round(len(filled) / len(attempted), 4) if attempted else None,
                win_rate=_rate([value > 0 for value in r_values]),
                expectancy_r=_mean(r_values),
                avg_return_pct=_mean(row.get("return_pct") for row in closed),
                avg_hold_sessions=_mean(row.get("hold_sessions") for row in closed),
                max_drawdown_r=_max_drawdown(closed),
                exits={
                    reason: sum(1 for row in closed if row.get("exit_reason") == reason)
                    for reason in ("target", "target_gap", "stop", "stop_gap", "ambiguous_stop", "timeout")
                },
                avg_buy_hold_pct=_mean(row.get("benchmark_return_pct") for row in closed),
                avg_spy_pct=_mean(row.get("spy_return_pct") for row in closed),
            )
        summary[group] = stats
    return summary


def render_summary_markdown(summary: Mapping[str, Mapping[str, Any]], *, as_of: str, config: EvalConfig) -> str:
    def fmt(value: Any, suffix: str = "") -> str:
        if value is None:
            return "N/A"
        if isinstance(value, float):
            return f"{value:.2f}{suffix}"
        return f"{value}{suffix}"

    lines = [
        f"# 📊 Paper Trading Review - {as_of}",
        "",
        f"Rules: limit entry at ideal buy valid {config.entry_valid_sessions} sessions; "
        f"max hold {config.max_hold_sessions} sessions; costs {config.cost_bps:g} bps per side; "
        "stop first when stop and target touch the same day. Paper simulation only.",
        "",
        "| Group | Signals | Closed | Open | Fill rate | Win rate | Expectancy (R) | Avg return | Max DD (R) | Buy&hold | SPY |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {GROUP_BUY: "Buy", GROUP_SHADOW: "Watch/Hold plans (shadow)"}
    for group, label in labels.items():
        s = summary.get(group, {})
        lines.append(
            f"| {label} | {s.get('signals', 0)} | {s.get('closed', 0)} | {s.get('open', 0)} | "
            f"{fmt(_pct(s.get('fill_rate')), '%')} | {fmt(_pct(s.get('win_rate')), '%')} | "
            f"{fmt(s.get('expectancy_r'))} | {fmt(s.get('avg_return_pct'), '%')} | {fmt(s.get('max_drawdown_r'))} | "
            f"{fmt(s.get('avg_buy_hold_pct'), '%')} | {fmt(s.get('avg_spy_pct'), '%')} |"
        )
    direction = summary.get(GROUP_DIRECTION, {})
    lines += [
        "",
        f"**Sell/Reduce direction check ({config.direction_sessions} sessions):** "
        f"{direction.get('closed', 0)} closed, hit rate {fmt(_pct(direction.get('direction_hit_rate')), '%')}, "
        f"avg change {fmt(direction.get('avg_change_pct'), '%')}.",
        "",
    ]
    for group, label in labels.items():
        exits = summary.get(group, {}).get("exits")
        if exits and any(exits.values()):
            lines.append(f"**{label} exits:** " + ", ".join(f"{k} {v}" for k, v in exits.items() if v))
    excluded = {label: summary.get(group, {}).get("excluded_plan", 0) for group, label in labels.items()}
    if any(excluded.values()):
        lines.append(
            "**Excluded (incoherent plan, e.g. stop not below every entry or risk below "
            f"{config.min_risk_pct:g}% of entry):** "
            + ", ".join(f"{label} {count}" for label, count in excluded.items() if count)
        )
    if any(summary.get(g, {}).get("small_sample") for g in summary):
        lines += ["", f"⚠️ Fewer than {SMALL_SAMPLE} closed trades in at least one group: results are not statistically meaningful yet."]
    lines += ["", "For reference only, not investment advice."]
    return "\n".join(lines) + "\n"


def rows_to_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(EVALUATION_FIELDS), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: _csv_value(row.get(key)) for key in EVALUATION_FIELDS})
    return buffer.getvalue()


# --------------------------------------------------------------------------- helpers


def _valid_plan(record: Mapping[str, Any]) -> bool:
    try:
        ideal = float(record.get("ideal_buy"))
        stop = float(record.get("stop_loss"))
        target = float(record.get("take_profit"))
    except (TypeError, ValueError):
        return False
    return all(math.isfinite(v) and v > 0 for v in (ideal, stop, target)) and stop < ideal < target


def _optional_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _parse_date(value: Any) -> Optional[date]:
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def _split_between(split_days: Sequence[date], start: date, end: date) -> bool:
    return any(start < day <= end for day in split_days)


def _hold_return(forward: Sequence[Bar], entry_index: int, exit_index: int) -> Optional[float]:
    start = forward[entry_index].open
    end = forward[exit_index].close
    return round((end / start - 1) * 100, 4) if start else None


def _window_return(bars: Sequence[Bar], start_day: date, end_day: date) -> Optional[float]:
    window = [bar for bar in sorted(bars, key=lambda b: b.day) if start_day <= bar.day <= end_day]
    if not window or not window[0].open:
        return None
    return round((window[-1].close / window[0].open - 1) * 100, 4)


def _mean(values: Iterable[Any]) -> Optional[float]:
    numbers = [float(v) for v in values if v is not None]
    return round(sum(numbers) / len(numbers), 4) if numbers else None


def _rate(flags: Sequence[Any]) -> Optional[float]:
    flags = [bool(flag) for flag in flags if flag is not None]
    return round(sum(flags) / len(flags), 4) if flags else None


def _pct(value: Optional[float]) -> Optional[float]:
    return None if value is None else value * 100


def _max_drawdown(closed: Sequence[Mapping[str, Any]]) -> Optional[float]:
    ordered = sorted(closed, key=lambda row: (row.get("exit_date") or "", row.get("ticker") or ""))
    if not ordered:
        return None
    equity = peak = drawdown = 0.0
    for row in ordered:
        equity += float(row["r_multiple"])
        peak = max(peak, equity)
        drawdown = min(drawdown, equity - peak)
    return round(drawdown, 4)


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return ";".join(str(item) for item in value)
    if isinstance(value, bool):
        return "true" if value else "false"
    return "" if value is None else value
