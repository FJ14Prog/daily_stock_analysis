# -*- coding: utf-8 -*-
"""Deterministic validation of the report action plan (battle_plan).

The LLM writes ``battle_plan.sniper_points`` and ``position_strategy`` as free
text.  This guardrail parses the price levels and checks that the plan is
internally coherent before the report is rendered and persisted:

- R1 price coherence: stop-loss below every entry (ideal and secondary) and
  below the target; target above the entries; stop not glued to the entry.
- R2 minimum risk/reward measured from the ideal entry.
- R3 ideal entry within a maximum distance of the current price.
- R4 position size coherent with a non-buy decision (note only).

Model numbers are never rewritten.  When R1-R3 fail on a ``buy`` decision the
decision is downgraded to watch, confidence is capped at medium and a visible
note explains which rule failed.  Missing levels, hold-plan issues and R4 only
add notes.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.report_language import (
    localize_confidence_level,
    localize_operation_advice,
    normalize_report_language,
)
from src.services.decision_profile_policy import DecisionSignalCandidate, _price_violations
from src.utils.sniper_points import extract_sniper_points

logger = logging.getLogger(__name__)

MIN_RISK_REWARD = 1.5
MAX_ENTRY_DISTANCE_PCT = 5.0
MIN_STOP_DISTANCE_PCT = 0.5

# Violations that downgrade a buy decision (R1-R3).  A plan with missing
# levels only gets a note: the integrity check already handles an absent
# stop-loss, and there is nothing numeric to contradict the decision.
BLOCKING_CODES = frozenset(
    {
        "entry_range_invalid",
        "stop_loss_not_below_target_price",
        "stop_loss_not_below_entry_high",
        "stop_loss_not_below_entry_low",
        "target_price_not_above_entry_low",
        "target_price_not_above_entry_high",
        "stop_loss_too_close_to_entry",
        "risk_reward_below_minimum",
        "entry_far_from_current_price",
    }
)

_POSITION_FRACTION_RE = re.compile(r"(\d+(?:\.\d+)?)\s*/\s*10\b")
_POSITION_PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[%％]")
_POSITION_CHENG_RE = re.compile(r"(\d+(?:\.\d+)?)\s*成")


@dataclass
class ActionPlanCheck:
    decision_type: str
    current_price: Optional[float] = None
    ideal_buy: Optional[float] = None
    secondary_buy: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_reward: Optional[float] = None
    entry_distance_pct: Optional[float] = None
    suggested_position: Optional[str] = None
    risk_control_prices: List[float] = field(default_factory=list)
    violations: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)

    @property
    def blocking(self) -> List[str]:
        return [code for code in self.violations if code in BLOCKING_CODES]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "decision_type": self.decision_type,
            "current_price": self.current_price,
            "ideal_buy": self.ideal_buy,
            "secondary_buy": self.secondary_buy,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "risk_reward": None if self.risk_reward is None else round(self.risk_reward, 2),
            "entry_distance_pct": (
                None if self.entry_distance_pct is None else round(self.entry_distance_pct, 2)
            ),
            "violations": list(self.violations),
            "missing": list(self.missing),
            "min_risk_reward": MIN_RISK_REWARD,
            "max_entry_distance_pct": MAX_ENTRY_DISTANCE_PCT,
        }


def evaluate_action_plan(result: Any, *, current_price: Optional[float] = None) -> ActionPlanCheck:
    """Evaluate the battle plan without mutating ``result``."""

    decision_type = str(getattr(result, "decision_type", "") or "").strip().lower()
    points = extract_sniper_points(result)
    price = _positive(current_price)
    if price is None:
        price = _positive(getattr(result, "current_price", None))
    if price is None:
        price = _positive(_price_position(result).get("current_price"))

    check = ActionPlanCheck(
        decision_type=decision_type,
        current_price=price,
        ideal_buy=points.get("ideal_buy"),
        secondary_buy=points.get("secondary_buy"),
        stop_loss=points.get("stop_loss"),
        take_profit=points.get("take_profit"),
        suggested_position=_suggested_position_text(result),
    )

    if decision_type == "buy":
        _check_buy_plan(check)
    elif decision_type == "hold":
        # A hold/watch card often still carries a conditional entry plan; the
        # same R1-R3 checks apply, but only as notes (see BLOCKING_CODES use).
        if check.ideal_buy is not None or check.secondary_buy is not None:
            _check_buy_plan(check)
        _check_hold_plan(check)

    elif decision_type == "sell":
        _check_sell_plan(check, _raw_sniper_text(result, "ideal_buy"))

    if decision_type in ("buy", "hold", "sell"):
        _check_risk_control_stop(check, _risk_control_text(result))

    if decision_type != "buy" and _position_allocation(check.suggested_position):
        check.violations.append("position_size_with_non_buy_decision")

    return check


def apply_action_plan_guardrail(
    result: Any,
    *,
    report_language: Optional[str] = None,
    current_price: Optional[float] = None,
) -> List[str]:
    """Validate the action plan in place. Returns the adjustment codes applied."""

    if result is None:
        return []
    try:
        language = normalize_report_language(
            report_language or getattr(result, "report_language", "zh")
        )
        check = evaluate_action_plan(result, current_price=current_price)
        if not check.violations:
            return []

        dashboard = getattr(result, "dashboard", None)
        if not isinstance(dashboard, dict):
            dashboard = {}
            result.dashboard = dashboard

        notes = [_note_for(code, check, language) for code in check.violations]
        notes = [note for note in notes if note]
        adjustments: List[str] = []

        downgraded = check.decision_type == "buy" and bool(check.blocking)
        if downgraded:
            _downgrade_buy(result, dashboard, check, notes, language)
            adjustments.append("action_plan_buy_downgraded")
            # R4 on the final decision: the position text was written for a buy.
            if _position_allocation(check.suggested_position):
                check.violations.append("position_size_with_non_buy_decision")
                notes.append(_note_for("position_size_with_non_buy_decision", check, language))
        if notes:
            _append_notes(dashboard, notes)
            adjustments.append("action_plan_notes_added")

        validation = check.as_dict()
        validation["downgraded"] = downgraded
        validation["notes"] = notes
        dashboard["action_plan_validation"] = validation
        logger.info(
            "[action_plan_guardrail] %s violations=%s downgraded=%s",
            getattr(result, "code", ""),
            check.violations,
            downgraded,
        )
        return adjustments
    except Exception as exc:  # pragma: no cover - defensive: never break the report
        logger.warning("[action_plan_guardrail] Validation skipped: %s", exc)
        return []


# --------------------------------------------------------------------------- rules


def _check_buy_plan(check: ActionPlanCheck) -> None:
    entry = check.ideal_buy
    stop = check.stop_loss
    target = check.take_profit

    for key, value in (("ideal_buy", entry), ("stop_loss", stop), ("take_profit", target)):
        if value is None:
            check.missing.append(key)
    if check.missing:
        check.violations.append("action_plan_incomplete")

    # Note only: a secondary entry above the ideal one mixes a pullback entry
    # with a breakout entry in a single plan (seen on JPM/XOM 2026-11-24).
    if entry is not None and check.secondary_buy is not None and check.secondary_buy > entry:
        check.violations.append("secondary_entry_above_ideal")

    entry_low, entry_high = _entry_range(entry, check.secondary_buy)
    candidate = DecisionSignalCandidate(
        action="buy",
        entry_low=entry_low,
        entry_high=entry_high,
        stop_loss=stop,
        target_price=target,
    )
    # Reuse the reassessment price checks; entry_range_invalid cannot occur
    # because _entry_range orders the bounds.
    check.violations.extend(_price_violations(candidate))
    # _price_violations only compares the stop with the upper entry; a stop at or
    # above the lower entry (usually secondary_buy) is just as contradictory.
    if (
        stop is not None
        and entry_low is not None
        and entry_low != entry_high
        and stop >= entry_low
        and "stop_loss_not_below_entry_high" not in check.violations
    ):
        check.violations.append("stop_loss_not_below_entry_low")

    # Likewise the target must clear the upper entry, otherwise the second
    # tranche (or the ideal entry) has no upside.
    if (
        target is not None
        and entry_high is not None
        and entry_low != entry_high
        and target <= entry_high
        and "target_price_not_above_entry_low" not in check.violations
    ):
        check.violations.append("target_price_not_above_entry_high")

    if stop is not None and entry_low is not None and stop < entry_low:
        if (entry_low - stop) / entry_low * 100 < MIN_STOP_DISTANCE_PCT:
            check.violations.append("stop_loss_too_close_to_entry")

    if entry is not None and stop is not None and stop < entry:
        if target is not None and target > entry:
            check.risk_reward = (target - entry) / (entry - stop)
            if check.risk_reward < MIN_RISK_REWARD:
                check.violations.append("risk_reward_below_minimum")

    if entry is not None and check.current_price is not None:
        check.entry_distance_pct = (entry - check.current_price) / check.current_price * 100
        if abs(check.entry_distance_pct) > MAX_ENTRY_DISTANCE_PCT:
            check.violations.append("entry_far_from_current_price")


def _check_risk_control_stop(check: ActionPlanCheck, risk_control: Optional[str]) -> None:
    """R5 (note only): prices quoted in risk_control should match the stop-loss."""

    stop = check.stop_loss
    if stop is None or not risk_control:
        return
    prices = _prices_in_text(risk_control)
    if not prices:
        return
    tolerance = stop * MIN_STOP_DISTANCE_PCT / 100
    if all(abs(price - stop) > tolerance for price in prices):
        check.risk_control_prices = prices
        check.violations.append("risk_control_stop_mismatch")


def _check_hold_plan(check: ActionPlanCheck) -> None:
    stop = check.stop_loss
    target = check.take_profit
    if (
        stop is not None
        and target is not None
        and stop >= target
        and "stop_loss_not_below_target_price" not in check.violations
    ):
        check.violations.append("hold_stop_loss_not_below_target_price")
    if stop is not None and check.current_price is not None and stop >= check.current_price:
        check.violations.append("hold_stop_loss_at_or_above_current_price")


# Buy-level text that only says there is no entry ("No buy point; wait for a
# reclaim of 150") must not count as a buy entry on a sell/reduce card.
_NO_BUY_MARKERS = (
    "no buy", "not a buy", "no entry", "no new", "not applicable", "n/a", "do not buy",
    "avoid buying", "wait for", "wait until", "none",
    "暂无", "无买点", "不建议", "不买", "不适用", "等待", "观望",
    "매수 없음", "매수하지", "해당 없음", "대기",
)


def _check_sell_plan(check: ActionPlanCheck, ideal_text: Optional[str]) -> None:
    """Sell/reduce cards (notes only): no buy entry, coherent stop and target."""

    text = (ideal_text or "").strip().lower()
    if check.ideal_buy is not None and not any(marker in text for marker in _NO_BUY_MARKERS):
        check.violations.append("sell_plan_has_buy_entry")
    stop = check.stop_loss
    target = check.take_profit
    if stop is not None and target is not None and stop >= target:
        check.violations.append("hold_stop_loss_not_below_target_price")
    if stop is not None and check.current_price is not None and stop >= check.current_price:
        check.violations.append("sell_stop_at_or_above_current_price")


def _raw_sniper_text(result: Any, key: str) -> Optional[str]:
    getter = getattr(result, "get_sniper_points", None)
    points = getter() if callable(getter) else None
    if not isinstance(points, dict):
        dashboard = getattr(result, "dashboard", None)
        battle_plan = dashboard.get("battle_plan") if isinstance(dashboard, dict) else None
        points = battle_plan.get("sniper_points") if isinstance(battle_plan, dict) else None
    value = points.get(key) if isinstance(points, dict) else None
    return None if value is None else str(value)


# --------------------------------------------------------------------------- effects


def _downgrade_buy(
    result: Any,
    dashboard: Dict[str, Any],
    check: ActionPlanCheck,
    notes: List[str],
    language: str,
) -> None:
    from src.analyzer import _bound_hold_watch_sentiment_score, _sync_stability_dashboard_fields

    advice = localize_operation_advice("观望", language)
    reason = _downgrade_reason(check, notes, language)

    result.decision_type = "hold"
    result.operation_advice = advice
    if str(getattr(result, "confidence_level", "") or "").strip().lower() in {"高", "high", "높음"}:
        result.confidence_level = localize_confidence_level("medium", language)
    # Bounds the score to the hold/watch band and records guardrail_reason so the
    # canonical score/action alignment does not re-upgrade the decision.
    _bound_hold_watch_sentiment_score(result, reason=reason, final_action="watch")

    core = dashboard.get("core_conclusion")
    if not isinstance(core, dict):
        core = {}
        dashboard["core_conclusion"] = core
    core["signal_type"] = {"zh": "🟡持有观望", "ko": "🟡 보유 / 관망"}.get(language, "🟡 Hold / Watch")
    separator = "：" if language == "zh" else ": "
    core["one_sentence"] = f"{advice}{separator}{reason}"

    risk_warning = str(getattr(result, "risk_warning", "") or "")
    if reason not in risk_warning:
        sep = "；" if language == "zh" else "; "
        result.risk_warning = f"{risk_warning}{sep}{reason}" if risk_warning else reason

    _sync_stability_dashboard_fields(result)


def _append_notes(dashboard: Dict[str, Any], notes: List[str]) -> None:
    phase_decision = dashboard.get("phase_decision")
    if not isinstance(phase_decision, dict):
        phase_decision = {}
        dashboard["phase_decision"] = phase_decision
    limitations = phase_decision.get("data_limitations")
    if not isinstance(limitations, list):
        limitations = []
    for note in notes:
        if note not in limitations:
            limitations.append(note)
    phase_decision["data_limitations"] = limitations


# --------------------------------------------------------------------------- texts


def _downgrade_reason(check: ActionPlanCheck, notes: List[str], language: str) -> str:
    blocking_notes = [
        _note_for(code, check, language) for code in check.violations if code in BLOCKING_CODES
    ]
    detail = " ".join(note for note in blocking_notes if note) or " ".join(notes)
    if language == "zh":
        return f"作战计划未通过校验，买入已降级为观望。{detail}"
    if language == "ko":
        return f"실행 계획이 검증을 통과하지 못해 매수를 관망으로 낮췄습니다. {detail}"
    return f"Action plan failed validation; buy downgraded to Watch. {detail}"


def _note_for(code: str, check: ActionPlanCheck, language: str) -> str:
    f = _fmt
    values = {
        "entry": f(check.ideal_buy),
        "secondary": f(check.secondary_buy),
        "stop": f(check.stop_loss),
        "target": f(check.take_profit),
        "price": f(check.current_price),
        "rr": f(check.risk_reward),
        "min_rr": f(MIN_RISK_REWARD),
        "dist": f(check.entry_distance_pct, signed=True, digits=1),
        "max_dist": f(MAX_ENTRY_DISTANCE_PCT, digits=0),
        "min_stop": f(MIN_STOP_DISTANCE_PCT, digits=1),
        "missing": ", ".join(check.missing),
        "entry_high": f(_entry_range(check.ideal_buy, check.secondary_buy)[1]),
        "entry_low": f(_entry_range(check.ideal_buy, check.secondary_buy)[0]),
        "position": check.suggested_position or "",
        "rc_prices": ", ".join(f(price) for price in check.risk_control_prices),
    }
    templates = _NOTE_TEMPLATES.get(language, _NOTE_TEMPLATES["en"])
    template = templates.get(code)
    return template.format(**values) if template else ""


_NOTE_TEMPLATES: Dict[str, Dict[str, str]] = {
    "en": {
        "action_plan_incomplete": "Action plan check: missing numeric level(s): {missing}.",
        "stop_loss_not_below_target_price": "Action plan check: stop-loss {stop} is not below the target {target}.",
        "stop_loss_not_below_entry_high": "Action plan check: stop-loss {stop} is not below the entry {entry_high}.",
        "stop_loss_not_below_entry_low": "Action plan check: stop-loss {stop} is not below the lower entry {entry_low}.",
        "target_price_not_above_entry_low": "Action plan check: target {target} is not above the entry {entry_low}.",
        "target_price_not_above_entry_high": "Action plan check: target {target} is not above the upper entry {entry_high}.",
        "stop_loss_too_close_to_entry": "Action plan check: stop-loss {stop} is within {min_stop}% of the entry {entry_low}.",
        "risk_reward_below_minimum": "Action plan check: risk/reward {rr} is below the {min_rr} minimum (entry {entry}, stop {stop}, target {target}).",
        "entry_far_from_current_price": "Action plan check: ideal entry {entry} is {dist}% from the current price {price} (limit ±{max_dist}%).",
        "hold_stop_loss_not_below_target_price": "Action plan check: stop-loss {stop} is not below the target {target}; review the risk line.",
        "hold_stop_loss_at_or_above_current_price": "Action plan check: stop-loss {stop} is at or above the current price {price}; review the risk line.",
        "position_size_with_non_buy_decision": "Action plan check: suggested position \"{position}\" applies only to existing holders; the decision is not a buy, so it does not justify opening a new position.",
        "risk_control_stop_mismatch": "Action plan check: risk control quotes {rc_prices} but the stop-loss is {stop}; the plan has two different stop levels.",
        "secondary_entry_above_ideal": "Action plan check: secondary entry {secondary} is above the ideal entry {entry}; the plan mixes a pullback entry with a breakout entry.",
        "sell_plan_has_buy_entry": "Action plan check: the decision is sell/reduce but the plan lists a buy entry {entry}; do not open or add to a position.",
        "sell_stop_at_or_above_current_price": "Action plan check: stop-loss {stop} is at or above the current price {price}; on a sell/reduce decision the remaining position would already be exited.",
    },
    "zh": {
        "action_plan_incomplete": "作战计划校验：缺少数值点位：{missing}。",
        "stop_loss_not_below_target_price": "作战计划校验：止损 {stop} 未低于目标位 {target}。",
        "stop_loss_not_below_entry_high": "作战计划校验：止损 {stop} 未低于买入位 {entry_high}。",
        "stop_loss_not_below_entry_low": "作战计划校验：止损 {stop} 未低于较低买入位 {entry_low}。",
        "target_price_not_above_entry_low": "作战计划校验：目标位 {target} 未高于买入位 {entry_low}。",
        "target_price_not_above_entry_high": "作战计划校验：目标位 {target} 未高于较高买入位 {entry_high}。",
        "stop_loss_too_close_to_entry": "作战计划校验：止损 {stop} 距买入位 {entry_low} 不足 {min_stop}%。",
        "risk_reward_below_minimum": "作战计划校验：盈亏比 {rr} 低于最低要求 {min_rr}（买入 {entry}，止损 {stop}，目标 {target}）。",
        "entry_far_from_current_price": "作战计划校验：理想买入位 {entry} 偏离现价 {price} {dist}%（上限 ±{max_dist}%）。",
        "hold_stop_loss_not_below_target_price": "作战计划校验：止损 {stop} 未低于目标位 {target}，请复核风控线。",
        "hold_stop_loss_at_or_above_current_price": "作战计划校验：止损 {stop} 不低于现价 {price}，请复核风控线。",
        "position_size_with_non_buy_decision": "作战计划校验：建议仓位“{position}”仅适用于已持仓者；当前结论并非买入，不构成新开仓依据。",
        "risk_control_stop_mismatch": "作战计划校验：风控说明提到 {rc_prices}，但止损位为 {stop}，计划中存在两个不同的止损位。",
        "secondary_entry_above_ideal": "作战计划校验：次级买入位 {secondary} 高于理想买入位 {entry}，计划混合了回踩买入与突破买入。",
        "sell_plan_has_buy_entry": "作战计划校验：结论为卖出/减仓，但计划仍给出买入位 {entry}；不应据此开仓或加仓。",
        "sell_stop_at_or_above_current_price": "作战计划校验：止损 {stop} 不低于现价 {price}；卖出/减仓结论下剩余仓位应已离场。",
    },
    "ko": {
        "action_plan_incomplete": "실행 계획 점검: 수치 가격대 누락: {missing}.",
        "stop_loss_not_below_target_price": "실행 계획 점검: 손절가 {stop}이(가) 목표가 {target}보다 낮지 않습니다.",
        "stop_loss_not_below_entry_high": "실행 계획 점검: 손절가 {stop}이(가) 매수가 {entry_high}보다 낮지 않습니다.",
        "stop_loss_not_below_entry_low": "실행 계획 점검: 손절가 {stop}이(가) 하단 매수가 {entry_low}보다 낮지 않습니다.",
        "target_price_not_above_entry_low": "실행 계획 점검: 목표가 {target}이(가) 매수가 {entry_low}보다 높지 않습니다.",
        "target_price_not_above_entry_high": "실행 계획 점검: 목표가 {target}이(가) 상단 매수가 {entry_high}보다 높지 않습니다.",
        "stop_loss_too_close_to_entry": "실행 계획 점검: 손절가 {stop}이(가) 매수가 {entry_low}와 {min_stop}% 이내입니다.",
        "risk_reward_below_minimum": "실행 계획 점검: 손익비 {rr}이(가) 최소 {min_rr} 미만입니다 (매수 {entry}, 손절 {stop}, 목표 {target}).",
        "entry_far_from_current_price": "실행 계획 점검: 이상 매수가 {entry}이(가) 현재가 {price} 대비 {dist}% 떨어져 있습니다 (한도 ±{max_dist}%).",
        "hold_stop_loss_not_below_target_price": "실행 계획 점검: 손절가 {stop}이(가) 목표가 {target}보다 낮지 않습니다. 리스크 라인을 재검토하세요.",
        "hold_stop_loss_at_or_above_current_price": "실행 계획 점검: 손절가 {stop}이(가) 현재가 {price} 이상입니다. 리스크 라인을 재검토하세요.",
        "position_size_with_non_buy_decision": "실행 계획 점검: 제안 비중 \"{position}\"은(는) 기존 보유자에게만 해당하며, 매수 결론이 아니므로 신규 진입 근거가 아닙니다.",
        "risk_control_stop_mismatch": "실행 계획 점검: 리스크 관리 문구는 {rc_prices}을(를) 언급하지만 손절가는 {stop}입니다. 손절 기준이 두 개입니다.",
        "secondary_entry_above_ideal": "실행 계획 점검: 보조 매수가 {secondary}이(가) 이상 매수가 {entry}보다 높습니다. 눌림목 매수와 돌파 매수가 섞여 있습니다.",
        "sell_plan_has_buy_entry": "실행 계획 점검: 결론은 매도/비중 축소인데 계획에 매수가 {entry}이(가) 있습니다. 신규 진입이나 추가 매수 근거가 아닙니다.",
        "sell_stop_at_or_above_current_price": "실행 계획 점검: 손절가 {stop}이(가) 현재가 {price} 이상입니다. 매도/비중 축소 결론에서는 잔여 포지션이 이미 청산되어야 합니다.",
    },
}


# --------------------------------------------------------------------------- helpers


def _entry_range(
    ideal_buy: Optional[float], secondary_buy: Optional[float]
) -> tuple[Optional[float], Optional[float]]:
    values = [value for value in (ideal_buy, secondary_buy) if value is not None]
    if not values:
        return None, None
    return min(values), max(values)


def _positive(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _price_position(result: Any) -> Dict[str, Any]:
    dashboard = getattr(result, "dashboard", None)
    if not isinstance(dashboard, dict):
        return {}
    perspective = dashboard.get("data_perspective")
    if not isinstance(perspective, dict):
        return {}
    position = perspective.get("price_position")
    return position if isinstance(position, dict) else {}


_PRICE_IN_TEXT_RE = re.compile(r"(?<![A-Za-z0-9.])\$?(\d+(?:,\d{3})*(?:\.\d+)?)(?![\d%％/])")


def _prices_in_text(text: str) -> List[float]:
    """Extract price-like numbers, skipping MA periods (MA20), percentages and X/10."""

    prices: List[float] = []
    for match in _PRICE_IN_TEXT_RE.finditer(text):
        start = match.start()
        if text[max(0, start - 2):start].upper() == "MA":
            continue
        try:
            value = float(match.group(1).replace(",", ""))
        except ValueError:
            continue
        if value > 0:
            prices.append(value)
    return prices


def _risk_control_text(result: Any) -> Optional[str]:
    dashboard = getattr(result, "dashboard", None)
    if not isinstance(dashboard, dict):
        return None
    battle_plan = dashboard.get("battle_plan")
    if not isinstance(battle_plan, dict):
        return None
    strategy = battle_plan.get("position_strategy")
    if not isinstance(strategy, dict):
        return None
    text = str(strategy.get("risk_control") or "").strip()
    return text or None


def _suggested_position_text(result: Any) -> Optional[str]:
    dashboard = getattr(result, "dashboard", None)
    if not isinstance(dashboard, dict):
        return None
    battle_plan = dashboard.get("battle_plan")
    if not isinstance(battle_plan, dict):
        return None
    strategy = battle_plan.get("position_strategy")
    if not isinstance(strategy, dict):
        return None
    text = str(strategy.get("suggested_position") or "").strip()
    return text or None


def _position_allocation(text: Optional[str]) -> Optional[float]:
    """Return a positive allocation parsed from X/10, X% or X成; otherwise None."""

    if not text:
        return None
    for pattern, scale in (
        (_POSITION_FRACTION_RE, 10.0),
        (_POSITION_PERCENT_RE, 1.0),
        (_POSITION_CHENG_RE, 10.0),
    ):
        match = pattern.search(text)
        if match:
            try:
                value = float(match.group(1)) * scale
            except ValueError:
                continue
            return value if value > 0 else None
    return None


def _fmt(value: Optional[float], *, signed: bool = False, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    text = f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"
    return text
