# -*- coding: utf-8 -*-
"""Tests for the deterministic action-plan (battle_plan) guardrail."""

from __future__ import annotations

from typing import Any, Optional

from src.action_plan_guardrail import apply_action_plan_guardrail, evaluate_action_plan
from src.analyzer import AnalysisResult
from src.services.decision_signal_extractor import resolve_decision_signal_action_fields


def _result(
    *,
    decision_type: str = "buy",
    current_price: Optional[float] = 100.0,
    ideal_buy: Any = "Ideal buy point: 99.00 (near MA5)",
    secondary_buy: Any = "Secondary buy point: 97.00 (near MA10)",
    stop_loss: Any = "Stop-loss: 95.00 (break below MA20)",
    take_profit: Any = "Target: 110.00 (previous high)",
    suggested_position: str = "3/10",
    sentiment_score: int = 75,
    confidence_level: str = "High",
    operation_advice: str = "Buy",
) -> AnalysisResult:
    return AnalysisResult(
        code="TEST",
        name="Test Corp",
        sentiment_score=sentiment_score,
        trend_prediction="Bullish",
        operation_advice=operation_advice,
        decision_type=decision_type,
        confidence_level=confidence_level,
        analysis_summary="Trend is constructive.",
        report_language="en",
        current_price=current_price,
        dashboard={
            "decision_type": decision_type,
            "operation_advice": operation_advice,
            "core_conclusion": {"one_sentence": "Buy on pullback."},
            "battle_plan": {
                "sniper_points": {
                    "ideal_buy": ideal_buy,
                    "secondary_buy": secondary_buy,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                },
                "position_strategy": {
                    "suggested_position": suggested_position,
                    "entry_plan": "Scale in at the two entries.",
                    "risk_control": "Exit below the stop-loss.",
                },
            },
            "phase_decision": {"data_limitations": []},
        },
    )


def _notes(result: AnalysisResult) -> list[str]:
    return result.dashboard["phase_decision"]["data_limitations"]


# --------------------------------------------------------------------------- pass


def test_coherent_buy_plan_is_untouched() -> None:
    # entry 99, stop 95, target 110 -> R:R = 11 / 4 = 2.75; entry 1% below price.
    result = _result()

    assert apply_action_plan_guardrail(result, report_language="en") == []
    assert result.decision_type == "buy"
    assert result.operation_advice == "Buy"
    assert result.confidence_level == "High"
    assert "action_plan_validation" not in result.dashboard
    assert _notes(result) == []


def test_risk_reward_exactly_at_minimum_passes() -> None:
    # entry 100, stop 96, target 106 -> R:R = 6 / 4 = 1.5
    result = _result(
        ideal_buy="100", secondary_buy=None, stop_loss="96", take_profit="106"
    )
    check = evaluate_action_plan(result)
    assert check.violations == []
    assert round(check.risk_reward, 2) == 1.5


# --------------------------------------------------------------------------- R1


def test_stop_equal_to_secondary_buy_downgrades() -> None:
    result = _result(secondary_buy="Secondary buy point: 95.00", stop_loss="Stop-loss: 95.00")

    adjustments = apply_action_plan_guardrail(result, report_language="en")

    assert "action_plan_buy_downgraded" in adjustments
    validation = result.dashboard["action_plan_validation"]
    assert "stop_loss_not_below_entry_low" in validation["violations"]
    assert result.decision_type == "hold"
    assert any("not below the lower entry 95.00" in note for note in _notes(result))


def test_stop_above_ideal_entry_downgrades() -> None:
    result = _result(ideal_buy="99", secondary_buy=None, stop_loss="101")
    apply_action_plan_guardrail(result, report_language="en")
    violations = result.dashboard["action_plan_validation"]["violations"]
    assert "stop_loss_not_below_entry_high" in violations
    assert result.decision_type == "hold"


def test_stop_glued_to_entry_downgrades() -> None:
    # 0.3% below the only entry
    result = _result(ideal_buy="100", secondary_buy=None, stop_loss="99.70", take_profit="120")
    apply_action_plan_guardrail(result, report_language="en")
    assert "stop_loss_too_close_to_entry" in result.dashboard["action_plan_validation"]["violations"]


def test_target_below_entry_downgrades() -> None:
    result = _result(take_profit="98")
    apply_action_plan_guardrail(result, report_language="en")
    violations = result.dashboard["action_plan_validation"]["violations"]
    assert "target_price_not_above_entry_high" in violations
    assert result.decision_type == "hold"


def test_missing_numeric_stop_is_note_only() -> None:
    result = _result(stop_loss="N/A")
    adjustments = apply_action_plan_guardrail(result, report_language="en")
    validation = result.dashboard["action_plan_validation"]
    assert adjustments == ["action_plan_notes_added"]
    assert validation["violations"] == ["action_plan_incomplete"]
    assert validation["missing"] == ["stop_loss"]
    assert result.decision_type == "buy"
    assert any("missing numeric level(s): stop_loss" in note for note in _notes(result))


# --------------------------------------------------------------------------- R2


def test_low_risk_reward_downgrades_and_explains() -> None:
    # entry 99, stop 95, target 101 -> R:R = 2 / 4 = 0.50
    result = _result(take_profit="Target: 101.00", suggested_position="Light position")

    apply_action_plan_guardrail(result, report_language="en")

    validation = result.dashboard["action_plan_validation"]
    assert validation["violations"] == ["risk_reward_below_minimum"]
    assert validation["risk_reward"] == 0.5
    assert validation["downgraded"] is True
    assert result.decision_type == "hold"
    assert result.operation_advice == "Watch"
    assert result.confidence_level == "Medium"
    assert result.sentiment_score <= 59
    assert result.dashboard["decision_type"] == "hold"
    assert result.dashboard["core_conclusion"]["one_sentence"].startswith("Watch: Action plan failed validation")
    assert "risk/reward 0.50 is below the 1.50 minimum" in result.risk_warning
    assert any("risk/reward 0.50" in note for note in _notes(result))


def test_downgrade_is_not_reupgraded_by_score_alignment() -> None:
    result = _result(take_profit="101", sentiment_score=78)
    apply_action_plan_guardrail(result, report_language="en")

    fields = resolve_decision_signal_action_fields(result, report_type="simple")

    assert fields["action"] in {"watch", "hold"}


def test_model_prices_are_never_rewritten() -> None:
    result = _result(take_profit="Target: 101.00")
    before = dict(result.dashboard["battle_plan"]["sniper_points"])
    apply_action_plan_guardrail(result, report_language="en")
    assert result.dashboard["battle_plan"]["sniper_points"] == before


# --------------------------------------------------------------------------- R3


def test_entry_far_from_current_price_downgrades() -> None:
    # entry 99 vs price 106 -> -6.6%
    result = _result(current_price=106.0, take_profit="120", suggested_position="Light position")
    apply_action_plan_guardrail(result, report_language="en")
    validation = result.dashboard["action_plan_validation"]
    assert validation["violations"] == ["entry_far_from_current_price"]
    assert validation["entry_distance_pct"] == -6.6
    assert any("-6.6% from the current price 106.00" in note for note in _notes(result))


def test_entry_distance_falls_back_to_price_position() -> None:
    result = _result(current_price=None, take_profit="120")
    result.dashboard["data_perspective"] = {"price_position": {"current_price": "110.0"}}
    check = evaluate_action_plan(result)
    assert "entry_far_from_current_price" in check.violations


def test_missing_current_price_skips_distance_rule() -> None:
    result = _result(current_price=None)
    assert evaluate_action_plan(result).violations == []


# --------------------------------------------------------------------------- R4


def test_position_size_on_watch_is_note_only() -> None:
    result = _result(
        decision_type="hold",
        operation_advice="Watch",
        suggested_position="Suggested position: 3/10",
        confidence_level="Medium",
        sentiment_score=55,
    )

    adjustments = apply_action_plan_guardrail(result, report_language="en")

    assert adjustments == ["action_plan_notes_added"]
    assert result.decision_type == "hold"
    assert result.operation_advice == "Watch"
    assert result.sentiment_score == 55
    assert result.dashboard["action_plan_validation"]["downgraded"] is False
    assert any("applies only to existing holders" in note for note in _notes(result))


def test_zero_position_on_watch_has_no_note() -> None:
    result = _result(decision_type="hold", suggested_position="0/10 (stay out)")
    assert apply_action_plan_guardrail(result, report_language="en") == []


def test_downgraded_buy_with_position_size_gets_both_notes() -> None:
    result = _result(take_profit="101", suggested_position="3/10")
    apply_action_plan_guardrail(result, report_language="en")
    violations = result.dashboard["action_plan_validation"]["violations"]
    assert violations == ["risk_reward_below_minimum", "position_size_with_non_buy_decision"]
    notes = _notes(result)
    assert any("risk/reward 0.50" in note for note in notes)
    assert any("applies only to existing holders" in note for note in notes)
    # R4 never rewrites the model's position text
    assert result.dashboard["battle_plan"]["position_strategy"]["suggested_position"] == "3/10"


# --------------------------------------------------------------------------- hold / sell


def test_hold_with_stop_above_price_is_note_only() -> None:
    result = _result(
        decision_type="hold",
        operation_advice="Hold",
        stop_loss="Stop-loss: 102.00",
        suggested_position="Keep current position",
    )
    adjustments = apply_action_plan_guardrail(result, report_language="en")
    assert adjustments == ["action_plan_notes_added"]
    assert result.decision_type == "hold"
    assert "hold_stop_loss_at_or_above_current_price" in result.dashboard["action_plan_validation"]["violations"]


def test_sell_decision_is_not_validated_for_entries() -> None:
    result = _result(decision_type="sell", ideal_buy="N/A", stop_loss="N/A", suggested_position="0/10")
    assert apply_action_plan_guardrail(result, report_language="en") == []


def test_chinese_notes_are_localized() -> None:
    result = _result(take_profit="目标位：101元")
    result.report_language = "zh"
    apply_action_plan_guardrail(result, report_language="zh")
    assert result.operation_advice == "观望"
    assert any("盈亏比 0.50 低于最低要求 1.50" in note for note in _notes(result))


# --------------------------------------------------------------------------- real cases
# Rendered text of the 2026-11-18 email (report_language=en). The rendered card
# may differ slightly from the model's raw sniper_points JSON.


def _real_case(
    *,
    price: float,
    ideal: str,
    secondary: str,
    stop: str,
    target: str,
    position: str,
    risk_control: str,
) -> AnalysisResult:
    result = _result(
        decision_type="hold",
        operation_advice="Watch",
        confidence_level="Medium",
        sentiment_score=50,
        current_price=price,
        ideal_buy=ideal,
        secondary_buy=secondary,
        stop_loss=stop,
        take_profit=target,
        suggested_position=position,
    )
    result.dashboard["battle_plan"]["position_strategy"]["risk_control"] = risk_control
    return result


def test_real_case_aapl_2026_11_18() -> None:
    result = _real_case(
        price=351.91,
        ideal="Ideal buy point: $354.24 (near MA5)",
        secondary="Secondary buy point: $350.51 (near MA10)",
        stop="Stop-loss: $347.05 (break below MA20)",
        target="$355.36 (previous high / resistance)",
        position="Suggested position: 3/10",
        risk_control="Strict stop-loss if price closes below the MA20 line.",
    )

    adjustments = apply_action_plan_guardrail(result, report_language="en")

    validation = result.dashboard["action_plan_validation"]
    assert adjustments == ["action_plan_notes_added"]
    assert result.decision_type == "hold" and result.operation_advice == "Watch"
    assert (validation["ideal_buy"], validation["secondary_buy"]) == (354.24, 350.51)
    assert (validation["stop_loss"], validation["take_profit"]) == (347.05, 355.36)
    # (355.36 - 354.24) / (354.24 - 347.05) = 1.12 / 7.19 = 0.156
    assert validation["risk_reward"] == 0.16
    assert validation["entry_distance_pct"] == 0.66
    assert validation["violations"] == [
        "risk_reward_below_minimum",
        "position_size_with_non_buy_decision",
    ]


def test_real_case_msft_2026_11_18() -> None:
    result = _real_case(
        price=506.24,
        ideal="Ideal buy point: 518.45 (near MA5 upon recovery)",
        secondary="Secondary buy point: 496.51 (near strong support low)",
        stop="Stop-loss: 486.00 (break below support)",
        target="527.33 (near MA20 resistance)",
        position="3/10",
        risk_control="Strictly enforce stop-loss if price breaks below the 496.51 low.",
    )

    apply_action_plan_guardrail(result, report_language="en")

    validation = result.dashboard["action_plan_validation"]
    # (527.33 - 518.45) / (518.45 - 486.00) = 8.88 / 32.45 = 0.274
    assert validation["risk_reward"] == 0.27
    assert validation["entry_distance_pct"] == 2.41
    assert validation["violations"] == [
        "risk_reward_below_minimum",
        "risk_control_stop_mismatch",
        "position_size_with_non_buy_decision",
    ]
    assert result.decision_type == "hold"
    assert any("risk control quotes 496.51 but the stop-loss is 486.00" in n for n in _notes(result))


def test_real_case_nvda_2026_11_18() -> None:
    result = _real_case(
        price=155.46,
        ideal="Ideal buy point: 150.33 (near current support)",
        secondary="Secondary buy point: 145.00 (deeper historical support)",
        stop="Stop-loss: 145.00 (break below key support)",
        target="162.87 (near MA5 resistance)",
        position="Suggested position: 0/10",
        risk_control=(
            "Strictly avoid aggressive bottom-fishing due to active securities "
            "litigation and bearish moving average alignment."
        ),
    )

    apply_action_plan_guardrail(result, report_language="en")

    validation = result.dashboard["action_plan_validation"]
    # (162.87 - 150.33) / (150.33 - 145.00) = 12.54 / 5.33 = 2.35 -> passes R2
    assert validation["risk_reward"] == 2.35
    assert validation["entry_distance_pct"] == -3.3
    assert validation["violations"] == ["stop_loss_not_below_entry_low"]
    assert result.decision_type == "hold"
    assert any("stop-loss 145.00 is not below the lower entry 145.00" in n for n in _notes(result))


def test_real_case_aapl_as_buy_would_be_downgraded() -> None:
    """Same AAPL plan under a buy decision: R2 fails, so buy -> watch."""
    result = _real_case(
        price=351.91,
        ideal="Ideal buy point: $354.24 (near MA5)",
        secondary="Secondary buy point: $350.51 (near MA10)",
        stop="Stop-loss: $347.05 (break below MA20)",
        target="$355.36 (previous high / resistance)",
        position="Suggested position: 3/10",
        risk_control="Strict stop-loss if price closes below the MA20 line.",
    )
    result.decision_type = "buy"
    result.operation_advice = "Buy"

    adjustments = apply_action_plan_guardrail(result, report_language="en")

    assert "action_plan_buy_downgraded" in adjustments
    assert result.decision_type == "hold"
    assert result.operation_advice == "Watch"


# --------------------------------------------------------------------------- entry order


def test_real_case_jpm_2026_11_24_secondary_above_ideal_is_note_only() -> None:
    result = _real_case(
        price=317.10,
        ideal="310.50 (Near key psychological support and analyst target low)",
        secondary="316.30 (Reclaim of MA5 with high volume confirmation)",
        stop="308.00 (Break below major technical support level)",
        target="326.30 (MA10 short-term resistance level)",
        position="2/10",
        risk_control="Strictly enforce a stop-loss at $308.00.",
    )

    apply_action_plan_guardrail(result, report_language="en")

    violations = result.dashboard["action_plan_validation"]["violations"]
    assert "secondary_entry_above_ideal" in violations
    assert result.decision_type == "hold" and result.confidence_level == "Medium"
    assert any(
        "secondary entry 316.30 is above the ideal entry 310.50" in note for note in _notes(result)
    )


def test_secondary_above_ideal_does_not_downgrade_buy() -> None:
    # entry 99, breakout add 101, stop 95, target 110 -> otherwise coherent.
    result = _result(secondary_buy="Secondary buy point: 101.00 (breakout)")

    apply_action_plan_guardrail(result, report_language="en")

    assert result.decision_type == "buy" and result.operation_advice == "Buy"
    assert result.dashboard["action_plan_validation"]["violations"] == ["secondary_entry_above_ideal"]
    assert len(_notes(result)) == 1


def test_secondary_below_ideal_has_no_order_note() -> None:
    check = evaluate_action_plan(_result())
    assert "secondary_entry_above_ideal" not in check.violations


# --------------------------------------------------------------------------- sell / reduce


def _sell(**kwargs) -> AnalysisResult:
    defaults = dict(
        decision_type="sell",
        operation_advice="Reduce",
        confidence_level="Medium",
        sentiment_score=30,
        ideal_buy="No buy point; wait for a reclaim of MA20 at 105.00",
        secondary_buy="N/A",
        stop_loss="Stop-loss: 95.00 (exit the rest below support)",
        take_profit="Reduce on a rebound to 104.00",
        suggested_position="0/10",
    )
    defaults.update(kwargs)
    return _result(**defaults)


def test_coherent_sell_plan_has_no_notes() -> None:
    result = _sell()
    assert apply_action_plan_guardrail(result, report_language="en") == []
    assert result.decision_type == "sell" and result.operation_advice == "Reduce"


def test_sell_plan_with_buy_entry_is_note_only() -> None:
    result = _sell(ideal_buy="Ideal buy point: 98.00 (near MA10)")
    assert apply_action_plan_guardrail(result, report_language="en") == ["action_plan_notes_added"]
    assert result.decision_type == "sell" and result.confidence_level == "Medium"
    assert result.dashboard["action_plan_validation"]["violations"] == ["sell_plan_has_buy_entry"]
    assert _notes(result) == [
        "Action plan check: the decision is sell/reduce but the plan lists a buy entry 98.00; "
        "do not open or add to a position."
    ]


def test_sell_plan_incoherent_stop_is_flagged() -> None:
    # Stop above the price and above the rebound target; risk control quotes another level.
    result = _sell(stop_loss="Stop-loss: 106.00", take_profit="104.00")
    result.dashboard["battle_plan"]["position_strategy"]["risk_control"] = "Exit below 97.00."
    apply_action_plan_guardrail(result, report_language="zh")
    assert result.dashboard["action_plan_validation"]["violations"] == [
        "hold_stop_loss_not_below_target_price",
        "sell_stop_at_or_above_current_price",
        "risk_control_stop_mismatch",
    ]
    assert "作战计划校验：止损 106.00 不低于现价 100.00；卖出/减仓结论下剩余仓位应已离场。" in _notes(result)


def test_sell_position_size_note_still_applies() -> None:
    result = _sell(suggested_position="3/10")
    apply_action_plan_guardrail(result, report_language="en")
    assert result.dashboard["action_plan_validation"]["violations"] == ["position_size_with_non_buy_decision"]
