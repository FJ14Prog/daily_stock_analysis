# -*- coding: utf-8 -*-
"""Primary model retried with waits on 503/429 before fallback; fallback results flagged."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.analyzer import (
    AnalysisResult,
    GeminiAnalyzer,
    _is_transient_llm_error,
    apply_llm_fallback_notice,
)
from src.config import Config, parse_llm_transient_retry_delays

RESPONSE = json.dumps(
    {"sentiment_score": 60, "trend_prediction": "Sideways", "operation_advice": "Watch", "analysis_summary": "ok"}
)
OVERLOAD = RuntimeError(
    'litellm.ServiceUnavailableError: GeminiException - {"error": {"code": 503, '
    '"message": "This model is currently experiencing high demand.", "status": "UNAVAILABLE"}}'
)


@pytest.fixture(autouse=True)
def _hmac_env(monkeypatch):
    monkeypatch.setenv("LLM_USAGE_HMAC_SECRET", "test-secret")
    monkeypatch.setenv("LLM_USAGE_HMAC_KEY_VERSION", "test-v1")


def _analyzer(delays):
    analyzer = GeminiAnalyzer.__new__(GeminiAnalyzer)
    analyzer._router = None
    analyzer._litellm_available = True
    analyzer._config_override = Config(
        litellm_model="primary-route",
        litellm_fallback_models=["backup-route"],
        gemini_request_delay=0,
        report_integrity_enabled=False,
        llm_transient_retry_delays=delays,
    )
    return analyzer


def _run(analyzer, failures_before_success):
    calls = []

    def dispatch(model, kwargs, **_):
        calls.append((model, bool(kwargs.get("stream"))))
        primary_calls = [c for c in calls if c[0] == "primary-route"]
        if model == "primary-route" and len(primary_calls) <= failures_before_success:
            raise OVERLOAD
        if kwargs.get("stream"):
            raise RuntimeError("stream not used in this test")
        return SimpleNamespace(
            model=model, usage=None,
            choices=[SimpleNamespace(message=SimpleNamespace(content=RESPONSE))],
        )

    with patch.object(analyzer, "_dispatch_litellm_completion", side_effect=dispatch), \
         patch.object(analyzer, "_get_analysis_system_prompt", return_value="system"), \
         patch.object(analyzer, "_get_skill_prompt_sections", return_value=(None, None, True)), \
         patch.object(analyzer, "_format_prompt", return_value="prompt"), \
         patch("src.analyzer.time.sleep") as sleep:
        result = analyzer.analyze({"code": "AAPL", "stock_name": "Apple"})
    return result, calls, [c.args[0] for c in sleep.call_args_list]


def test_overload_is_transient_but_generic_failure_is_not() -> None:
    assert _is_transient_llm_error(OVERLOAD)
    assert _is_transient_llm_error(RuntimeError("429 RESOURCE_EXHAUSTED"))
    assert not _is_transient_llm_error(RuntimeError("primary unavailable"))
    assert not _is_transient_llm_error(ValueError("invalid api key"))


def test_primary_recovers_after_wait_and_no_fallback_used() -> None:
    # stream (503) + non-stream (503) on the first attempt, then non-stream ok after 30 s
    result, calls, sleeps = _run(_analyzer([30.0, 60.0]), failures_before_success=2)
    assert result.success
    assert sleeps.count(30.0) == 1 and 60.0 not in sleeps
    assert all(model == "primary-route" for model, _ in calls)
    assert calls[-1] == ("primary-route", False)  # retry skips streaming
    assert result.llm_fallback is None


def test_falls_back_after_retries_exhausted_and_flags_result() -> None:
    result, calls, sleeps = _run(_analyzer([30.0, 60.0]), failures_before_success=99)
    assert result.success
    assert [s for s in sleeps if s in (30.0, 60.0)] == [30.0, 60.0]
    assert calls[-1][0] == "backup-route"
    # 1st attempt stream + non-stream, then 2 non-stream retries
    assert sum(1 for model, _ in calls if model == "primary-route") == 4
    assert result.llm_fallback == {"configured": "primary-route", "used": "backup-route"}


def test_retries_disabled() -> None:
    result, calls, sleeps = _run(_analyzer([]), failures_before_success=99)
    assert [s for s in sleeps if s in (30.0, 60.0)] == []
    assert calls[-1][0] == "backup-route"


def test_parse_delays() -> None:
    assert parse_llm_transient_retry_delays(None) == [30.0, 60.0]
    assert parse_llm_transient_retry_delays("") == [30.0, 60.0]
    assert parse_llm_transient_retry_delays("off") == []
    assert parse_llm_transient_retry_delays("0") == []
    assert parse_llm_transient_retry_delays("10, x, 900, -1") == [10.0, 300.0]


def test_fallback_notice_marks_result() -> None:
    result = AnalysisResult(
        code="NVDA", name="NVIDIA", sentiment_score=35, trend_prediction="Bearish",
        operation_advice="Watch", confidence_level="Medium", report_language="en",
        dashboard={"phase_decision": {"data_limitations": ["quote: fallback"]}},
        llm_fallback={"configured": "gemini/gemini-3.6-flash", "used": "gemini-3.5-flash-lite"},
    )
    assert apply_llm_fallback_notice(result) == ["llm_fallback_flagged"]
    assert result.confidence_level == "Low"
    notes = result.dashboard["phase_decision"]["data_limitations"]
    assert notes[0].startswith("⚠️ Generated by fallback model gemini-3.5-flash-lite because the primary model gemini-3.6-flash")
    assert notes[1] == "quote: fallback"
    # idempotent
    apply_llm_fallback_notice(result)
    assert len(result.dashboard["phase_decision"]["data_limitations"]) == 2


def test_no_notice_without_fallback() -> None:
    result = AnalysisResult(code="AAPL", name="Apple", sentiment_score=50, trend_prediction="x", operation_advice="Watch")
    assert apply_llm_fallback_notice(result) == []
