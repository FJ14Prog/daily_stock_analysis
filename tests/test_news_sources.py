# -*- coding: utf-8 -*-
"""News sources: code-rendered links of what the model saw; publication-date rules."""

from __future__ import annotations

from src.analyzer import AnalysisResult
from src.notification import NotificationService
from src.report_language import get_report_labels
from src.search_service import SearchResponse, SearchResult, SearchService, collect_news_sources


def _item(n: int, url: str = None, date: str = "2026-11-19") -> SearchResult:
    return SearchResult(
        title=f"Headline {n} [x]",
        snippet="snippet " * 5,
        url=url if url is not None else f"https://news.example.com/{n}",
        source="news.example.com",
        published_date=date,
    )


def _intel():
    return {
        "risk_check": SearchResponse(query="q", results=[_item(5), _item(6)], provider="Tavily"),
        "latest_news": SearchResponse(
            query="q",
            results=[_item(1), _item(2, url=""), _item(3), _item(1), _item(4), _item(99)],
            provider="Tavily",
        ),
        "earnings": SearchResponse(query="q", results=[_item(7)], provider="Tavily", success=False),
    }


def test_collect_follows_prompt_order_cap_and_dedup() -> None:
    sources = collect_news_sources(_intel(), limit=10)
    urls = [s["url"].rsplit("/", 1)[-1] for s in sources]
    # latest_news first (only first 4 shown to the model; empty URL and duplicate skipped), then risk_check
    assert urls == ["1", "3", "5", "6"]
    assert sources[0] == {
        "date": "2026-11-19",
        "source": "news.example.com",
        "title": "Headline 1 [x]",
        "url": "https://news.example.com/1",
    }


def test_collect_respects_limit_and_empty_input() -> None:
    assert len(collect_news_sources(_intel(), limit=2)) == 2
    assert collect_news_sources(None) == []


def test_intel_report_states_publication_dates() -> None:
    service = SearchService.__new__(SearchService)
    text = SearchService.format_intel_report(service, _intel(), "Apple")
    assert "dates in brackets are publication dates" in text


def _result(sources):
    return AnalysisResult(
        code="AAPL", name="Apple", sentiment_score=55, trend_prediction="Sideways",
        operation_advice="Watch", report_language="en", news_sources=sources,
    )


def test_render_sources_block() -> None:
    payload = {"window_days": 3, "items": collect_news_sources(_intel(), limit=2)}
    lines = NotificationService._news_sources_lines(_result(payload), get_report_labels("en"))
    assert lines[1] == "**📎 Sources** (publication dates, last 3 days):"
    assert lines[2] == "- 2026-11-19 · news.example.com · [Headline 1 (x)](https://news.example.com/1)"
    assert len(lines) == 4


def test_render_nothing_without_sources() -> None:
    labels = get_report_labels("en")
    assert NotificationService._news_sources_lines(_result(None), labels) == []
    assert NotificationService._news_sources_lines(_result({"window_days": 3, "items": []}), labels) == []


def test_prompt_contains_publication_date_rule() -> None:
    from types import SimpleNamespace
    from unittest.mock import patch

    from src.analyzer import GeminiAnalyzer

    with patch.object(GeminiAnalyzer, "_init_litellm", return_value=None):
        analyzer = GeminiAnalyzer()
    context = {"code": "AAPL", "stock_name": "Apple", "date": "2026-11-20", "today": {}}
    fake_cfg = SimpleNamespace(news_max_age_days=3, news_strategy_profile="medium")
    with patch("src.analyzer.get_config", return_value=fake_cfg):
        en = analyzer._format_prompt(context, "Apple", news_context="news", report_language="en")
        zh = analyzer._format_prompt(context, "Apple", news_context="news", report_language="zh")
    assert "is the article's publication date, not necessarily when the event happened" in en
    assert "Never present an old event as new" in en
    assert "方括号中的日期为文章发布日期" in zh
