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
    # Only the first 4 per dimension (as shown to the model); empty URL and duplicate skipped.
    assert urls == ["1", "3", "5", "6"]
    assert sources[0] == {
        "date": "2026-11-19",
        "source": "news.example.com",
        "title": "Headline 1 [x]",
        "url": "https://news.example.com/1",
        "undated": False,
    }


def test_dated_first_newest_first_and_undated_capped() -> None:
    intel = {
        "latest_news": SearchResponse(
            query="q",
            results=[_item(1, date=None), _item(2, date="2026-11-21"), _item(3, date=None), _item(4, date="2026-11-23")],
            provider="Tavily",
        ),
        "market_analysis": SearchResponse(
            query="q", results=[_item(5, date=None), _item(6, date="2026-11-22")], provider="Tavily"
        ),
    }
    sources = collect_news_sources(intel, limit=6, max_undated=2)
    assert [s["url"].rsplit("/", 1)[-1] for s in sources] == ["4", "6", "2", "1", "3"]
    assert [s["undated"] for s in sources] == [False, False, False, True, True]
    # Dated items always win the slots.
    assert len(collect_news_sources(intel, limit=2)) == 2
    assert all(not s["undated"] for s in collect_news_sources(intel, limit=2))


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
    items = collect_news_sources(_intel(), limit=2)
    items.append({"date": None, "source": "quotes.example.com", "title": "Quote page", "url": "https://quotes.example.com/q?a=(1)", "undated": True})
    payload = {"window_days": 3, "items": items}
    lines = NotificationService._news_sources_lines(_result(payload), get_report_labels("en"))
    assert lines[1] == "**📎 Sources** (publication dates, last 3 days):"
    # Domain inside the link text (no bare auto-linked domain).
    assert lines[2] == "- 2026-11-19 · [Headline 1 (x) — news.example.com](https://news.example.com/1)"
    assert lines[4] == "- undated · [Quote page — quotes.example.com](https://quotes.example.com/q?a=%281%29)"
    assert len(lines) == 5


def test_render_does_not_repeat_source_already_in_title() -> None:
    payload = {"window_days": 3, "items": [{"date": "2026-11-23", "source": "finance.yahoo.com", "title": "AAPL quote - finance.yahoo.com", "url": "https://finance.yahoo.com/quote/AAPL"}]}
    lines = NotificationService._news_sources_lines(_result(payload), get_report_labels("en"))
    assert lines[2] == "- 2026-11-23 · [AAPL quote - finance.yahoo.com](https://finance.yahoo.com/quote/AAPL)"


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


# --------------------------------------------------------------------------- low-quality sources


def test_low_quality_reason(monkeypatch) -> None:
    from src.utils.source_quality import low_quality_reason

    assert low_quality_reason("https://www.facebook.com/cnninternational/posts/1") == "blocked_domain"
    assert low_quality_reason("https://m.facebook.com/x") == "blocked_domain"
    assert low_quality_reason("https://www.benzinga.com:8443/quote/XOM/analyst-ratings") == "non_standard_port"
    assert (
        low_quality_reason("https://www.jdsupra.com/post/fileServer.aspx?fName=74247f44-3d6d.pdf")
        == "pdf_attachment"
    )
    # Exchange filings served as direct PDFs stay.
    assert low_quality_reason("https://www1.hkexnews.hk/listedco/listconews/sehk/2026/0613/example.pdf") is None
    assert low_quality_reason("https://www.benzinga.com/news/1") is None
    assert low_quality_reason("https://finance.yahoo.com/quote/XOM") is None
    monkeypatch.setenv("NEWS_SOURCE_BLOCKLIST", "settlementinsight.com, .spam.net")
    assert low_quality_reason("https://settlementinsight.com/apple-charger-lawsuit") == "blocked_domain"
    assert low_quality_reason("https://a.spam.net/x") == "blocked_domain"


def test_low_quality_sources_are_filtered_from_news_and_sources() -> None:
    from datetime import date

    today = date.today().isoformat()
    bad = [
        _item(1, url="https://www.facebook.com/oilnowgy/posts/1", date=today),
        _item(2, url="https://www.benzinga.com:8443/quote/XOM/analyst-ratings", date=today),
        _item(3, url="https://www.jdsupra.com/post/fileServer.aspx?fName=a.pdf", date=today),
    ]
    good = _item(4, url="https://www.reuters.com/markets/xom", date=today)
    response = SearchResponse(query="q", results=[*bad, good], provider="Tavily")
    service = SearchService.__new__(SearchService)
    filtered = service._filter_news_response(response, search_days=3, max_results=6, log_scope="test", keep_unknown=True)
    assert [item.url for item in filtered.results] == ["https://www.reuters.com/markets/xom"]
    sources = collect_news_sources({"latest_news": response})
    assert [entry["url"] for entry in sources] == ["https://www.reuters.com/markets/xom"]


# --------------------------------------------------------------------------- market review sources


def _market_analyzer(language: str):
    from unittest.mock import patch

    from src.config import Config
    from src.market_analyzer import MarketAnalyzer

    with patch("src.market_analyzer.DataFetcherManager"):
        analyzer = MarketAnalyzer(region="us", config=Config())
    analyzer._get_output_language = lambda: language
    return analyzer


def _market_news():
    return [
        _item(1, url="https://www.reuters.com/markets/us/stocks-1", date="2026-11-25"),
        _item(2, url="", date="2026-11-25"),
        _item(3, url="https://apnews.com/article/stocks (live)", date=""),
    ]


def test_market_prompt_numbers_news_and_requires_grounding() -> None:
    from src.market_analyzer import MarketOverview

    prompt = _market_analyzer("en")._build_review_prompt(MarketOverview(date="2026-11-25"), _market_news())
    assert "[N1] Headline 1 [x]" in prompt and "[N3] Headline 3 [x]" in prompt
    assert "News grounding: describe news events ONLY if they appear in [Market News]" in prompt
    zh = _market_analyzer("zh")._build_review_prompt(MarketOverview(date="2026-11-25"), _market_news())
    assert "新闻须有据" in zh


def test_market_review_sources_block_after_news_catalysts() -> None:
    from src.market_analyzer import MarketOverview

    review = (
        "## 2026-11-25 US Market Recap\n\n### 1. Market Summary\nx\n\n"
        "### 3. News Catalysts\nDip buying [N1].\n\n### 4. Outlook\ny\n"
    )
    out = _market_analyzer("en")._inject_data_into_review(review, MarketOverview(date="2026-11-25"), _market_news())
    catalysts, outlook = out.split("### 4. Outlook")
    assert "**📎 Sources** (news given to the model" in catalysts
    assert "- [N1] 2026-11-25 · [Headline 1 \\[x\\] — reuters.com](https://www.reuters.com/markets/us/stocks-1)" in catalysts
    assert "[N2]" not in catalysts  # no URL -> not listed
    assert "- [N3] undated · [Headline 3 \\[x\\] — apnews.com](https://apnews.com/article/stocks%20%28live%29)" in catalysts
    # Without the heading the block is appended at the end; without news nothing is added.
    plain = _market_analyzer("en")._inject_data_into_review("## Recap\ntext\n", MarketOverview(date="2026-11-25"), _market_news())
    assert plain.rstrip().endswith("(https://apnews.com/article/stocks%20%28live%29)")
    assert "Sources" not in _market_analyzer("en")._inject_data_into_review(review, MarketOverview(date="2026-11-25"), [])
