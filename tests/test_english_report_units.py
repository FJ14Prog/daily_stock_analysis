# -*- coding: utf-8 -*-
"""English reports must not leak Chinese index names or 万/亿/美元 units."""

from unittest.mock import patch

import pytest

from data_provider.us_index_mapping import get_us_index_english_name
from src.notification import NotificationService


@pytest.mark.parametrize(
    "value,expected",
    [
        ("SPX", "S&P 500"),
        ("^GSPC", "S&P 500"),
        ("标普500指数", "S&P 500"),
        ("^IXIC", "Nasdaq Composite"),
        ("纳斯达克综合指数", "Nasdaq Composite"),
        ("DJI", "Dow Jones Industrial Average"),
        ("道琼斯工业指数", "Dow Jones Industrial Average"),
        ("VIX恐慌指数", "CBOE Volatility Index (VIX)"),
        ("^RUT", "Russell 2000"),
        ("AAPL", None),
        ("", None),
        (None, None),
    ],
)
def test_us_index_english_name(value, expected):
    assert get_us_index_english_name(value) == expected


@pytest.mark.parametrize(
    "value,currency,language,expected",
    [
        # Values from the 2026-11-18 email (AAPL / MSFT / NVDA financial summary)
        (114_477_000_000, "USD", "en", "114.48B USD"),
        (30_711_000_000, "USD", "en", "30.71B USD"),
        (96_221_000_000, "USD", "en", "96.22B USD"),
        (-2_500_000, "USD", "en", "-2.50M USD"),
        (999_999_999, "USD", "en", "1.00B USD"),  # rounding rolls over to next unit
        (12_345, "HKD", "en", "12.35K HKD"),
        (512, None, "en", "512"),  # unknown currency: no guessed suffix
        (114_477_000_000, "USD", "ko", "114.48B USD"),
        # Chinese reports unchanged
        (114_477_000_000, "USD", "zh", "1144.77 亿美元"),
        (114_477_000_000, "USD", None, "1144.77 亿美元"),
        ("n/a", "USD", "en", "N/A"),
    ],
)
def test_format_amount_by_language(value, currency, language, expected):
    assert NotificationService._format_amount_cn(value, currency, language) == expected


@pytest.mark.parametrize(
    "value,currency,language,expected",
    [
        (1.07, "USD", "en", "1.0700 USD"),
        (0.52, "USD", "en", "0.5200 USD"),
        (1.07, None, "en", "1.0700"),
        (1.07, "USD", "zh", "1.0700 美元"),
        (None, "USD", "en", "N/A"),
    ],
)
def test_format_per_share_by_language(value, currency, language, expected):
    assert NotificationService._format_per_share(value, currency, language) == expected


def test_market_snapshot_volume_follows_report_language():
    from src.analyzer import GeminiAnalyzer

    analyzer = GeminiAnalyzer.__new__(GeminiAnalyzer)
    context = {
        "date": "2026-11-17",
        "today": {"close": 351.91, "open": 352.83, "high": 355.36, "low": 349.67, "volume": 37_341_600},
        "yesterday": {"close": 352.78},
    }
    en = analyzer._build_market_snapshot(context, "en")
    zh = analyzer._build_market_snapshot(context)
    assert en["volume"] == "37.34M shares"
    assert "万" not in en["volume"] and "股" not in en["volume"]
    assert zh["volume"] == "3734.16 万股"


def _overview_with_us_indices():
    from src.market_analyzer import MarketIndex, MarketOverview

    overview = MarketOverview(date="2026-11-17")
    overview.indices = [
        MarketIndex(code="SPX", name="标普500指数", current=7340.53, change_pct=-1.03),
        MarketIndex(code="^VIX", name="VIX恐慌指数", current=25.46, change_pct=11.13),
        MarketIndex(code="XYZ", name="Unknown Index", current=1.0, change_pct=0.0),
    ]
    return overview


def _analyzer(language):
    from src.config import Config
    from src.market_analyzer import MarketAnalyzer

    with patch("src.market_analyzer.DataFetcherManager"):
        analyzer = MarketAnalyzer(region="us", config=Config())
    analyzer._get_output_language = lambda: language
    return analyzer


def test_indices_block_english_names():
    block = _analyzer("en")._build_indices_block(_overview_with_us_indices())
    assert "| S&P 500 |" in block
    assert "| CBOE Volatility Index (VIX) |" in block
    assert "| Unknown Index |" in block  # unknown names left unchanged
    assert "标普" not in block and "恐慌" not in block


def test_indices_block_chinese_unchanged():
    block = _analyzer("zh")._build_indices_block(_overview_with_us_indices())
    assert "| 标普500指数 |" in block
    assert "| VIX恐慌指数 |" in block


def test_indices_block_keeps_provider_english_names():
    from src.market_analyzer import MarketIndex, MarketOverview

    overview = MarketOverview(date="2026-11-17")
    overview.indices = [MarketIndex(code="^VIX", name="Volatility", current=25.46, change_pct=11.13)]
    assert "| Volatility |" in _analyzer("en")._build_indices_block(overview)
