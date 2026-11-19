# -*- coding: utf-8 -*-
"""Localized section headings used when main.py pushes/merges reports."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import main


def test_english_headings() -> None:
    config = SimpleNamespace(report_language="en")
    assert main._push_section_heading(config, "market_review") == "# 📈 Market Review"
    assert main._push_section_heading(config, "dashboard") == "# 🚀 Stock Decision Dashboard"


def test_chinese_headings_unchanged() -> None:
    config = SimpleNamespace(report_language="zh")
    assert main._push_section_heading(config, "market_review") == "# 📈 大盘复盘"
    assert main._push_section_heading(config, "dashboard") == "# 🚀 个股决策仪表盘"


def test_non_string_language_falls_back_to_chinese() -> None:
    assert main._push_section_heading(MagicMock(), "market_review") == "# 📈 大盘复盘"
