# -*- coding: utf-8 -*-
"""Volume ratio / turnover rate display never shows None or fake zeros."""

from src.utils.data_processing import format_volume_metric


def test_missing_values_show_na() -> None:
    for value in (None, "None", "null", "", "N/A", "abc", 0, 0.0, "0.0", "0.0%", -1, float("nan"), True):
        assert format_volume_metric(value, "%") == "N/A", value


def test_real_values_keep_their_text() -> None:
    assert format_volume_metric(1.97) == "1.97"
    assert format_volume_metric("0.91") == "0.91"
    assert format_volume_metric(0.42, "%") == "0.42%"
    assert format_volume_metric("0.42%", "%") == "0.42%"
