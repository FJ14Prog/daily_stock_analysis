#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Weekly paper-trading review of the signal ledger.

Reads ``signals.jsonl``, downloads unadjusted daily bars from yfinance for each
ticker (plus SPY as benchmark), writes ``evaluations.csv`` and ``summary.md``
next to the ledger and optionally emails the summary.

Usage:
    python scripts/evaluate_paper_ledger.py --ledger paper-ledger/signals.jsonl [--send-email]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.services.paper_evaluator import (  # noqa: E402
    Bar,
    EvalConfig,
    dedupe_ledger,
    evaluate_signal,
    render_summary_markdown,
    rows_to_csv,
    summarize,
)

logger = logging.getLogger("paper_evaluator")
BENCHMARK = "SPY"


def load_ledger(path: Path) -> List[dict]:
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning("Skipping malformed ledger line %d", number)
    return records


def fetch_bars(symbol: str, start: date) -> Tuple[List[Bar], List[date]]:
    """Unadjusted daily bars (dividends not added back) and split dates."""
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    frame = ticker.history(start=start.isoformat(), auto_adjust=False, actions=True)
    bars: List[Bar] = []
    splits: List[date] = []
    if frame is None or frame.empty:
        return bars, splits
    for index, row in frame.iterrows():
        day = index.date()
        try:
            bars.append(Bar(day, float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"])))
        except (KeyError, TypeError, ValueError):
            continue
        if float(row.get("Stock Splits", 0) or 0) not in (0.0, 1.0):
            splits.append(day)
    today = datetime.now(timezone.utc).date()
    # Today's bar may be partial while the US session is open; drop it.
    bars = [bar for bar in bars if bar.day < today]
    return bars, splits


def evaluate(records: Sequence[dict], config: EvalConfig) -> List[dict]:
    signals = dedupe_ledger(records)
    if not signals:
        return []
    start = min(date.fromisoformat(r["session_date"][:10]) for r in signals) - timedelta(days=5)
    benchmark_bars, _ = fetch_bars(BENCHMARK, start)
    cache: Dict[str, Tuple[List[Bar], List[date]]] = {}
    rows = []
    for record in signals:
        ticker = str(record["ticker"])
        if ticker not in cache:
            try:
                cache[ticker] = fetch_bars(ticker, start)
            except Exception as exc:  # keep evaluating the rest
                logger.warning("Bars unavailable for %s: %s", ticker, exc)
                cache[ticker] = ([], [])
        bars, splits = cache[ticker]
        rows.append(
            evaluate_signal(record, bars, config=config, split_days=splits, benchmark_bars=benchmark_bars)
        )
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", default="paper-ledger/signals.jsonl")
    parser.add_argument("--entry-valid-sessions", type=int, default=5)
    parser.add_argument("--max-hold-sessions", type=int, default=20)
    parser.add_argument("--direction-sessions", type=int, default=10)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--send-email", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    ledger = Path(args.ledger)
    if not ledger.exists():
        logger.info("Ledger %s not found; nothing to evaluate.", ledger)
        return 0
    config = EvalConfig(
        entry_valid_sessions=args.entry_valid_sessions,
        max_hold_sessions=args.max_hold_sessions,
        direction_sessions=args.direction_sessions,
        cost_bps=args.cost_bps,
    )
    rows = evaluate(load_ledger(ledger), config)
    as_of = datetime.now(timezone.utc).date().isoformat()
    summary_md = render_summary_markdown(summarize(rows), as_of=as_of, config=config)
    (ledger.parent / "evaluations.csv").write_text(rows_to_csv(rows), encoding="utf-8", newline="\n")
    (ledger.parent / "summary.md").write_text(summary_md, encoding="utf-8", newline="\n")
    logger.info("Evaluated %d signal(s).", len(rows))
    print(summary_md)

    if args.send_email:
        try:
            from src.config import get_config
            from src.notification_sender.email_sender import EmailSender

            sent = EmailSender(get_config()).send_to_email(
                summary_md, subject=f"📊 Paper Trading Review - {as_of}"
            )
            logger.info("Summary email sent: %s", sent)
        except Exception as exc:
            logger.warning("Summary email failed: %s", exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
