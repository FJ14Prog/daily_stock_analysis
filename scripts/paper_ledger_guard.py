#!/usr/bin/env python3
"""Decide whether a scheduled daily run should be skipped.

The daily workflow has a primary and a backup cron. The backup must only run
the analysis when the primary did not record the latest completed US session
in the paper ledger. Fail-open: any error means "run the analysis".

Writes ``skip=true|false`` and ``session=<date>`` to ``$GITHUB_OUTPUT`` when set.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def decide(ledger: Path, market: str = "us", now: Optional[datetime] = None) -> Tuple[bool, str]:
    """Return ``(skip, session_date)`` for the latest completed session."""
    try:
        from src.core.trading_calendar import get_effective_trading_date
        from src.services.paper_ledger import ledger_has_session

        session = get_effective_trading_date(market, current_time=now).isoformat()
        return ledger_has_session(ledger, session), session
    except Exception as exc:  # pragma: no cover - defensive fail-open
        print(f"guard error, running analysis: {exc}")
        return False, ""


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", default="paper-ledger/signals.jsonl")
    parser.add_argument("--market", default="us")
    args = parser.parse_args(argv)

    skip, session = decide(Path(args.ledger), args.market)
    if skip:
        print(f"Session {session} already in ledger; skipping this scheduled run.")
    else:
        print(f"Session {session or 'unknown'} not in ledger; running analysis.")

    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as handle:
            handle.write(f"skip={'true' if skip else 'false'}\n")
            handle.write(f"session={session}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
