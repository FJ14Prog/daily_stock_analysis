# Paper-trading ledger

Append-only record of the daily signals, used to measure how the signals would
have performed before any real money is committed.

## How it works

- `src/services/paper_ledger.py` appends one JSON line per analysed stock to the
  file named by `PAPER_LEDGER_FILE` after each run (`main.py`, skipped with
  `--dry-run`). Unset → disabled. Write errors are logged and never fail the run.
- Only data known when the report was generated is recorded (no look-ahead):
  session date, decision/action/advice, score, confidence, price, ideal and
  secondary entry, stop-loss, target, risk/reward from the ideal entry, entry
  distance to price, action-plan guardrail violations, model, report language,
  `run_id`.
- A re-run for the same session appends a new line; evaluators should keep the
  last line per `(session_date, ticker)`.

## GitHub Actions

`00-daily-analysis.yml` checks out the `paper-ledger` branch into
`paper-ledger/` (optional, `continue-on-error`). When it exists the analysis step
sets `PAPER_LEDGER_FILE=paper-ledger/signals.jsonl` and a final step commits and
pushes new lines to that branch. Without the branch nothing changes.

The job needs `permissions: contents: write`. GitHub cannot scope that to one
branch; protect `main` with a branch rule if direct pushes must be blocked.

Create the branch once (orphan, unrelated to `main`):

```bash
git worktree add --orphan -b paper-ledger ../dsa-paper-ledger
cd ../dsa-paper-ledger
printf '# Paper ledger\n\nDaily signals (signals.jsonl). Never merge into main.\n' > README.md
git add README.md && git commit -m "chore: init paper ledger" && git push -u origin paper-ledger
cd - && git worktree remove ../dsa-paper-ledger
```

## Record example

```json
{"schema": 1, "session_date": "2026-11-18", "ticker": "AAPL", "decision": "hold", "action": "watch",
 "advice": "Watch", "score": 59, "price": 352.45, "ideal_buy": 353.66, "secondary_buy": 350.83,
 "stop_loss": 347.49, "take_profit": 356.42, "risk_reward": 0.4473, "entry_distance_pct": 0.3433,
 "plan_violations": ["risk_reward_below_minimum"], "downgraded": false, "model": "gemini/gemini-3.6-flash"}
```

## Weekly evaluation

`.github/workflows/paper-evaluation.yml` runs every Saturday (and on manual
dispatch). `scripts/evaluate_paper_ledger.py` reads `signals.jsonl`, downloads
daily bars from yfinance (`auto_adjust=False`: split-adjusted, dividends not
added back) for each ticker and SPY, and writes `evaluations.csv` (one row per
signal, recomputed from scratch each run) and `summary.md` to the
`paper-ledger` branch; the summary is emailed (`📊 Paper Trading Review - <date>`).
Logic lives in `src/services/paper_evaluator.py` (pure, unit-tested).

Rules (defaults; override with repository variables
`PAPER_EVAL_ENTRY_VALID_SESSIONS`, `PAPER_EVAL_MAX_HOLD_SESSIONS`,
`PAPER_EVAL_DIRECTION_SESSIONS`, `PAPER_EVAL_COST_BPS`):

- Last ledger line per `(session_date, ticker)`; only bars after the session
  date are used; today's (possibly partial) bar is ignored.
- `buy` → trade; `hold` with a coherent plan (stop < ideal < target) → "shadow"
  trade, to judge the downgrades; `sell` → direction check over 10 sessions.
- Entry: limit at `ideal_buy`, valid 5 sessions; fills at the open if it gaps
  below the limit, otherwise at the limit. Not filled after 5 sessions →
  `not_filled`.
- Exit: stop (at the open if it gaps through), target (at the open if it gaps
  above), stop first when both touch the same bar, target on the fill bar only
  if filled at the open, otherwise close after 20 sessions (`timeout`).
  Unfinished trades are `open` and excluded from statistics.
- Costs: 10 bps per side. R multiple = net P/L ÷ (ideal_buy − stop_loss).
- A split inside the window excludes the signal.
- Incoherent plans are excluded from the trade statistics (`excluded_plan`,
  reasons in the `plan_issues` CSV column): structural action-plan violations
  recorded by the guardrail (stop not below every entry or the target, target
  not above the entries, stop within 0.5 % of an entry), a stop not below a
  lower secondary entry, or a risk (ideal − stop) under 0.5 % of the entry.
  Such plans would produce meaningless or inflated R multiples; the summary
  shows how many were excluded per group.
- Summary per group: signals, closed, open, fill rate, win rate, expectancy
  (R), average net return, max drawdown of cumulative R, buy-and-hold of the
  same stock and SPY over the same window, exit breakdown; flagged when fewer
  than 30 closed trades.

Limitations: daily bars cannot order intraday events (conservative
assumptions above); shadow trades are hypothetical; results need months of
signals before they mean anything.

## Backup schedule guard

`00-daily-analysis.yml` has two crons: a primary one after the US close
(`7 22 * * 1-5`) and a backup a few hours later (`43 2 * * 2-6`). GitHub can
delay or drop scheduled runs, so the backup covers a missed or failed primary.

A light `guard` job runs first on scheduled triggers only:
`scripts/paper_ledger_guard.py` resolves the latest completed US session with
the trading calendar and checks whether `signals.jsonl` already holds it. If
so, the `analyze` job is skipped (no LLM, search or email). Manual runs always
proceed, and any guard failure fails open (the analysis runs).

Limitation: if a run records the ledger but fails to send the email, the
backup does not resend it.
