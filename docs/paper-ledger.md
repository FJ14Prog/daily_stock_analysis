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
