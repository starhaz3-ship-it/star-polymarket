# EVOLUTION PROTOCOL — Autonomous Daily Improvement Cycle
# This prompt is fed to Claude Code via `claude -p` for headless execution.

You are running the **daily evolution cycle** for Star's Polymarket trading bot.
You have FULL AUTONOMY to research, optimize, and deploy improvements.
Working directory: C:\Users\Star\.local\bin\star-polymarket

## STEP 1: Health Check
- Check if live trader (`run_ta_live.py`) is running. Look for recent output in `ta_live_output.log`.
- Check if paper trader (`run_ta_paper.py`) is running. Look for recent output in `ta_paper_output.log`.
- If either is down, restart it using PowerShell Start-Process.
- Check for errors in `ta_live_error.log` and `ta_paper_error.log`.

## STEP 2: Read Fresh Data
- Read `latest_analysis.json` (auto-generated every 6h by auto_analyze.py).
- Read `ta_live_results.json` for live trade history.
- Read `ta_paper_results.json` for paper trade history + skip_hour_shadows.
- Read `RESEARCH_BACKLOG.md` for hypotheses to test.
- Read memory files in `C:\Users\Star\.claude\projects\C--Users-Star--local-bin\memory\`

## STEP 3: Performance Audit
- Compare live WR vs paper WR. If live is >10% lower, investigate why.
- Check for losing price buckets, assets, or hours in live data.
- If any strategy/bucket has <40% WR with 10+ trades, flag for removal.
- Check hourly_stats: any hour with <40% WR and 5+ trades → reduce multiplier.

## STEP 4: Skip-Hour Shadow Verdicts
- Check `skip_hour_stats` in paper results.
- Any skip hour with 55%+ WR and 10+ shadow trades → REOPEN it in both live and paper.
- Any skip hour with <40% WR and 10+ trades → confirmed bad, note in failed_experiments.

## STEP 5: Research
- Pick the top untested hypothesis from RESEARCH_BACKLOG.md.
- Test it against available data (paper trades, live trades).
- Mark it as TESTED with results.
- If proven beneficial, deploy to live trader immediately.

## STEP 6: Deploy Improvements
- Edit `run_ta_live.py` and/or `run_ta_paper.py` with improvements.
- Restart the affected trader(s).
- Verify they start cleanly (check logs after 15s).

## STEP 7: Update Knowledge Base
- Append findings to `evolution_log.md` with date and rationale.
- Update `proven_edges.md` if new edges discovered.
- Update `failed_experiments.md` if experiments failed.
- Update `active_hypotheses.md` with current status.
- Update `RESEARCH_BACKLOG.md` — mark tested items, add new ones.

## STEP 8: Git Commit
- `git add` changed files and commit with descriptive message.
- `git push` to remote.

## RULES
- NEVER launch copy traders (run_copy_live.py, run_copy_k9Q2.py).
- NEVER increase position size beyond $8 max without Star's approval.
- NEVER remove proven edge filters (death zone, NYU model, etc.) without 100+ trade evidence.
- Always restart traders after code changes.
- Be conservative with live changes — test on paper first when possible.
- Log EVERYTHING you change in evolution_log.md.
