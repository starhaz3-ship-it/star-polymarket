@echo off
REM ============================================================
REM Star Polymarket - Daily Evolution Cycle
REM Runs Claude Code headlessly to analyze and improve the bot.
REM Schedule via Windows Task Scheduler to run daily at 5 AM MST.
REM ============================================================

cd /d C:\Users\Star\.local\bin\star-polymarket

REM Step 1: Run auto-analysis first (generates fresh latest_analysis.json)
echo [%date% %time%] Running auto-analysis...
python auto_analyze.py

REM Step 2: Run Claude Code with evolution prompt
echo [%date% %time%] Starting Claude evolution cycle...

REM Read the evolution prompt from file and pass to claude
set /p PROMPT=<nul
C:\Users\Star\.local\bin\claude.exe -p --model sonnet --allowedTools "Edit Write Read Bash Glob Grep" --max-budget-usd 2.00 "Read the file C:\Users\Star\.local\bin\star-polymarket\evolution_prompt.md and execute every step in it. You have full autonomy. Log all changes to evolution_log.md." > C:\Users\Star\.local\bin\star-polymarket\evolution_output.log 2>&1

echo [%date% %time%] Evolution cycle complete. See evolution_output.log for details.
