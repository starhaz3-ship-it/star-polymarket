@echo off
REM ============================================================
REM SETUP: Daily Evolution Cycle + Auto-Analysis
REM RIGHT-CLICK THIS FILE -> "Run as administrator"
REM ============================================================

echo Setting up Star Polymarket automated evolution...
echo.

REM Daily Evolution Cycle at 5:00 AM MST (12:00 UTC)
schtasks /Create /TN "StarPolymarket-Evolution" /TR "C:\Users\Star\.local\bin\star-polymarket\run_evolution.bat" /SC DAILY /ST 05:00 /F /RL HIGHEST
echo [OK] Daily evolution cycle: 5:00 AM MST

REM Auto-Analysis every 6 hours
schtasks /Create /TN "StarPolymarket-Analyze-00" /TR "python C:\Users\Star\.local\bin\star-polymarket\auto_analyze.py" /SC DAILY /ST 00:00 /F
schtasks /Create /TN "StarPolymarket-Analyze-06" /TR "python C:\Users\Star\.local\bin\star-polymarket\auto_analyze.py" /SC DAILY /ST 06:00 /F
schtasks /Create /TN "StarPolymarket-Analyze-12" /TR "python C:\Users\Star\.local\bin\star-polymarket\auto_analyze.py" /SC DAILY /ST 12:00 /F
schtasks /Create /TN "StarPolymarket-Analyze-18" /TR "python C:\Users\Star\.local\bin\star-polymarket\auto_analyze.py" /SC DAILY /ST 18:00 /F
echo [OK] Auto-analysis: 12AM, 6AM, 12PM, 6PM MST

echo.
echo ============================================================
echo ALL DONE! Your bot will now evolve autonomously.
echo.
echo Schedule:
echo   Every 6h:  auto_analyze.py generates latest_analysis.json
echo   Daily 5AM: Claude wakes up, reads analysis, deploys improvements
echo   Always:    Live + Paper traders running 24/7
echo.
echo To check status:  schtasks /Query /TN "StarPolymarket-Evolution"
echo To run manually:  run_evolution.bat
echo To remove:        schtasks /Delete /TN "StarPolymarket-Evolution" /F
echo ============================================================
pause
