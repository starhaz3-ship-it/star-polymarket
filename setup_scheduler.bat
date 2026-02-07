@echo off
echo Setting up auto-analysis scheduled task (requires admin)...
schtasks /Create /TN "StarPolymarket-AutoAnalyze" /TR "python C:\Users\Star\.local\bin\star-polymarket\auto_analyze.py" /SC DAILY /ST 00:00 /F
schtasks /Create /TN "StarPolymarket-AutoAnalyze-06" /TR "python C:\Users\Star\.local\bin\star-polymarket\auto_analyze.py" /SC DAILY /ST 06:00 /F
schtasks /Create /TN "StarPolymarket-AutoAnalyze-12" /TR "python C:\Users\Star\.local\bin\star-polymarket\auto_analyze.py" /SC DAILY /ST 12:00 /F
schtasks /Create /TN "StarPolymarket-AutoAnalyze-18" /TR "python C:\Users\Star\.local\bin\star-polymarket\auto_analyze.py" /SC DAILY /ST 18:00 /F
echo Done! Auto-analysis will run at 00:00, 06:00, 12:00, 18:00 local time.
pause
