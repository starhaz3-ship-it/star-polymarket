import subprocess
r = subprocess.run(
    ['powershell', '-Command',
     'Get-CimInstance Win32_Process -Filter "Name=\'python.exe\'" | Select-Object ProcessId, CommandLine | Format-Table -Wrap -AutoSize'],
    capture_output=True, text=True
)
print(r.stdout[:3000])
