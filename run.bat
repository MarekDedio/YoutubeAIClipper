@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 goto :fail
)

echo Installing or updating required packages...
call .venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto :fail

echo Starting Speech Cutter...
call .venv\Scripts\python.exe app.py
goto :eof

:fail
echo.
echo Could not start Speech Cutter.
pause
