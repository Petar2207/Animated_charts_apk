@echo off
setlocal

REM Go to the folder where this BAT file lives (project root)
cd /d "%~dp0"

echo === Plot Animator ===
echo.

where py >nul 2>nul
if errorlevel 1 (
  echo ERROR: Python is not installed.
  echo Please install Python from https://www.python.org
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  py -m venv .venv
)

call ".venv\Scripts\activate"

echo Installing dependencies...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo Starting application...
python app.py

pause
endlocal

