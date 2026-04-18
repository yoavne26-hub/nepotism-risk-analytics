@echo off
setlocal

cd /d "%~dp0"

if not exist "backend\api.py" (
    echo Could not find backend\api.py in:
    echo %CD%
    pause
    exit /b 1
)

if exist ".venv\Scripts\python.exe" (
    set "PYTHON_CMD=.venv\Scripts\python.exe"
) else (
    echo Could not find the project virtual environment in:
    echo %CD%\.venv
    echo.
    echo Create or restore the virtual environment before running the web app.
    pause
    exit /b 1
)

echo Starting Nepotism Risk Analytics web app...
echo The web interface will be available at http://127.0.0.1:8000
echo.

call %PYTHON_CMD% -m uvicorn backend.api:app --host 127.0.0.1 --port 8000
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo The web app exited with code %EXIT_CODE%.
    pause
)

endlocal
