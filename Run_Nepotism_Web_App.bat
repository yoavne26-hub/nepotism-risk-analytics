@echo off
setlocal

cd /d "%~dp0"
set "APP_URL=http://127.0.0.1:8000/?ui=enterprise"
set "HEALTH_URL=http://127.0.0.1:8000/api/health"
set "CHROME_EXE="

if not exist "backend\api.py" (
    echo Could not find backend\api.py in:
    echo %CD%
    pause
    exit /b 1
)

if not exist "frontend\index.html" (
    echo Could not find frontend\index.html in:
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

if exist "%ProgramFiles%\Google\Chrome\Application\chrome.exe" (
    set "CHROME_EXE=%ProgramFiles%\Google\Chrome\Application\chrome.exe"
) else if exist "%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe" (
    set "CHROME_EXE=%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"
) else if exist "%LocalAppData%\Google\Chrome\Application\chrome.exe" (
    set "CHROME_EXE=%LocalAppData%\Google\Chrome\Application\chrome.exe"
)

echo Starting Nepotism Risk Analytics enterprise web app...
echo The new analyst UI will be available at %APP_URL%
echo.

if defined CHROME_EXE (
    start "" powershell -NoProfile -Command "$url = $env:APP_URL; $health = $env:HEALTH_URL; $chrome = $env:CHROME_EXE; for ($i = 0; $i -lt 60; $i++) { try { $resp = Invoke-WebRequest -UseBasicParsing -Uri $health -TimeoutSec 2; if ($resp.StatusCode -ge 200) { Start-Process -FilePath $chrome -ArgumentList $url; exit 0 } } catch { } Start-Sleep -Seconds 1 }; Start-Process -FilePath $chrome -ArgumentList $url"
) else (
    start "" powershell -NoProfile -Command "$url = $env:APP_URL; $health = $env:HEALTH_URL; for ($i = 0; $i -lt 60; $i++) { try { $resp = Invoke-WebRequest -UseBasicParsing -Uri $health -TimeoutSec 2; if ($resp.StatusCode -ge 200) { Start-Process $url; exit 0 } } catch { } Start-Sleep -Seconds 1 }; Start-Process $url"
)

call %PYTHON_CMD% -m uvicorn backend.api:app --host 127.0.0.1 --port 8000
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo The web app exited with code %EXIT_CODE%.
    pause
)

endlocal
