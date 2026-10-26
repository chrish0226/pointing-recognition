@echo off
setlocal
"%~dp0.venv_rtmw3d_trt\Scripts\python.exe" -u "%~dp0realtime_rtmw3d.py" %*
if errorlevel 1 (
    echo.
    echo Failed. Read the error above. Use the project TensorRT environment.
    pause
    exit /b 1
)
