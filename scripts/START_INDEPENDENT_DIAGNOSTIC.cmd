@echo off
cd /d "%~dp0.."
set "PYTHONPATH=.sdk_1_3;src"
echo Independent dual camera diagnostics. Click window: SPACE start, SPACE stop, Q quit.
if not defined PINGPONG_PYTHON if exist "%USERPROFILE%\anaconda3\python.exe" set "PINGPONG_PYTHON=%USERPROFILE%\anaconda3\python.exe"
if not defined PINGPONG_PYTHON set "PINGPONG_PYTHON=python"
"%PINGPONG_PYTHON%" -u tools\dual_capture_isolated.py
if errorlevel 1 pause
