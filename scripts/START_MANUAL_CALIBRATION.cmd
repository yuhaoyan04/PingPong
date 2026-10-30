@echo off
cd /d "%~dp0.."
set "PYTHONPATH=%CD%\.sdk_1_3;%CD%\src"
if not defined PINGPONG_PYTHON if exist "%USERPROFILE%\anaconda3\python.exe" set "PINGPONG_PYTHON=%USERPROFILE%\anaconda3\python.exe"
if not defined PINGPONG_PYTHON set "PINGPONG_PYTHON=python"
"%PINGPONG_PYTHON%" -u tools\manual_calibration_capture.py
pause
