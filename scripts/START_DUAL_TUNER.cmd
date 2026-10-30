@echo off
cd /d "%~dp0.."
if not defined PINGPONG_PYTHON if exist "%USERPROFILE%\anaconda3\python.exe" set "PINGPONG_PYTHON=%USERPROFILE%\anaconda3\python.exe"
if not defined PINGPONG_PYTHON set "PINGPONG_PYTHON=python"
"%PINGPONG_PYTHON%" tools\dual_camera_tuner.py
if errorlevel 1 pause
