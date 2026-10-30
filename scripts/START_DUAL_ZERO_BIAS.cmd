@echo off
cd /d "%~dp0.."
set "PYTHONPATH=src"
set "BATCH_US=%~1"
if not defined BATCH_US set "BATCH_US=1000"
echo Two-camera preview: click the window, press SPACE to start, SPACE again to stop and save.
echo Preview accumulation: %BATCH_US% microseconds. Press Q or Esc to exit.
if not defined PINGPONG_PYTHON if exist "%USERPROFILE%\anaconda3\python.exe" set "PINGPONG_PYTHON=%USERPROFILE%\anaconda3\python.exe"
if not defined PINGPONG_PYTHON set "PINGPONG_PYTHON=python"
"%PINGPONG_PYTHON%" -u tools\live_ball_capture.py --manual-start --manual-stop --batch-us %BATCH_US% --preview-decay 0 --all-bias-zero --output-dir output\zero_bias_capture
if errorlevel 1 pause
