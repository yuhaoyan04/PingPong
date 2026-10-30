@echo off
cd /d "%~dp0.."
set "PYTHONPATH=src"
echo Role swap: MASTER F3E, SLAVE F59. Click preview; SPACE starts and stops recording.
if not defined PINGPONG_PYTHON if exist "%USERPROFILE%\anaconda3\python.exe" set "PINGPONG_PYTHON=%USERPROFILE%\anaconda3\python.exe"
if not defined PINGPONG_PYTHON set "PINGPONG_PYTHON=python"
"%PINGPONG_PYTHON%" -u tools\live_ball_capture.py --master DSP01F1FFF3E --slave DSP01F1FFF59 --manual-start --manual-stop --batch-us 1000 --preview-decay 0 --all-bias-zero --output-dir output\role_swap_capture
if errorlevel 1 pause
