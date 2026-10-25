@echo off
cd /d "%~dp0"
set PY="C:\Users\HUAWEI\.workbuddy\binaries\python\versions\3.13.12\python.exe"
if not exist %PY% set PY=python

echo [Demo5] Detect + optical-flow tracking
echo.
%PY% demo5_track.py --video demo5_clip.mp4
echo.
echo Done. Open demo5_out.mp4 to see the result.
echo Webcam realtime:
echo   %PY% demo5_track.py
echo.
pause
