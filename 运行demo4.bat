@echo off
cd /d "%~dp0"
set PY="C:\Users\HUAWEI\.workbuddy\binaries\python\versions\3.13.12\python.exe"
if not exist %PY% set PY=python

echo [Demo4] Target lock + aim offset demo
echo.
%PY% demo4_aim.py --image demo3_test.jpg
echo.
echo Done. Open demo4_out.jpg to see the result.
echo For video mode:
echo   %PY% demo4_aim.py --video demo4_pan.mp4
echo For webcam realtime:
echo   %PY% demo4_aim.py
echo.
pause
