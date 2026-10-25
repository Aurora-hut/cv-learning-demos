@echo off
cd /d "%~dp0"
set PY="C:\Users\HUAWEI\.workbuddy\binaries\python\versions\3.13.12\python.exe"
if not exist %PY% set PY=python

echo [Demo3] Camera detection, low threshold (conf 0.05) + bigger model
echo Press S to save a snapshot (cam_snapshot.jpg), Q to quit.
echo.
%PY% demo3_detect.py --conf 0.05 --model models/yolo26s.onnx
echo.
pause
