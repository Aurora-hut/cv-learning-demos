@echo off
cd /d "%~dp0"
set PY="C:\Users\HUAWEI\.workbuddy\binaries\python\versions\3.13.12\python.exe"
if not exist %PY% set PY=python

echo [Demo7] PnP pose estimation: distance + yaw/pitch/roll
echo.
echo 1) Synthetic 3D demo with known ground truth
%PY% demo7_pnp.py --no-show
echo.
echo 2) Quantitative self-test (zero-noise baseline / distance bands / noise sweep / ambiguity)
%PY% demo7_pnp.py --selftest
echo.
echo Done. Check demo7_out.jpg
echo.
echo Other modes:
echo   Camera     : %PY% demo7_pnp.py
echo   Your photo : %PY% demo7_pnp.py --image your.jpg
echo   With calib : %PY% demo7_pnp.py --calib camera.json
echo.
echo Camera calibration (run this first for real distance):
echo   %PY% calibrate_camera.py --make-board
echo   %PY% calibrate_camera.py
echo.
pause
