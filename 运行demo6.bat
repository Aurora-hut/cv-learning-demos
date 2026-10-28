@echo off
cd /d "%~dp0"
set PY="C:\Users\HUAWEI\.workbuddy\binaries\python\versions\3.13.12\python.exe"
if not exist %PY% set PY=python

echo [Demo6] Armor plate via light-bar pairing (pure OpenCV, no neural net)
echo.
echo 1) Synthetic demo (no camera needed) -^> demo6_out.jpg
%PY% demo6_armor.py --no-show
echo.
echo 2) Quantitative self-test (40 synthetic scenes)
%PY% demo6_armor.py --selftest --selftest-n 40
echo.
echo Done. Check demo6_out.jpg and docs\demo6\.
echo.
echo Other modes:
echo   Camera    : %PY% demo6_armor.py
echo   Your photo: %PY% demo6_armor.py --image your.jpg
echo   Lower thr : %PY% demo6_armor.py --image your.jpg --thr 150
echo.
pause
