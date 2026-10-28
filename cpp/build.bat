@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

set "MSYS=D:\msys64"
set "MSYSTEM=UCRT64"
set "PATH=%MSYS%\ucrt64\bin;%MSYS%\usr\bin;%PATH%"

echo === Build C++ OpenCV demo ===

if not exist "%MSYS%\ucrt64\bin\g++.exe" (
  echo [ERROR] g++ not found at %MSYS%\ucrt64\bin\g++.exe
  echo         Install MSYS2 toolchain first.
  pause
  exit /b 1
)

set "CVPKG=opencv5"
set "CVFLAGS="
for /f "usebackq delims=" %%i in (`pkg-config --cflags --libs opencv5`) do set "CVFLAGS=%%i"
if "%CVFLAGS%"=="" (
  set "CVPKG=opencv4"
  for /f "usebackq delims=" %%i in (`pkg-config --cflags --libs opencv4`) do set "CVFLAGS=%%i"
)

if "%CVFLAGS%"=="" (
  echo [ERROR] pkg-config cannot find opencv5 / opencv4.
  echo         Try: pacman -S mingw-w64-ucrt-x86_64-opencv
  pause
  exit /b 1
)
echo Using package: %CVPKG%

echo Compiling demo1_cpp.cpp ...
g++ -O2 -std=c++17 demo1_cpp.cpp -o demo1_cpp.exe %CVFLAGS%
if errorlevel 1 (
  echo [ERROR] compile failed.
  pause
  exit /b 1
)

echo Compiled OK: demo1_cpp.exe
echo.
echo --- Running ---
echo.
demo1_cpp.exe %1
echo.
echo Done. Output image: cpp_out.jpg
pause
