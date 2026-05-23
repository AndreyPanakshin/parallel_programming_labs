@echo off
echo ========================================
echo   Matrix Multiplication - Compile
echo ========================================
echo.

g++ -O2 -std=c++11 -o matrix_mult.exe main.cpp

if %errorlevel% equ 0 (
    echo.
    echo Compilation successful!
    echo    File: matrix_mult.exe
) else (
    echo.
    echo Compilation failed!
    echo    Make sure MinGW is installed and in PATH
)
echo.
pause