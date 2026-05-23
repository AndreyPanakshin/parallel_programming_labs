@echo off
echo ========================================
echo   Matrix Multiplication - Compile (OpenMP)
echo ========================================
echo.

g++ -O3 -fopenmp -std=c++11 -o matrix_mult.exe main.cpp

if %errorlevel% equ 0 (
    echo.
    echo Compilation successful!
    echo File: matrix_mult.exe
    echo OpenMP: enabled
) else (
    echo.
    echo Compilation failed!
    echo Make sure MinGW with OpenMP support is installed
)
echo.
pause