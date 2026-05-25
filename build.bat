@echo off
title CUDA Matrix Multiplication Build

echo ========================================
echo   Building CUDA Matrix Multiplication
echo ========================================
echo.

where nvcc >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] nvcc not found in PATH!
    echo Please install CUDA Toolkit from:
    echo https://developer.nvidia.com/cuda-downloads
    pause
    exit /b 1
)

echo Compiling with NVCC...
nvcc -O3 -arch=sm_61 -o gpu_matmul.exe gpu_matmul.cu

if %errorlevel% equ 0 (
    echo.
    echo [SUCCESS] Compilation complete!
    echo   Output: gpu_matmul.exe
    echo.
    echo Quick test:
    echo   gpu_matmul.exe 400 16 16 0 10 0
) else (
    echo.
    echo [FAILED] Compilation error!
    echo Check CUDA installation and GPU compatibility
)

echo.
pause