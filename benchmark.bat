@echo off
setlocal enabledelayedexpansion

echo ========================================
echo   Matrix Multiplication - Benchmark
echo ========================================
echo.

REM Записываем заголовок CSV
echo size,time_sec,operations,data_kb > stats.csv

echo Starting tests...
echo.

for %%S in (32 64 128 256 512) do (
    echo [%%S x %%S]
    
    REM Генерация матриц
    matrix_mult.exe generate input.txt %%S
    
    REM Запуск и получение времени
    for /f "tokens=1,2" %%a in ('matrix_mult.exe run input.txt output.txt') do (
        set "TIME=%%b"
    )
    
    REM Вычисляем операции и данные
    set /a "SIZE_SQ=%%S * %%S"
    set /a "OPS=!SIZE_SQ! * (2 * %%S - 1)"
    set /a "DATA_KB=3 * !SIZE_SQ! * 8 / 1024"
    
    REM Записываем в CSV
    echo %%S,!TIME!,!OPS!,!DATA_KB! >> stats.csv
    
    REM Вывод на экран
    echo   Time: !TIME! s ^| Ops: !OPS! ^| Data: !DATA_KB! KB
    
    REM Верификация
    python verify.py input.txt output.txt
    echo ----------------------------------------
)

echo.
echo Tests completed!
echo Results saved to stats.csv
echo.
pause