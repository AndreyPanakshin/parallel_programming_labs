@echo off
setlocal enabledelayedexpansion

echo ========================================
echo   Matrix Multiplication - OpenMP Benchmark
echo ========================================
echo.

REM Заголовок CSV
echo size,threads,time_sec,operations,data_kb > stats.csv

echo Starting tests...
echo.

for %%S in (200 400 800 1200) do (
    echo [Size: %%S x %%S]
    
    REM Генерация матриц - подавляем вывод
    matrix_mult.exe generate input.txt %%S >nul 2>&1
    
    REM Последовательная версия
    echo   Running sequential...
    matrix_mult.exe seq input.txt output_seq.txt > temp_seq.txt 2>&1
    for /f "tokens=3" %%t in (temp_seq.txt) do set "SEQ_TIME=%%t"
    echo     Seq time: !SEQ_TIME! s
    del temp_seq.txt 2>nul
    
    REM Параллельная версия с разным количеством потоков
    for %%T in (1 2 4 8) do (
        echo   Testing with %%T threads...
        
        matrix_mult.exe run input.txt output.txt %%T > temp_par.txt 2>&1
        for /f "tokens=3" %%t in (temp_par.txt) do set "TIME=%%t"
        
        REM Вычисляем объём задачи
        set /a "SIZE_SQ=%%S * %%S"
        set /a "OPS=!SIZE_SQ! * (2 * %%S - 1)"
        set /a "DATA_KB=3 * !SIZE_SQ! * 8 / 1024"
        
        REM Записываем в CSV
        echo %%S,%%T,!TIME!,!OPS!,!DATA_KB! >> stats.csv
        
        echo     Time: !TIME! s
        
        REM Проверка
        python verify.py input.txt output.txt > temp_verify.txt 2>&1
        findstr /C:"PASS" temp_verify.txt >nul
        if !errorlevel! equ 0 (
            echo     Verification: PASS
        ) else (
            echo     Verification: FAIL
        )
        del temp_verify.txt 2>nul
    )
    
    echo ----------------------------------------
)

REM Очистка временных файлов
del temp_seq.txt temp_par.txt 2>nul

echo.
echo Results saved to stats.csv
echo.
pause