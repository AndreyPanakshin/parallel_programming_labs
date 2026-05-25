#!/bin/bash

echo "=============================================="
echo "  MPI Умножение матриц - Бенчмарк"
echo "=============================================="
echo ""

echo "size,processes,time_sec,operations,data_kb" > results.csv

echo "Запуск тестов..."
echo ""

for SIZE in 200 400 800 1200 1600 2000; do
    echo "[Размер: ${SIZE} x ${SIZE}]"
    
    # Генерация матриц
    mpirun --allow-run-as-root -n 1 ./matrix_mpi gen data/matrices.txt $SIZE > /dev/null 2>&1
    echo "  Сгенерировано: data/matrices.txt"
    
    for PROCS in 1 2 4 8; do
        # Запуск MPI
        mpirun --allow-run-as-root -n $PROCS ./matrix_mpi calc data/matrices.txt output/result.txt 2>/dev/null | tail -1 > .tmp_output
        
        TIME=$(cat .tmp_output | awk '{print $3}')
        
        if [ -z "$TIME" ] || [ "$TIME" = "0" ]; then
            TIME=$(grep -E "^[0-9]+ [0-9]+ [0-9]" .tmp_output | awk '{print $3}')
        fi
        
        if [ -z "$TIME" ]; then
            TIME="0"
        fi
        
        OPS=$((SIZE * SIZE * (2 * SIZE - 1)))
        DATA_KB=$((3 * SIZE * SIZE * 8 / 1024))
        
        echo "${SIZE},${PROCS},${TIME},${OPS},${DATA_KB}" >> results.csv
        
        echo "  Процессов: ${PROCS} | Время: ${TIME} с"
        
        if [ $PROCS -eq 1 ]; then
            python3 check_result.py data/matrices.txt output/result.txt 2>/dev/null
        fi
    done
    
    rm -f .tmp_output
    echo "----------------------------------------------"
done

echo ""
echo "Результаты сохранены в: results.csv"
echo ""