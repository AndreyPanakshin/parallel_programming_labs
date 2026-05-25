#!/usr/bin/env python3
"""
Проверка корректности умножения матриц MPI.
Сравнение с эталонным вычислением через NumPy.
"""

import numpy as np
import sys

def load_matrices(filename):
    """Загрузка матриц A и B из входного файла"""
    with open(filename, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if not l.startswith('#') and l.strip()]
    
    n = int(lines[0])
    data = []
    for line in lines[1:]:
        data += [float(x) for x in line.split()]
    
    if len(data) >= 2 * n * n:
        A = np.array(data[:n*n]).reshape(n, n)
        B = np.array(data[n*n:2*n*n]).reshape(n, n)
        return A, B, n
    return None, None, n

def load_result_matrix(filename):
    """Загрузка результирующей матрицы из выходного файла"""
    with open(filename, 'r') as f:
        lines = [l.strip() for l in f if not l.startswith('#') and l.strip()]
    
    n = int(lines[0])
    matrix = []
    for line in lines[1:n+1]:
        matrix.append([float(x) for x in line.split()])
    
    return np.array(matrix)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Использование: python3 check_result.py <input_file> <result_file>")
        sys.exit(1)
    
    inp = sys.argv[1]
    res = sys.argv[2]
    
    A, B, n = load_matrices(inp)
    if A is None:
        print("[ОШИБКА] Не удалось прочитать входные матрицы")
        sys.exit(1)
    
    C_computed = load_result_matrix(res)
    C_expected = A @ B
    
    max_diff = np.max(np.abs(C_computed - C_expected))
    
    print(f"Размер матрицы: {n}x{n}")
    print(f"Максимальное отклонение: {max_diff:.2e}")
    
    if max_diff < 1e-6:
        print("РЕЗУЛЬТАТ: УСПЕШНО ✓")
        sys.exit(0)
    else:
        print("РЕЗУЛЬТАТ: ОШИБКА ✗")
        sys.exit(1)