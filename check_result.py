#!/usr/bin/env python3
"""
Скрипт верификации для CUDA умножения матриц
Сравнение GPU-результата с эталоном NumPy
"""

import numpy as np
import sys

def parse_input_matrices(path):
    """Чтение A и B из входного файла"""
    with open(path, 'r', encoding='utf-8') as f:
        lines = [ln.strip() for ln in f if not ln.startswith('#') and ln.strip()]
    
    sz = int(lines[0])
    numbers = []
    for ln in lines[1:]:
        numbers += [float(x) for x in ln.split()]
    
    if len(numbers) >= 2 * sz * sz:
        A = np.array(numbers[:sz*sz]).reshape(sz, sz)
        B = np.array(numbers[sz*sz:2*sz*sz]).reshape(sz, sz)
        return A, B, sz
    return None, None, sz

def parse_output_matrix(path):
    """Чтение результирующей матрицы"""
    with open(path, 'r', encoding='utf-8') as f:
        lines = [ln.strip() for ln in f if not ln.startswith('#') and ln.strip()]
    
    sz = int(lines[0])
    mat = []
    for ln in lines[1:sz+1]:
        mat.append([float(x) for x in ln.split()])
    
    return np.array(mat)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python validate.py <input_file> <output_file>")
        sys.exit(1)
    
    inp = sys.argv[1]
    out = sys.argv[2]
    
    A, B, sz = parse_input_matrices(inp)
    if A is None:
        print("[ERROR] Could not parse input matrices")
        sys.exit(1)
    
    computed = parse_output_matrix(out)
    expected = A @ B
    
    max_diff = np.max(np.abs(computed - expected))
    
    print(f"Matrix size: {sz}x{sz}")
    print(f"Maximum difference: {max_diff:.2e}")
    
    if max_diff < 1e-3:
        print("[OK] Verification PASSED")
        sys.exit(0)
    else:
        print("[FAIL] Verification FAILED")
        sys.exit(1)