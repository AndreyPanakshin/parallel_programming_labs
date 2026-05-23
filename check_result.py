#!/usr/bin/env python3
"""
OpenMP Matrix Multiplication Validator
Compares C++ results with NumPy reference
"""

import numpy as np
import sys

def parse_input(filepath):
    with open(filepath) as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith('#')]
    n = int(lines[0])
    values = []
    for ln in lines[1:]:
        values.extend([float(x) for x in ln.split()])
    if len(values) >= 2 * n * n:
        A = np.array(values[:n*n]).reshape(n, n)
        B = np.array(values[n*n:2*n*n]).reshape(n, n)
        return A, B, n
    return None, None, n

def parse_output(filepath):
    with open(filepath) as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith('#')]
    n = int(lines[0])
    mat = []
    for ln in lines[1:n+1]:
        mat.append([float(x) for x in ln.split()])
    return np.array(mat)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python validate.py <input_file> <output_file>")
        sys.exit(1)

    A, B, dim = parse_input(sys.argv[1])
    computed = parse_output(sys.argv[2])
    expected = A @ B
    max_error = np.max(np.abs(computed - expected))

    print(f"[VALID] Matrix size: {dim}x{dim}")
    print(f"[VALID] Max absolute error: {max_error:.2e}")
    print("[VALID] Status: " + ("PASS" if max_error < 1e-6 else "FAIL"))