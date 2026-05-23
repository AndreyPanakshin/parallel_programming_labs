#!/usr/bin/env python3
"""
Matrix multiplication verifier using NumPy.
"""

import numpy as np
import sys

def load_input(filepath):
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

def load_result(filepath):
    with open(filepath) as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith('#')]
    n = int(lines[0])
    mat = []
    for ln in lines[1:n+1]:
        mat.append([float(x) for x in ln.split()])
    return np.array(mat)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python check_result.py <input_file> <output_file>")
        sys.exit(1)

    A, B, dim = load_input(sys.argv[1])
    computed = load_result(sys.argv[2])
    expected = A @ B
    max_err = np.max(np.abs(computed - expected))

    print(f"[VERIFY] Matrix size: {dim}x{dim}")
    print(f"[VERIFY] Max absolute error: {max_err:.2e}")
    print("[VERIFY] RESULT: " + ("PASS" if max_err < 1e-6 else "FAIL"))