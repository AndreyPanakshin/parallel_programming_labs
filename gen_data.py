import random
import sys

def write_matrix(filename, n):
    f = open(filename, "w")
    f.write(str(n) + "\n")
    for i in range(n):
        row = []
        for j in range(n):
            row.append(str(round(random.uniform(0, 10), 6)))
        f.write(" ".join(row) + "\n")
    f.close()

if len(sys.argv) < 2:
    print("Usage: python generate_matrices.py <size>")
    sys.exit(1)

n = int(sys.argv[1])
write_matrix("matrix_a.txt", n)
write_matrix("matrix_b.txt", n)

print("Matrices generated:", n, "x", n)
