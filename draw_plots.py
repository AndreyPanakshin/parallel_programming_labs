import os
import matplotlib.pyplot as plt

sizes = [200, 400, 800, 1200, 1600, 2000]

times_1 = [82.32, 692.416, 5063.61, 17401.4, 55374, 105364]
times_2 = [41.8279, 346.735, 2656.55, 9037.3, 27975.7, 52809.1]
times_4 = [20.9632, 175.4, 1555.51, 5401.98, 14035.8, 26745.6]
times_8 = [10.664, 88.644, 925.443, 2785.75, 7213.48, 13654.7]

speedup_2 = [t1 / t2 for t1, t2 in zip(times_1, times_2)]
speedup_4 = [t1 / t4 for t1, t4 in zip(times_1, times_4)]
speedup_8 = [t1 / t8 for t1, t8 in zip(times_1, times_8)]

os.makedirs("images", exist_ok=True)

plt.figure(figsize=(10, 6))
plt.plot(sizes, times_1, marker='o', label='1 процесс')
plt.plot(sizes, times_2, marker='o', label='2 процесса')
plt.plot(sizes, times_4, marker='o', label='4 процесса')
plt.plot(sizes, times_8, marker='o', label='8 процессов')
plt.xlabel('Размер матрицы')
plt.ylabel('Время выполнения, мс')
plt.title('Время выполнения MPI-программы на суперкомпьютере')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('images/time_vs_size.jpg', format='jpg', dpi=300)
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(sizes, speedup_2, marker='o', label='S(2)')
plt.plot(sizes, speedup_4, marker='o', label='S(4)')
plt.plot(sizes, speedup_8, marker='o', label='S(8)')
plt.xlabel('Размер матрицы')
plt.ylabel('Ускорение')
plt.title('Ускорение MPI-программы на суперкомпьютере')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('images/speedup_vs_size.jpg', format='jpg', dpi=300)
plt.close()

print("Plots saved:")
print("images/time_vs_size.jpg")
print("images/speedup_vs_size.jpg")