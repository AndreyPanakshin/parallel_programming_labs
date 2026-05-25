#!/usr/bin/env python3
"""
Plot results from MPI benchmark.
"""

import csv
import sys

# Пробуем импортировать matplotlib
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("   Warning: matplotlib not installed. Skipping graphs.")
    print("   Install with: pip3 install matplotlib")
    sys.exit(0)

def read_stats(filename):
    """Read statistics from CSV file"""
    sizes = []
    processes = []
    times = []
    
    try:
        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    sizes.append(int(row['size']))
                    processes.append(int(row['processes']))
                    times.append(float(row['time_sec']))
                except (ValueError, KeyError):
                    continue
    except FileNotFoundError:
        print(f"  Error: {filename} not found")
        return [], [], []
    
    return sizes, processes, times

def plot_time_vs_processes(sizes, processes, times):
    """Plot execution time vs number of processes"""
    plt.figure(figsize=(10, 6))
    
    for size in sorted(set(sizes)):
        idx = [i for i, s in enumerate(sizes) if s == size]
        procs = [processes[i] for i in idx]
        t = [times[i] for i in idx]
        plt.plot(procs, t, 'o-', linewidth=2, label=f'Size {size}')
    
    plt.xlabel('Number of Processes')
    plt.ylabel('Execution Time (s)')
    plt.title('MPI: Time vs Processes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('mpi_time_vs_procs.png', dpi=150)
    plt.close()
    print("  Saved: mpi_time_vs_procs.png")

def plot_speedup(sizes, processes, times):
    """Plot speedup vs number of processes"""
    plt.figure(figsize=(10, 6))
    
    for size in sorted(set(sizes)):
        idx = [i for i, s in enumerate(sizes) if s == size]
        procs = [processes[i] for i in idx]
        t = [times[i] for i in idx]
        
        if len(t) > 0 and t[0] > 0:
            speedup = [t[0] / ti if ti > 0 else 0 for ti in t]
            plt.plot(procs, speedup, 's-', linewidth=2, label=f'Size {size}')
    
    # Ideal speedup line
    plt.plot([1, 4], [1, 4], 'k--', alpha=0.5, label='Ideal')
    
    plt.xlabel('Number of Processes')
    plt.ylabel('Speedup')
    plt.title('MPI: Speedup vs Processes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('mpi_speedup.png', dpi=150)
    plt.close()
    print("  Saved: mpi_speedup.png")

if __name__ == "__main__":
    sizes, processes, times = read_stats('stats.csv')
    
    if len(sizes) == 0:
        print("  No data found in stats.csv")
        sys.exit(1)
    
    print(f"Loaded {len(sizes)} data points")
    
    if HAS_MATPLOTLIB:
        plot_time_vs_processes(sizes, processes, times)
        plot_speedup(sizes, processes, times)
        print("\n  Graphs generated successfully!")
    else:
        print("\n   Graphs skipped (matplotlib not installed)")
