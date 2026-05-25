#!/usr/bin/env python3
"""
Генерация графиков для CUDA лабораторной работы #4
Формат CSV: size,block_x,block_y,use_shared,time_ms,operations,data_kb
"""

import csv
import matplotlib.pyplot as plt
import numpy as np

def load_data(filename='results.csv'):
    """Загрузка статистики из CSV"""
    data = {
        'size': [],
        'bx': [],
        'by': [],
        'shared': [],
        'time': [],
        'ops': [],
        'data_kb': []
    }
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    data['size'].append(int(row['size']))
                    data['bx'].append(int(row['block_x']))
                    data['by'].append(int(row['block_y']))
                    data['shared'].append(int(row['use_shared']))
                    data['time'].append(float(row['time_ms']))
                    data['ops'].append(int(row['operations']))
                    data['data_kb'].append(int(row['data_kb']))
                except (ValueError, KeyError) as e:
                    print(f"Skipping row: {e}")
                    continue
    except FileNotFoundError:
        print(f"Error: {filename} not found!")
        return None
    
    return data

def chart_time_vs_size(data, out_file='cuda_time_plot.png'):
    """График: время выполнения от размера матрицы"""
    plt.figure(figsize=(11, 6))
    
    sizes = sorted(set(data['size']))
    block_configs = sorted(set(zip(data['bx'], data['by'])))
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    
    for i, (bw, bh) in enumerate(block_configs):
        idx = [j for j in range(len(data['size'])) 
               if data['bx'][j] == bw and data['by'][j] == bh and data['shared'][j] == 0]
        
        sz_sub = [data['size'][j] for j in idx]
        tm_sub = [data['time'][j] for j in idx]
        
        if sz_sub:
            plt.plot(sz_sub, tm_sub, 'o-', linewidth=2, 
                     color=colors[i % len(colors)],
                     label=f'{bw}x{bh} (global)', markersize=8)
    
    plt.xlabel('Matrix size (N x N)', fontsize=12)
    plt.ylabel('Execution time (ms)', fontsize=12)
    plt.title('CUDA: Execution time vs Matrix size', fontsize=14)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xticks(sizes)
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"Saved: {out_file}")

def chart_speedup(data, out_file='cuda_speedup_plot.png'):
    """График: ускорение относительно блока 8x8"""
    plt.figure(figsize=(11, 6))
    
    sizes = sorted(set(data['size']))
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
    
    for i, sz in enumerate(sizes):
        idx = [j for j, s in enumerate(data['size']) if s == sz and data['shared'][j] == 0]
        
        if not idx:
            continue
            
        # Находим базовое время для 8x8
        base_idx = None
        for j in idx:
            if data['bx'][j] == 8 and data['by'][j] == 8:
                base_idx = j
                break
        
        if base_idx is None:
            continue
            
        base_time = data['time'][base_idx]
        
        configs = [f"{data['bx'][j]}x{data['by'][j]}" for j in idx]
        times = [data['time'][j] for j in idx]
        speedups = [base_time / t if t > 0 else 0 for t in times]
        
        xpos = range(len(configs))
        plt.plot(xpos, speedups, 'o-', linewidth=2, 
                 color=colors[i % len(colors)], 
                 label=f'{sz}x{sz}', markersize=8)
    
    plt.xlabel('Block configuration', fontsize=12)
    plt.ylabel('Speedup', fontsize=12)
    plt.title('CUDA: Speedup relative to 8x8 block', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.xticks(range(len(configs)), configs, rotation=45)
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"Saved: {out_file}")

def chart_shared_comparison(data, out_file='cuda_memory_compare.png'):
    """Сравнение производительности shared vs global памяти"""
    plt.figure(figsize=(10, 6))
    
    sizes = sorted(set(data['size']))
    
    global_times = []
    shared_times = []
    
    for sz in sizes:
        idx_global = [j for j, s in enumerate(data['size']) 
                     if s == sz and data['shared'][j] == 0 and data['bx'][j] == 16 and data['by'][j] == 16]
        idx_shared = [j for j, s in enumerate(data['size']) 
                     if s == sz and data['shared'][j] == 1 and data['bx'][j] == 16 and data['by'][j] == 16]
        
        global_times.append(data['time'][idx_global[0]] if idx_global else 0)
        shared_times.append(data['time'][idx_shared[0]] if idx_shared else 0)
    
    xpos = range(len(sizes))
    width = 0.35
    
    plt.bar([x - width/2 for x in xpos], global_times, width, label='Global memory', color='#3498db', alpha=0.8)
    plt.bar([x + width/2 for x in xpos], shared_times, width, label='Shared memory', color='#2ecc71', alpha=0.8)
    
    plt.xlabel('Matrix size', fontsize=12)
    plt.ylabel('Execution time (ms)', fontsize=12)
    plt.title('CUDA: Global vs Shared memory (16x16 block)', fontsize=14)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(xpos, sizes)
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"Saved: {out_file}")

def chart_gflops(data, out_file='cuda_gflops_plot.png'):
    """График производительности в GFLOPS"""
    plt.figure(figsize=(11, 6))
    
    sizes = sorted(set(data['size']))
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    
    for i, sz in enumerate(sizes):
        idx = [j for j, s in enumerate(data['size']) if s == sz and data['shared'][j] == 0]
        
        if not idx:
            continue
        
        block_labels = [f"{data['bx'][j]}x{data['by'][j]}" for j in idx]
        times = [data['time'][j] for j in idx]
        ops_arr = [data['ops'][j] for j in idx]
        
        gflops_vals = [(op / (t / 1000 * 1e9)) if t > 0 else 0 for op, t in zip(ops_arr, times)]
        
        xpos = range(len(block_labels))
        plt.plot(xpos, gflops_vals, 's-', linewidth=2, 
                 color=colors[i % len(colors)], 
                 label=f'{sz}x{sz}', markersize=8)
    
    plt.xlabel('Block configuration', fontsize=12)
    plt.ylabel('Performance (GFLOPS)', fontsize=12)
    plt.title('CUDA: Matrix multiplication performance', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.xticks(xpos, block_labels, rotation=45)
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"Saved: {out_file}")

def print_summary(data):
    """Текстовый анализ результатов"""
    print("\n" + "="*70)
    print("CUDA BENCHMARK SUMMARY (GTX 1060 3GB)")
    print("="*70)
    
    sizes = sorted(set(data['size']))
    
    for sz in sizes:
        idx = [j for j, s in enumerate(data['size']) if s == sz and data['shared'][j] == 0]
        
        if not idx:
            continue
            
        print(f"\nMatrix {sz}x{sz} (global memory):")
        print("-"*60)
        print(f"{'Block':<12} {'Threads':<10} {'Time (ms)':<15} {'GFLOPS':<12}")
        print("-"*60)
        
        for j in idx:
            bw, bh = data['bx'][j], data['by'][j]
            threads = bw * bh
            t_ms = data['time'][j]
            ops_cnt = data['ops'][j]
            gflops_val = (ops_cnt / (t_ms / 1000 * 1e9)) if t_ms > 0 else 0
            print(f"{bw}x{bh:<11} {threads:<10} {t_ms:<15.3f} {gflops_val:<12.2f}")

def main():
    print("Loading results.csv...")
    data = load_data()
    
    if data is None or len(data['size']) == 0:
        print("No data found in results.csv")
        return 1
    
    print(f"Loaded {len(data['size'])} records")
    print(f"  Sizes: {sorted(set(data['size']))}")
    print(f"  Blocks: {sorted(set(zip(data['bx'], data['by'])))}")
    
    print("\nGenerating charts...")
    print("-"*50)
    
    chart_time_vs_size(data)
    chart_speedup(data)
    chart_shared_comparison(data)
    chart_gflops(data)
    
    print_summary(data)
    
    print("\n" + "="*70)
    print("ALL CHARTS GENERATED SUCCESSFULLY!")
    print("="*70)
    print("\nOutput files:")
    print("  1. cuda_time_plot.png       - Execution time vs matrix size")
    print("  2. cuda_speedup_plot.png    - Speedup comparison")
    print("  3. cuda_memory_compare.png  - Shared vs Global memory")
    print("  4. cuda_gflops_plot.png     - GFLOPS performance")
    
    return 0

if __name__ == "__main__":
    exit(main())