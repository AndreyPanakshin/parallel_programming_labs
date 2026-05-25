#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <random>
#include <stdexcept>
#include <cuda_runtime.h>

using namespace std;
using VecFloat = vector<float>;

// Обработка CUDA ошибок
inline void cuda_error_check(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        cerr << "CUDA ERROR: " << cudaGetErrorString(err) 
             << " (" << err << ") at " << file << ":" << line << endl;
        exit(err);
    }
}
#define CUDA_SAFE(call) cuda_error_check(call, __FILE__, __LINE__)

// Заполнение матрицы случайными числами
VecFloat create_random_matrix(int dim, float lo, float hi) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dist(lo, hi);
    
    VecFloat mat(dim * dim);
    for (int i = 0; i < dim * dim; ++i) {
        mat[i] = dist(gen);
    }
    return mat;
}

// Базовое ядро: глобальная память
__global__ void matmul_kernel_global(const float* A, const float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        float acc = 0.0f;
        for (int k = 0; k < n; ++k) {
            acc += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = acc;
    }
}

// Оптимизированное ядро: разделяемая память (блок 16x16)
__global__ void matmul_kernel_shared(const float* A, const float* B, float* C, int n) {
    const int TILE_W = 16;
    
    __shared__ float tileA[TILE_W][TILE_W];
    __shared__ float tileB[TILE_W][TILE_W];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_W + ty;
    int col = bx * TILE_W + tx;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (n + TILE_W - 1) / TILE_W; t++) {
        // Загрузка блока A
        int aCol = t * TILE_W + tx;
        if (row < n && aCol < n)
            tileA[ty][tx] = A[row * n + aCol];
        else
            tileA[ty][tx] = 0.0f;
        
        // Загрузка блока B
        int bRow = t * TILE_W + ty;
        if (bRow < n && col < n)
            tileB[ty][tx] = B[bRow * n + col];
        else
            tileB[ty][tx] = 0.0f;
        
        __syncthreads();
        
        // Умножение блоков
        for (int k = 0; k < TILE_W; k++)
            sum += tileA[ty][k] * tileB[k][tx];
        
        __syncthreads();
    }
    
    if (row < n && col < n)
        C[row * n + col] = sum;
}

int main(int argc, char* argv[]) {
    try {
        if (argc < 4) {
            cerr << "Usage: " << argv[0] 
                 << " DIM BX BY [min_val max_val] [use_shared]" << endl;
            return 1;
        }
        
        int dim = stoi(argv[1]);
        int bx = stoi(argv[2]);
        int by = stoi(argv[3]);
        float min_val = (argc >= 5) ? stof(argv[4]) : 0.0f;
        float max_val = (argc >= 6) ? stof(argv[5]) : 10.0f;
        bool use_shared = (argc >= 7) ? (stoi(argv[6]) != 0) : false;
        
        if (dim > 2000) {
            cerr << "[WARN] Large matrix " << dim << " may exceed VRAM!" << endl;
        }
        
        // Генерация данных
        VecFloat matA = create_random_matrix(dim, min_val, max_val);
        VecFloat matB = create_random_matrix(dim, min_val, max_val);
        VecFloat matC(dim * dim, 0.0f);
        
        size_t mem_bytes = dim * dim * sizeof(float);
        long long total_ops = 2LL * dim * dim * dim;
        
        // Выделение памяти на GPU
        float *dA = nullptr, *dB = nullptr, *dC = nullptr;
        CUDA_SAFE(cudaMalloc(&dA, mem_bytes));
        CUDA_SAFE(cudaMalloc(&dB, mem_bytes));
        CUDA_SAFE(cudaMalloc(&dC, mem_bytes));
        
        // Копирование данных на GPU
        CUDA_SAFE(cudaMemcpy(dA, matA.data(), mem_bytes, cudaMemcpyHostToDevice));
        CUDA_SAFE(cudaMemcpy(dB, matB.data(), mem_bytes, cudaMemcpyHostToDevice));
        
        dim3 block_conf(bx, by);
        dim3 grid_conf((dim + bx - 1) / bx, (dim + by - 1) / by);
        
        // Таймеры
        cudaEvent_t start_ev, stop_ev;
        CUDA_SAFE(cudaEventCreate(&start_ev));
        CUDA_SAFE(cudaEventCreate(&stop_ev));
        
        // Прогрев
        if (use_shared) {
            matmul_kernel_shared<<<grid_conf, block_conf>>>(dA, dB, dC, dim);
        } else {
            matmul_kernel_global<<<grid_conf, block_conf>>>(dA, dB, dC, dim);
        }
        CUDA_SAFE(cudaDeviceSynchronize());
        
        // Основной замер
        CUDA_SAFE(cudaEventRecord(start_ev));
        if (use_shared) {
            matmul_kernel_shared<<<grid_conf, block_conf>>>(dA, dB, dC, dim);
        } else {
            matmul_kernel_global<<<grid_conf, block_conf>>>(dA, dB, dC, dim);
        }
        CUDA_SAFE(cudaEventRecord(stop_ev));
        CUDA_SAFE(cudaEventSynchronize(stop_ev));
        
        float elapsed_ms = 0.0f;
        CUDA_SAFE(cudaEventElapsedTime(&elapsed_ms, start_ev, stop_ev));
        
        // Обратное копирование
        CUDA_SAFE(cudaMemcpy(matC.data(), dC, mem_bytes, cudaMemcpyDeviceToHost));
        
        // Вывод результатов
        cout << "Matrix dimension: " << dim << "x" << dim << endl;
        cout << "Block size: (" << bx << ", " << by << ")" << endl;
        cout << "Memory mode: " << (use_shared ? "shared" : "global") << endl;
        cout << "Execution time: " << elapsed_ms << " ms" << endl;
        cout << "Arithmetic ops: " << total_ops << endl;
        
        // Освобождение ресурсов
        CUDA_SAFE(cudaEventDestroy(start_ev));
        CUDA_SAFE(cudaEventDestroy(stop_ev));
        CUDA_SAFE(cudaFree(dA));
        CUDA_SAFE(cudaFree(dB));
        CUDA_SAFE(cudaFree(dC));
        
        return 0;
        
    } catch (const exception& e) {
        cerr << "[ERROR] " << e.what() << endl;
        return 1;
    }
}