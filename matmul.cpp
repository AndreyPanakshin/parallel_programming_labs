#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <omp.h>

using std::vector;
using std::string;
using std::cout;
using std::cerr;
using std::endl;
using std::fixed;
using std::setprecision;

using Clock = std::chrono::high_resolution_clock;
using Seconds = std::chrono::duration<double>;

class OMPMatrixProcessor {
private:
    vector<vector<double>> A, B, C;
    int N;

public:
    bool loadData(const string& path) {
        std::ifstream in(path);
        if (!in.is_open()) {
            cerr << "[ERROR] Cannot open: " << path << endl;
            return false;
        }
        in >> N;
        A.assign(N, vector<double>(N));
        B.assign(N, vector<double>(N));
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                in >> A[i][j];
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                in >> B[i][j];
        in.close();
        return true;
    }

    void sequentialMultiply() {
        C.assign(N, vector<double>(N, 0.0));
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                for (int k = 0; k < N; ++k) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
    }

    void parallelMultiply(int threadCount) {
        C.assign(N, vector<double>(N, 0.0));
        omp_set_num_threads(threadCount);
        
        #pragma omp parallel for collapse(2) schedule(static)
        for (int row = 0; row < N; ++row) {
            for (int col = 0; col < N; ++col) {
                double accumulator = 0.0;
                for (int k = 0; k < N; ++k) {
                    accumulator += A[row][k] * B[k][col];
                }
                C[row][col] = accumulator;
            }
        }
    }

    void saveResult(const string& path, double elapsed, int threads) const {
        std::ofstream out(path);
        if (!out.is_open()) return;
        
        out << "# Dimension: " << N << "x" << N << "\n";
        out << "# Threads: " << threads << "\n";
        out << "# Execution time: " << elapsed << " sec\n";
        long long ops = (long long)N * N * (2LL * N - 1LL);
        double dataKB = (3.0 * N * N * sizeof(double)) / 1024.0;
        out << "# Operations: " << ops << "\n";
        out << "# Data volume: " << dataKB << " KB\n\n";
        out << N << "\n";
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                out << fixed << setprecision(6) << C[i][j] << " ";
            }
            out << "\n";
        }
        out.close();
    }

    int getDimension() const { return N; }
};

void generateTestData(const string& path, int dim) {
    std::ofstream out(path);
    out << dim << "\n";
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j)
            out << (i + j + 1) << " ";
        out << "\n";
    }
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j)
            out << (i * j + 1) << " ";
        out << "\n";
    }
    out.close();
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cout << "Usage:\n";
        cout << "  " << argv[0] << " gen <file> <size>\n";
        cout << "  " << argv[0] << " par <input> <output> [threads]\n";
        cout << "  " << argv[0] << " seq <input> <output>\n";
        return 1;
    }

    string mode = argv[1];
    if (mode == "gen" && argc >= 4) {
        generateTestData(argv[2], std::atoi(argv[3]));
        cout << "[OK] Generated: " << argv[2] << endl;
    }
    else if (mode == "par" && argc >= 4) {
        int threads = (argc >= 5) ? std::atoi(argv[4]) : 4;
        OMPMatrixProcessor proc;
        if (!proc.loadData(argv[2])) return 1;
        
        auto start = Clock::now();
        proc.parallelMultiply(threads);
        auto end = Clock::now();
        double elapsed = Seconds(end - start).count();
        
        proc.saveResult(argv[3], elapsed, threads);
        cout << proc.getDimension() << " " << threads << " " << elapsed << endl;
    }
    else if (mode == "seq" && argc >= 4) {
        OMPMatrixProcessor proc;
        if (!proc.loadData(argv[2])) return 1;
        
        auto start = Clock::now();
        proc.sequentialMultiply();
        auto end = Clock::now();
        double elapsed = Seconds(end - start).count();
        
        proc.saveResult(argv[3], elapsed, 1);
        cout << proc.getDimension() << " 1 " << elapsed << endl;
    }
    else {
        cerr << "[ERROR] Unknown command" << endl;
        return 1;
    }
    return 0;
}