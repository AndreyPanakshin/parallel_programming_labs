#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>

using std::vector;
using std::string;
using std::cout;
using std::cerr;
using std::endl;
using std::fixed;
using std::setprecision;

using Clock = std::chrono::high_resolution_clock;
using Seconds = std::chrono::duration<double>;

class MatrixProcessor {
private:
    vector<vector<double>> matA;
    vector<vector<double>> matB;
    vector<vector<double>> matC;
    int dim;

public:
    bool load(const string& path) {
        std::ifstream in(path);
        if (!in.is_open()) {
            cerr << "[ERROR] Cannot open: " << path << endl;
            return false;
        }
        in >> dim;
        matA.assign(dim, vector<double>(dim));
        matB.assign(dim, vector<double>(dim));
        for (int i = 0; i < dim; ++i)
            for (int j = 0; j < dim; ++j)
                in >> matA[i][j];
        for (int i = 0; i < dim; ++i)
            for (int j = 0; j < dim; ++j)
                in >> matB[i][j];
        in.close();
        return true;
    }

    void compute() {
        matC.assign(dim, vector<double>(dim, 0.0));
        for (int row = 0; row < dim; ++row) {
            for (int col = 0; col < dim; ++col) {
                double acc = 0.0;
                for (int k = 0; k < dim; ++k) {
                    acc += matA[row][k] * matB[k][col];
                }
                matC[row][col] = acc;
            }
        }
    }

    void save(const string& path, double elapsed) const {
        std::ofstream out(path);
        if (!out.is_open()) return;
        out << "# Dimension: " << dim << "x" << dim << "\n";
        out << "# Execution time: " << elapsed << " sec\n";
        long long ops = (long long)dim * dim * (2LL * dim - 1LL);
        double dataKB = (3.0 * dim * dim * sizeof(double)) / 1024.0;
        out << "# Operations: " << ops << "\n";
        out << "# Data volume: " << dataKB << " KB\n\n";
        out << dim << "\n";
        for (int i = 0; i < dim; ++i) {
            for (int j = 0; j < dim; ++j) {
                out << fixed << setprecision(6) << matC[i][j] << " ";
            }
            out << "\n";
        }
        out.close();
    }

    int dimension() const { return dim; }
};

void generateData(const string& path, int n) {
    std::ofstream out(path);
    out << n << "\n";
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j)
            out << (i + j + 1) << " ";
        out << "\n";
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j)
            out << (i * j + 1) << " ";
        out << "\n";
    }
    out.close();
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cout << "Usage:\n";
        cout << "  " << argv[0] << " gen <file> <size>\n";
        cout << "  " << argv[0] << " exec <input> <output>\n";
        return 1;
    }

    string mode = argv[1];
    if (mode == "gen" && argc >= 4) {
        generateData(argv[2], std::atoi(argv[3]));
        cout << "[OK] Generated: " << argv[2] << endl;
    }
    else if (mode == "exec" && argc >= 4) {
        MatrixProcessor proc;
        if (!proc.load(argv[2])) return 1;
        auto start = Clock::now();
        proc.compute();
        auto end = Clock::now();
        double elapsed = Seconds(end - start).count();
        proc.save(argv[3], elapsed);
        cout << proc.dimension() << " " << elapsed << endl;
    }
    else {
        cerr << "[ERROR] Unknown command" << endl;
        return 1;
    }
    return 0;
}