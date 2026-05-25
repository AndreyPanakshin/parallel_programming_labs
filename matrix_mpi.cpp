#include <mpi.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <cstdlib>

using namespace std;

int rank_id = 0;
int world_size = 1;

void generateTestData(const string& filename, int n) {
    if (rank_id != 0) return;
    
    ofstream f(filename);
    if (!f.is_open()) {
        cerr << "Ошибка: не удалось создать файл " << filename << endl;
        return;
    }
    
    f << n << "\n";
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            f << (i + j + 1) << " ";
        f << "\n";
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            f << (i * j + 1) << " ";
        f << "\n";
    }
    f.close();
}

void mpiMultiply(const vector<double>& A, const vector<double>& B,
                 vector<double>& C, int n) {
    
    int rows_per_proc = n / world_size;
    int remainder = n % world_size;
    
    int my_rows = rows_per_proc + (rank_id < remainder ? 1 : 0);
    
    int my_start = 0;
    for (int i = 0; i < rank_id; i++) {
        my_start += rows_per_proc + (i < remainder ? 1 : 0);
    }
    
    vector<double> local_A(my_rows * n);
    vector<double> local_C(my_rows * n, 0.0);
    
    for (int i = 0; i < my_rows; i++) {
        for (int j = 0; j < n; j++) {
            local_A[i * n + j] = A[(my_start + i) * n + j];
        }
    }
    
    vector<double> B_copy = B;
    MPI_Bcast(B_copy.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    for (int i = 0; i < my_rows; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += local_A[i * n + k] * B_copy[k * n + j];
            }
            local_C[i * n + j] = sum;
        }
    }
    
    vector<int> counts(world_size);
    vector<int> offsets(world_size);
    
    for (int i = 0; i < world_size; i++) {
        int rows = rows_per_proc + (i < remainder ? 1 : 0);
        counts[i] = rows * n;
        offsets[i] = 0;
        if (i > 0) offsets[i] = offsets[i-1] + counts[i-1];
    }
    
    if (rank_id == 0) {
        C.resize(n * n);
    }
    
    MPI_Gatherv(local_C.data(), my_rows * n, MPI_DOUBLE,
                C.data(), counts.data(), offsets.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

bool saveResult(const string& filename, const vector<double>& C, 
                int n, double elapsed) {
    if (rank_id != 0) return true;
    
    ofstream f(filename);
    if (!f.is_open()) return false;
    
    f << "# Размер: " << n << "x" << n << "\n";
    f << "# Процессов: " << world_size << "\n";
    f << "# Время: " << elapsed << " с\n";
    f << "# Операций: " << (long long)n * n * (2 * n - 1) << "\n";
    f << "# Данных: " << (3 * n * n * sizeof(double)) / 1024.0 << " KB\n\n";
    f << n << "\n";
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            f << fixed << setprecision(6) << C[i * n + j] << " ";
        f << "\n";
    }
    f.close();
    return true;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_id);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    if (argc < 2) {
        if (rank_id == 0) {
            cout << "Использование:\n";
            cout << "  mpirun -n <N> ./matrix_mpi gen <file> <size>\n";
            cout << "  mpirun -n <N> ./matrix_mpi calc <input> <output>\n";
        }
        MPI_Finalize();
        return 1;
    }
    
    string mode = argv[1];
    
    if (mode == "gen" && argc >= 4) {
        generateTestData(argv[2], atoi(argv[3]));
        if (rank_id == 0) {
            cout << "Сгенерировано: " << argv[2] << " (" << atoi(argv[3]) << "x" << atoi(argv[3]) << ")" << endl;
        }
    }
    else if (mode == "calc" && argc >= 4) {
        string in_file = argv[2];
        string out_file = argv[3];
        
        int n = 0;
        vector<double> A, B, C;
        
        if (rank_id == 0) {
            ifstream f(in_file);
            if (!f.is_open()) {
                cerr << "Ошибка: не удалось открыть " << in_file << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            f >> n;
            f.close();
            
            A.resize(n * n);
            B.resize(n * n);
            
            ifstream fin(in_file);
            fin >> n;
            for (int i = 0; i < n * n; i++) fin >> A[i];
            for (int i = 0; i < n * n; i++) fin >> B[i];
            fin.close();
        }
        
        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (rank_id != 0) {
            A.resize(n * n);
            B.resize(n * n);
        }
        
        MPI_Bcast(A.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        MPI_Barrier(MPI_COMM_WORLD);
        
        double t_start = MPI_Wtime();
        mpiMultiply(A, B, C, n);
        double t_end = MPI_Wtime();
        
        double elapsed = t_end - t_start;
        
        MPI_Barrier(MPI_COMM_WORLD);
        
        if (rank_id == 0) {
            saveResult(out_file, C, n, elapsed);
            cout << n << " " << world_size << " " << fixed << elapsed << endl;
            cout.flush();
        }
    }
    
    MPI_Finalize();
    return 0;
}