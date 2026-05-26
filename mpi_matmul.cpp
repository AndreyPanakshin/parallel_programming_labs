#include <mpi.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>
#include <cstdlib>

using namespace std;

vector<double> readMatrix(const string& filename, int& n) {
    ifstream fin(filename);
    if (!fin) {
        cerr << "Error: cannot open " << filename << endl;
        exit(1);
    }

    fin >> n;
    if (n <= 0) {
        cerr << "Error: wrong matrix size in " << filename << endl;
        exit(1);
    }

    vector<double> a(n * n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            fin >> a[i * n + j];
        }
    }

    fin.close();
    return a;
}

void writeMatrix(const string& filename, const vector<double>& c, int n) {
    ofstream fout(filename);
    fout << n << endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            fout << fixed << setprecision(6) << c[i * n + j];
            if (j != n - 1) {
                fout << " ";
            }
        }
        fout << endl;
    }
    fout.close();
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 0;
    vector<double> A, B, C;

    if (rank == 0) {
        int n1, n2;
        A = readMatrix("matrix_a.txt", n1);
        B = readMatrix("matrix_b.txt", n2);

        if (n1 != n2) {
            cerr << "Error: matrix sizes are different" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        n = n1;
        C.resize(n * n, 0.0);
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        B.resize(n * n);
    }

    MPI_Bcast(B.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    vector<int> rows_per_proc(size), sendcounts(size), displs(size);
    int base_rows = n / size;
    int remainder = n % size;

    for (int p = 0; p < size; p++) {
        rows_per_proc[p] = base_rows + (p < remainder ? 1 : 0);
        sendcounts[p] = rows_per_proc[p] * n;
    }

    displs[0] = 0;
    for (int p = 1; p < size; p++) {
        displs[p] = displs[p - 1] + sendcounts[p - 1];
    }

    int local_rows = rows_per_proc[rank];
    vector<double> local_A(local_rows * n);
    vector<double> local_C(local_rows * n, 0.0);

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    MPI_Scatterv(
        rank == 0 ? A.data() : nullptr,
        sendcounts.data(),
        displs.data(),
        MPI_DOUBLE,
        local_A.data(),
        local_rows * n,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += local_A[i * n + k] * B[k * n + j];
            }
            local_C[i * n + j] = sum;
        }
    }

    MPI_Gatherv(
        local_C.data(),
        local_rows * n,
        MPI_DOUBLE,
        rank == 0 ? C.data() : nullptr,
        sendcounts.data(),
        displs.data(),
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    MPI_Barrier(MPI_COMM_WORLD);
    double finish = MPI_Wtime();

    if (rank == 0) {
        double time_ms = (finish - start) * 1000.0;
        writeMatrix("result.txt", C, n);

        long long volume = 1LL * n * n * n + 1LL * n * n * (n - 1);

        cout << "MPI matrix multiplication" << endl;
        cout << "Processes: " << size << endl;
        cout << "Matrix size: " << n << "x" << n << endl;
        cout << "Execution time (ms): " << time_ms << endl;
        cout << "Task volume: " << volume << endl;
        cout << "Result file: result.txt" << endl;
    }

    MPI_Finalize();
    return 0;
}
