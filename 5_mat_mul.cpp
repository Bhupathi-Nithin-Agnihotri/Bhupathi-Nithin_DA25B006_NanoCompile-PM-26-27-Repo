#include <chrono>
#include "5.hpp"

int main() {
    constexpr size_t M = 1024;
    constexpr size_t K = 1024;
    constexpr size_t N = 1024;
    constexpr size_t BLOCK_SIZE = 64;

    vector<double> A(M * K);
    vector<double> B(K * N);

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < K; ++j) {
            A[i * K + j] = (i + j) / 100.0;
        }
    }

    for (size_t i = 0; i < K; ++i) {
        for (size_t j = 0; j < N; ++j) {
            B[i * N + j] = (i - j) / 100.0;
        }
    }

    vector<double> C1, C2;
    auto start1 = chrono::high_resolution_clock::now();
    mat_mul_naive(A, B, C1, M, K, N);
    auto end1 = chrono::high_resolution_clock::now();

    auto start2 = chrono::high_resolution_clock::now();
    mat_mul(A, B, C2, M, K, N, BLOCK_SIZE);
    auto end2 = chrono::high_resolution_clock::now();

    double naive_time = chrono::duration<double>(end1 - start1).count();
    double tiled_time = chrono::duration<double>(end2 - start2).count();

    cout << "Naive Time: " << naive_time << " sec\n";
    cout << "Tiled Time: " << tiled_time << " sec\n";
    cout << "Speedup: " << naive_time / tiled_time << "x\n";

    double max_error = 0.0;

    for (size_t i = 0; i < C1.size(); ++i) {
        max_error = max(max_error, abs(C1[i] - C2[i]));
    }
    cout << "Max Error  : " << max_error << "\n";
    return 0;
}