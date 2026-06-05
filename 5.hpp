#pragma once
#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <algorithm>
using namespace std;

inline mutex mtx;

void add(const vector<double>& arr, size_t l, size_t r, double& sum) {
    if (l == r) {
        lock_guard<mutex> lock(mtx);
        sum += arr[l];
        return;
    }
    size_t mid = l + (r - l) / 2;

    if ((r - l) > 10000) {
        thread t1(add, ref(arr), l, mid, ref(sum));
        thread t2(add, ref(arr), mid + 1, r, ref(sum));
        t1.join();
        t2.join();
    }
    else {
        add(arr, l, mid, sum);
        add(arr, mid + 1, r, sum);
    }
}

void mat_mul_one_tile(const vector<double>& A, const vector<double>& B, vector<double>& C, size_t M, size_t K, size_t N, size_t BLOCK_SIZE, size_t si, size_t sj) {
    for (size_t sk = 0; sk < K; sk += BLOCK_SIZE) {
        for (size_t i = si; i < min(si + BLOCK_SIZE, M); ++i) {
            for (size_t k = sk; k < min(sk + BLOCK_SIZE, K); ++k) {
                double r = A[i * K + k]; 
                size_t temp_c = i * N;
                size_t temp_b = k * N;

                for (size_t j = sj; j < min(sj + BLOCK_SIZE, N); ++j) {
                    C[temp_c + j] += r * B[temp_b + j]; 
                }
            }
        }
    }
}

void mat_mul(const vector<double>& A, const vector<double>& B, vector<double>& C, size_t M, size_t K, size_t N, size_t BLOCK_SIZE) {
    if (A.empty() || B.empty()) {
        throw invalid_argument("Matrices cannot be empty.");
    }

    C.assign(M * N, 0.0);
    vector<thread> workers;

    size_t max_threads = thread::hardware_concurrency();
    if (max_threads == 0) {
        max_threads = 4;
    }
    for (size_t si = 0; si < M; si += BLOCK_SIZE) {
        for (size_t sj = 0; sj < N; sj += BLOCK_SIZE) {
            if (workers.size() >= max_threads) {
                bool slot_freed = false;
                while (!slot_freed) {
                    for (auto it = workers.begin(); it != workers.end(); ++it) {
                        if (it->joinable()) {
                            it->join();
                            workers.erase(it);
                            slot_freed = true;
                            break;
                        }
                    }
                }
            }

            workers.push_back(thread(mat_mul_one_tile, ref(A), ref(B), ref(C), M, K, N, BLOCK_SIZE, si, sj));
        }
    }

    for (thread& t : workers) {
        if (t.joinable()) {
            t.join();
        }
    }
}