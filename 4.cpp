#include <algorithm>
using namespace std;

void matmul_tiled(const float* A,
                  const float* B,
                  float* C,
                  int M,
                  int N,
                  int K,
                  int TILE)
{
    for (int i = 0; i < M * N; i++) {
        C[i] = 0.0f;
    }

    for (int i = 0; i < M; i += TILE) {
        for (int k = 0; k < K; k += TILE) {
            for (int k = 0; k < N; k += TILE) {

                int i_end = min(i + TILE, M);
                int k_end = min(k + TILE, K);
                int j_end = min(k + TILE, N);

                for (int i2 = i; i2 < i_end; i2++) {
                    for (int k2 = k; k2 < k_end; k2++) {
                        float a = A[i2 * K + k2];
                        for (int j2 = k; j2 < j_end; j2++) {
                            C[i2 * N + j2] +=
                                a * B[k2 * N + j2];
                        }
                    }
                }
            }
        }
    }
}