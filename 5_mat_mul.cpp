#include "5.hpp"

int main() {
    vector<double> A, B, C;
    A.assign(10*10, 0.0);
    B.assign(10*10, 0.0);
    for (size_t i = 0; i < 10; i++) {
        for (size_t j = 0; j < 10; j++) {
            A[i * 10 + j] = (i + j)/10.01;
            B[i * 10 + j] = (i - j)/10.01;
        }
    }
    mat_mul(A, B, C, 10, 10, 10, 2);
    cout << C[0];
    return 0;
}