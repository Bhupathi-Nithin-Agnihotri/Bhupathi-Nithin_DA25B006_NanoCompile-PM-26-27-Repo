#include <optional>
#include <chrono>
#include "2.hpp"
using namespace std;

int main() {
    vector<size_t> test_dimensions = {1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000};

    for (size_t d : test_dimensions) {
        vector<int64_t> a(d, 1), b(d, 1); 

        auto start = chrono::high_resolution_clock::now();
        for (int it = 0; it < 100000; it++) {
            auto res = broadcastShape(a, b);
        }

        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double, nano> total_time = end - start;
        double avg_time_ns = total_time.count() / 100000;
        cout << "Dim: " << d << " -- " << avg_time_ns << " ns\n";
    }
    return 0;
}
