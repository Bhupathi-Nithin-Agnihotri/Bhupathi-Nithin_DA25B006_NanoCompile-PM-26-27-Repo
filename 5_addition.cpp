#include "5.hpp"

int main() {
    size_t n = 0;
    cin >> n;
    vector<double> arr(n, 0);
    for (double &x : arr) {
        cin >> x;
    }

    double sum = 0;
    
    add(arr, 0, n - 1, sum);

    cout << sum << "\n";
    return 0;
}
