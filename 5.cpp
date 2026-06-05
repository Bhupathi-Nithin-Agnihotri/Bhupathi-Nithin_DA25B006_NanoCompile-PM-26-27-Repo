#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
using namespace std;

mutex mtx;

void add(const vector<double>& arr, size_t l, size_t r, double& sum, size_t& num_ele) {
    if (l == r) {
        lock_guard<mutex> lock(mtx);
        sum += arr[l];
        num_ele++;
        return;
    }
    size_t mid = l + (r - l) / 2;
    thread t1(add, ref(arr), l, mid, ref(sum), ref(num_ele));
    thread t2(add, ref(arr), mid + 1, r, ref(sum), ref(num_ele));
    t1.join();
    t2.join();
}

int main() {
    size_t n = 0;
    cin >> n;
    vector<double> arr(n, 0);
    for (double &x : arr) {
        cin >> x;
    }

    double sum = 0;
    size_t num_ele = 0;
    
    add(arr, 0, n - 1, sum, num_ele);

    cout << sum << "\n";
    return 0;
}
