#include "2.hpp"

int main() {
    vector<int64_t> a;
    vector<int64_t> b;
    vector<int64_t> out;
    a = {1, 2};
    b = {2, 1};
    out = {2, 2};
    if (out != broadcastShape(a, b)) {
        throw runtime_error("Test Failed");
    }

    a = {6, 7};
    b = {6, 7};
    out = {6, 7};
    if (out != broadcastShape(a, b)) {
        throw runtime_error("Test Failed");
    }

    a = {1, 1, 32, 64};
    b = {6, 7, 1, 1};
    out = {6, 7, 32, 64};
    if (out != broadcastShape(a, b)) {
        throw runtime_error("Test Failed");
    }

    a = {10, 10, 10};
    b = {1, 10};
    out = {10, 10, 10};
    if (out != broadcastShape(a, b)) {
        throw runtime_error("Test Failed");
    }

    a = {10, 6, 7};
    b = {10, 1};
    // invalid
    if (broadcastShape(a, b) != nullopt) {
        throw runtime_error("Test Failed");
    }

    a = {6, 7, 6, 7};
    b = {7, 6};
    //invalid
    if (broadcastShape(a, b) != nullopt) {
        throw runtime_error("Test Failed");
    }

    a  = {2};
    b = {3};
    //invalid
    if (broadcastShape(a, b) != nullopt) {
        throw runtime_error("Test Failed");
    }
    cout << "All Test Passed.\n";
    return 0;
}
