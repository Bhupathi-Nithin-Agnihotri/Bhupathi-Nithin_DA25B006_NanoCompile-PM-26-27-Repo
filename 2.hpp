#pragma once
#include <iostream>
#include <vector>
#include <optional>
#include <cstdint>
#include <algorithm> 

using namespace std;

optional<vector<int64_t>> broadcastShape(const vector<int64_t>& a, const vector<int64_t>& b) {
    size_t size_a = a.size();
    size_t size_b = b.size();
    size_t max_size = max(size_a, size_b);
    
    vector<int64_t> res(max_size);
    
    for (size_t i = 0; i < max_size; ++i) {
        int64_t dim_a = (i < size_a) ? a[size_a - 1 - i] : 1;
        int64_t dim_b = (i < size_b) ? b[size_b - 1 - i] : 1;
        
        if (dim_a == dim_b) {
            res[max_size - 1 - i] = dim_a;
        }
        else if (dim_a == 1 || dim_b == 1) {
            res[max_size - 1 - i] = max(dim_b, dim_a);
        }
        else {
            return nullopt; 
        }
    }
    
    return res;
}