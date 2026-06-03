#include "instantiate_graph.hpp"

template<typename T>
double max_memory(const Graph<T>& nn, unordered_map<string, double> sizes) {
    using Tptr = unique_ptr<Tensor<T>>;
    using Optr = unique_ptr<Operator<T>>;
    const vector<Tptr>& tens = nn.tensors();

    if (sizes.size() != tens.size()) {
        throw invalid_argument("Size Mismatch in sizes vector.");
    }
    unordered_map<Tensor<T>*, size_t> tens_idx_map;
    size_t num_tens = tens.size();
    for (size_t i = 0; i < num_tens; i++) {
        tens_idx_map[tens[i].get()] = i;
    }

    vector<Operator<T>*> exec_order = nn.topological_sort();
    unordered_map<Operator<T>*, size_t> op_idx_map;
    size_t num_ops = exec_order.size();
    if (num_ops == 0) {
        return 0.0;
    }
    for (size_t i = 0; i < num_ops; i++) {
        op_idx_map[exec_order[i]] = i;
    }
    vector<pair<size_t, size_t>> alive(num_tens, {SIZE_MAX, SIZE_MAX});

    for (Operator<T>* op : exec_order) {
        for (auto& temp : op->inputs()) {
            if (alive[tens_idx_map[temp]].first == SIZE_MAX) {
                alive[tens_idx_map[temp]].first = op_idx_map[op];
            }
            alive[tens_idx_map[temp]].second = op_idx_map[op];
        }
        for (auto& temp : op->outputs()) {
            if (alive[tens_idx_map[temp]].first == SIZE_MAX) {
                alive[tens_idx_map[temp]].first = op_idx_map[op];
            }
        }
    }
    vector<double> memory(num_ops, 0);
    for (size_t i = 0; i < num_tens; i++) {
        auto [l, r] = alive[i];
        if (l == SIZE_MAX) {
            continue;
        }
        if (r == SIZE_MAX) {
            r = l;
        }
        for (size_t j = l; j <= r; j++) {
            memory[j] += sizes.at(tens[i]->name());
        } 
    }
    return *max_element(memory.begin(), memory.end());
}

int main() {
    vector<string> graph_input = {
        "n1: Conv(X, W1) -> A",
        "n2: ReLU(A) -> B",
        "n3: Conv(B, W2) -> C",
        "n4: Conv(X, W3) -> S",
        "n5: Add(C, S) -> D",
        "n6: ReLU(D) -> E",
        "n7: Conv(E, W4) -> F",
        "n8: Add(F, B) -> G",
        "n9: ReLU(G) -> Y"
    };

    unique_ptr<Graph<double>> nn = instantiate_graph<double>(graph_input);
    
    unordered_map<string, double> sizes = {
        {"A", 24},
        {"B", 24},
        {"C", 32},
        {"D", 32},
        {"E", 32},
        {"F", 24},
        {"G", 24},
        {"Y", 24},
        {"S", 32},

        {"X", 0},
        {"W1", 0},
        {"W2", 0},
        {"W3", 0},
        {"W4", 0}
    };
    cout << max_memory((*nn.get()), sizes) <<"\n";
}