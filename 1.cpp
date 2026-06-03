#include "instantiate_graph.hpp"

int main() {
    vector<string> graph_input = {
        "n1: Conv(x, w1) -> a",
        "n2: ReLU(a) -> b",
        "n3: Conv(b, w2) -> c",
        "n4: Conv(x, w_skip) -> s",
        "n5: Add(c, s) -> d",
        "n6: ReLU(d) -> e",
        "n7: Mul(e, scale) -> f",
        "n8: Add(f, bias) -> y"
    };

    unordered_map<string, double> costs = {
        {"n1", 8},
        {"n3", 10},
        {"n4", 5}
    };

    unique_ptr<Graph<double>> nn = instantiate_graph<double>(graph_input);
    const vector<unique_ptr<Operator<double>>>& ops = nn->operators();
    for (const unique_ptr<Operator<double>>& op : ops) {
        auto it = costs.find(op->name());

        if (it != costs.end()) {
            op->change_cost(it->second);
        }
    }
    
    cout << nn.get()->get_fastest_execution() << "\n";
    nn.get()->print_graph();
    return 0;
}